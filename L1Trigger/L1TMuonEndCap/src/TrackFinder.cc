#include "L1Trigger/L1TMuonEndCap/interface/TrackFinder.h"

#include <iostream>
#include <sstream>

#include "L1Trigger/L1TMuonEndCap/interface/EMTFSubsystemCollector.h"


TrackFinder::TrackFinder(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iConsumes) :
    geometry_translator_(),
    condition_helper_(),
    sector_processor_lut_(),
    pt_assign_engine_(),
    sector_processors_(),
    config_(iConfig),
    tokenCSC_(iConsumes.consumes<CSCTag::digi_collection>(iConfig.getParameter<edm::InputTag>("CSCInput"))),
    tokenRPC_(iConsumes.consumes<RPCTag::digi_collection>(iConfig.getParameter<edm::InputTag>("RPCInput"))),
    tokenCPPF_(iConsumes.consumes<CPPFTag::digi_collection>(iConfig.getParameter<edm::InputTag>("CPPFInput"))),
    tokenGEM_(iConsumes.consumes<GEMTag::digi_collection>(iConfig.getParameter<edm::InputTag>("GEMInput"))),
    verbose_(iConfig.getUntrackedParameter<int>("verbosity")),
    primConvLUT_(iConfig.getParameter<edm::ParameterSet>("spPCParams16").getParameter<int>("PrimConvLUT")),
    fwConfig_(iConfig.getParameter<bool>("FWConfig")),
    useCSC_(iConfig.getParameter<bool>("CSCEnable")),
    useRPC_(iConfig.getParameter<bool>("RPCEnable")),
    useCPPF_(iConfig.getParameter<bool>("CPPFEnable")),
    useGEM_(iConfig.getParameter<bool>("GEMEnable")),
    era_(iConfig.getParameter<std::string>("Era"))
{

  if (era_ == "Run2_2016") {
    pt_assign_engine_.reset(new PtAssignmentEngine2016());
  } else if (era_ == "Run2_2017" || era_ == "Run2_2018") {
    pt_assign_engine_.reset(new PtAssignmentEngine2017());
  } else {
    edm::LogError("L1T") << "era_ = " << era_; return;
  }

  auto minBX       = iConfig.getParameter<int>("MinBX");
  auto maxBX       = iConfig.getParameter<int>("MaxBX");
  auto bxWindow    = iConfig.getParameter<int>("BXWindow");
  auto bxShiftCSC  = iConfig.getParameter<int>("CSCInputBXShift");
  auto bxShiftRPC  = iConfig.getParameter<int>("RPCInputBXShift");
  auto bxShiftGEM  = iConfig.getParameter<int>("GEMInputBXShift");

  const auto& spPCParams16 = config_.getParameter<edm::ParameterSet>("spPCParams16");
  auto zoneBoundaries     = spPCParams16.getParameter<std::vector<int> >("ZoneBoundaries");
  auto zoneOverlap        = spPCParams16.getParameter<int>("ZoneOverlap");
  auto includeNeighbor    = spPCParams16.getParameter<bool>("IncludeNeighbor");
  auto duplicateTheta     = spPCParams16.getParameter<bool>("DuplicateTheta");
  auto fixZonePhi         = spPCParams16.getParameter<bool>("FixZonePhi");
  auto useNewZones        = spPCParams16.getParameter<bool>("UseNewZones");
  auto fixME11Edges       = spPCParams16.getParameter<bool>("FixME11Edges");

  const auto& spPRParams16 = config_.getParameter<edm::ParameterSet>("spPRParams16");
  auto pattDefinitions    = spPRParams16.getParameter<std::vector<std::string> >("PatternDefinitions");
  auto symPattDefinitions = spPRParams16.getParameter<std::vector<std::string> >("SymPatternDefinitions");
  auto useSymPatterns     = spPRParams16.getParameter<bool>("UseSymmetricalPatterns");

  const auto& spTBParams16 = config_.getParameter<edm::ParameterSet>("spTBParams16");
  auto thetaWindow        = spTBParams16.getParameter<int>("ThetaWindow");
  auto thetaWindowZone0   = spTBParams16.getParameter<int>("ThetaWindowZone0");
  auto useSingleHits      = spTBParams16.getParameter<bool>("UseSingleHits");
  auto bugSt2PhDiff       = spTBParams16.getParameter<bool>("BugSt2PhDiff");
  auto bugME11Dupes       = spTBParams16.getParameter<bool>("BugME11Dupes");
  auto bugAmbigThetaWin   = spTBParams16.getParameter<bool>("BugAmbigThetaWin");
  auto twoStationSameBX   = spTBParams16.getParameter<bool>("TwoStationSameBX");

  const auto& spGCParams16 = config_.getParameter<edm::ParameterSet>("spGCParams16");
  auto maxRoadsPerZone    = spGCParams16.getParameter<int>("MaxRoadsPerZone");
  auto maxTracks          = spGCParams16.getParameter<int>("MaxTracks");
  auto useSecondEarliest  = spGCParams16.getParameter<bool>("UseSecondEarliest");
  auto bugSameSectorPt0   = spGCParams16.getParameter<bool>("BugSameSectorPt0");

  const auto& spPAParams16 = config_.getParameter<edm::ParameterSet>("spPAParams16");
  auto readPtLUTFile      = spPAParams16.getParameter<bool>("ReadPtLUTFile");
  auto fixMode15HighPt    = spPAParams16.getParameter<bool>("FixMode15HighPt");
  auto bug9BitDPhi        = spPAParams16.getParameter<bool>("Bug9BitDPhi");
  auto bugMode7CLCT       = spPAParams16.getParameter<bool>("BugMode7CLCT");
  auto bugNegPt           = spPAParams16.getParameter<bool>("BugNegPt");
  auto bugGMTPhi          = spPAParams16.getParameter<bool>("BugGMTPhi");
  auto promoteMode7       = spPAParams16.getParameter<bool>("PromoteMode7");
  auto modeQualVer        = spPAParams16.getParameter<int>("ModeQualVer");

  // Configure sector processors
  for (int endcap = emtf::MIN_ENDCAP; endcap <= emtf::MAX_ENDCAP; ++endcap) {
    for (int sector = emtf::MIN_TRIGSECTOR; sector <= emtf::MAX_TRIGSECTOR; ++sector) {
      const int es = (endcap - emtf::MIN_ENDCAP) * (emtf::MAX_TRIGSECTOR - emtf::MIN_TRIGSECTOR + 1) + (sector - emtf::MIN_TRIGSECTOR);

      sector_processors_.at(es).configure(
          &geometry_translator_,
          &condition_helper_,
          &sector_processor_lut_,
          pt_assign_engine_.get(),
          verbose_, endcap, sector,
          minBX, maxBX, bxWindow, bxShiftCSC, bxShiftRPC, bxShiftGEM,
          era_,
          zoneBoundaries, zoneOverlap,
          includeNeighbor, duplicateTheta, fixZonePhi, useNewZones, fixME11Edges,
          pattDefinitions, symPattDefinitions, useSymPatterns,
          thetaWindow, thetaWindowZone0, useRPC_, useSingleHits, bugSt2PhDiff, bugME11Dupes, bugAmbigThetaWin, twoStationSameBX,
          maxRoadsPerZone, maxTracks, useSecondEarliest, bugSameSectorPt0,
          readPtLUTFile, fixMode15HighPt, bug9BitDPhi, bugMode7CLCT, bugNegPt, bugGMTPhi, promoteMode7, modeQualVer
      );
    }
  }

} // End constructor: TrackFinder::TrackFinder()

TrackFinder::~TrackFinder() {

}

void TrackFinder::process(
    const edm::Event& iEvent, const edm::EventSetup& iSetup,
    EMTFHitCollection& out_hits,
    EMTFTrackCollection& out_tracks
) {

  // Clear output collections
  out_hits.clear();
  out_tracks.clear();

  // Get the geometry for TP conversions
  geometry_translator_.checkAndUpdateGeometry(iSetup);

  // Get the conditions, primarily the firmware version and the BDT forests
  condition_helper_.checkAndUpdateConditions(iEvent, iSetup);

  // ___________________________________________________________________________
  // Extract all trigger primitives

  TriggerPrimitiveCollection muon_primitives;

  EMTFSubsystemCollector collector;
  if (useCSC_)
    collector.extractPrimitives(CSCTag(), iEvent, tokenCSC_, muon_primitives);
  if (useRPC_ && useCPPF_)
    collector.extractPrimitives(CPPFTag(), iEvent, tokenCPPF_, muon_primitives);
  else if (useRPC_)
    collector.extractPrimitives(RPCTag(), iEvent, tokenRPC_, muon_primitives);
  if (useGEM_)
    collector.extractPrimitives(GEMTag(), iEvent, tokenGEM_, muon_primitives);

  // Check trigger primitives
  if (verbose_ > 2) {  // debug
    std::cout << "Num of TriggerPrimitive: " << muon_primitives.size() << std::endl;
    for (const auto& p : muon_primitives) {
      p.print(std::cout);
    }
  }

  // ___________________________________________________________________________
  // Run each sector processor

  // Reload primitive conversion LUTs if necessary
  sector_processor_lut_.read(iEvent.isRealData(), fwConfig_ ? condition_helper_.get_pc_lut_version() : primConvLUT_);

  // Reload pT LUT if necessary
  pt_assign_engine_->load(condition_helper_.get_pt_lut_version(), &(condition_helper_.getForest()));

  // MIN/MAX ENDCAP and TRIGSECTOR set in interface/Common.h
  for (int endcap = emtf::MIN_ENDCAP; endcap <= emtf::MAX_ENDCAP; ++endcap) {
    for (int sector = emtf::MIN_TRIGSECTOR; sector <= emtf::MAX_TRIGSECTOR; ++sector) {
      const int es = (endcap - emtf::MIN_ENDCAP) * (emtf::MAX_TRIGSECTOR - emtf::MIN_TRIGSECTOR + 1) + (sector - emtf::MIN_TRIGSECTOR);

      // Run-dependent configure. This overwrites many of the configurables passed by the python config file.
      if (iEvent.isRealData() && fwConfig_) {
        sector_processors_.at(es).configure_by_fw_version(condition_helper_.get_fw_version());
      }

      // Process
      sector_processors_.at(es).process(
          iEvent.id().event(),
          muon_primitives,
          out_hits,
          out_tracks
      );
    }
  }


  // ___________________________________________________________________________
  // Check emulator input and output. They are printed in a way that is friendly
  // for comparison with the firmware simulator.

  if (verbose_ > 0) {  // debug
    std::cout << "Run number: " << iEvent.id().run() << " pc_lut_ver: " << condition_helper_.get_pc_lut_version()
        << " pt_lut_ver: " << condition_helper_.get_pt_lut_version() << ", " << pt_assign_engine_->get_pt_lut_version()
        << " fw_ver: " << condition_helper_.get_fw_version()
        << std::endl;

    for (int endcap = emtf::MIN_ENDCAP; endcap <= emtf::MAX_ENDCAP; ++endcap) {
      for (int sector = emtf::MIN_TRIGSECTOR; sector <= emtf::MAX_TRIGSECTOR; ++sector) {
        const int es = (endcap - emtf::MIN_ENDCAP) * (emtf::MAX_TRIGSECTOR - emtf::MIN_TRIGSECTOR + 1) + (sector - emtf::MIN_TRIGSECTOR);

        // _____________________________________________________________________
        // This prints the hits as raw text input to the firmware simulator
        // "12345" is the BX separator

        std::cout << "==== Endcap " << endcap << " Sector " << sector << " Hits ====" << std::endl;
        std::cout << "bx e s ss st vf ql cp wg id bd hs" << std::endl;

        bool empty_sector = true;
        for (const auto& h : out_hits) {
          if (h.Sector_idx() != es)  continue;
          empty_sector = false;
        }

        for (int ibx = -3-5; (ibx < +3+5+5) && !empty_sector; ++ibx) {

          for (const auto& h : out_hits) {
            if (h.Subsystem() == TriggerPrimitive::kCSC) {
              if (h.Sector_idx() != es)  continue;
              if (h.BX() != ibx)  continue;

              int bx        = 1;
              int endcap    = (h.Endcap() == 1) ? 1 : 2;
              int sector    = h.PC_sector();
              int station   = (h.PC_station() == 0 && h.Subsector() == 1) ? 1 : h.PC_station();
              int chamber   = h.PC_chamber() + 1;
              int strip     = (h.Station() == 1 && h.Ring() == 4) ? h.Strip() + 128 : h.Strip();  // ME1/1a
              int wire      = h.Wire();
              int valid     = 1;
              std::cout << bx << " " << endcap << " " << sector << " " << h.Subsector() << " "
                  << station << " " << valid << " " << h.Quality() << " " << h.Pattern() << " "
                  << wire << " " << chamber << " " << h.Bend() << " " << strip << std::endl;

            } else if (h.Subsystem() == TriggerPrimitive::kRPC) {
              if (h.Sector_idx() != es)  continue;
              if (h.BX()+6 != ibx)  continue;  // RPC hits should be supplied 6 BX later relative to CSC hits

              // Assign RPC link index. Code taken from src/PrimitiveSelection.cc
              int rpc_sub = -1;
              int rpc_chm = -1;
              if (!h.Neighbor()) {
                rpc_sub = h.Subsector() - 1;
              } else {
                rpc_sub = 6;
              }
              if (h.Station() <= 2) {
                rpc_chm = (h.Station() - 1);
              } else {
                rpc_chm = 2 + (h.Station() - 3)*2 + (h.Ring() - 2);
              }

              int bx        = 1;
              int endcap    = (h.Endcap() == 1) ? 1 : 2;
              int sector    = h.PC_sector();
              int station   = rpc_sub;
              int chamber   = rpc_chm + 1;
              int strip     = (h.Phi_fp() >> 2);
              int wire      = (h.Theta_fp() >> 2);
              int valid     = 2;  // this marks RPC stub
              std::cout << bx << " " << endcap << " " << sector << " " << 0 << " "
                  << station << " " << valid << " " << 0 << " " << 0 << " "
                  << wire << " " << chamber << " " << 0 << " " << strip << std::endl;
            }
          }  // end loop over hits

          std::cout << "12345" << std::endl;
        }  // end loop over bx

        // _____________________________________________________________________
        // This prints the tracks as raw text output from the firmware simulator

        std::cout << "==== Endcap " << endcap << " Sector " << sector << " Tracks ====" << std::endl;
        std::cout << "bx e s a mo et ph cr q pt" << std::endl;

        for (const auto& t : out_tracks) {
          if (t.Sector_idx() != es)  continue;

          std::cout << t.BX() << " " << (t.Endcap() == 1 ? 1 : 2) << " " << t.Sector() << " " << t.PtLUT().address << " " << t.Mode() << " "
              << (t.GMT_eta() >= 0 ? t.GMT_eta() : t.GMT_eta()+512)<< " " << t.GMT_phi() << " "
              << t.GMT_charge() << " " << t.GMT_quality() << " " << t.Pt() << std::endl;
        }  // end loop over tracks

      }  // end loop over sector
    }  // end loop over endcap
  }  // end debug

  return;
}
