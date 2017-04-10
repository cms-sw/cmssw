#include "L1Trigger/L1TMuonEndCap/interface/TrackFinder.hh"

#include <iostream>
#include <sstream>

#include "L1Trigger/L1TMuonEndCap/interface/EMTFSubsystemCollector.hh"


TrackFinder::TrackFinder(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iConsumes) :
    geometry_translator_(),
    sector_processor_lut_(),
    pt_assign_engine_(),
    sector_processors_(),
    config_(iConfig),
    tokenCSC_(iConsumes.consumes<CSCTag::digi_collection>(iConfig.getParameter<edm::InputTag>("CSCInput"))),
    tokenRPC_(iConsumes.consumes<RPCTag::digi_collection>(iConfig.getParameter<edm::InputTag>("RPCInput"))),
    verbose_(iConfig.getUntrackedParameter<int>("verbosity")),
    useCSC_(iConfig.getParameter<bool>("CSCEnable")),
    useRPC_(iConfig.getParameter<bool>("RPCEnable"))
{
  auto minBX       = iConfig.getParameter<int>("MinBX");
  auto maxBX       = iConfig.getParameter<int>("MaxBX");
  auto bxWindow    = iConfig.getParameter<int>("BXWindow");
  auto bxShiftCSC  = iConfig.getParameter<int>("CSCInputBXShift");
  auto bxShiftRPC  = iConfig.getParameter<int>("RPCInputBXShift");
  //auto version     = iConfig.getParameter<int>("Version");        // not yet used
  //auto ptlut_ver   = iConfig.getParameter<int>("PtLUTVersion");   // not yet used

  const auto& spPCParams16 = config_.getParameter<edm::ParameterSet>("spPCParams16");
  auto zoneBoundaries     = spPCParams16.getParameter<std::vector<int> >("ZoneBoundaries");
  auto zoneOverlap        = spPCParams16.getParameter<int>("ZoneOverlap");
  auto zoneOverlapRPC     = spPCParams16.getParameter<int>("ZoneOverlapRPC");
  auto coordLUTDir        = spPCParams16.getParameter<std::string>("CoordLUTDir");
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
  auto thetaWindowRPC     = spTBParams16.getParameter<int>("ThetaWindowRPC");
  auto bugME11Dupes       = spTBParams16.getParameter<bool>("BugME11Dupes");

  const auto& spGCParams16 = config_.getParameter<edm::ParameterSet>("spGCParams16");
  auto maxRoadsPerZone    = spGCParams16.getParameter<int>("MaxRoadsPerZone");
  auto maxTracks          = spGCParams16.getParameter<int>("MaxTracks");
  auto useSecondEarliest  = spGCParams16.getParameter<bool>("UseSecondEarliest");
  auto bugSameSectorPt0   = spGCParams16.getParameter<bool>("BugSameSectorPt0");

  const auto& spPAParams16 = config_.getParameter<edm::ParameterSet>("spPAParams16");
  auto bdtXMLDir          = spPAParams16.getParameter<std::string>("BDTXMLDir");
  auto readPtLUTFile      = spPAParams16.getParameter<bool>("ReadPtLUTFile");
  auto fixMode15HighPt    = spPAParams16.getParameter<bool>("FixMode15HighPt");
  auto bug9BitDPhi        = spPAParams16.getParameter<bool>("Bug9BitDPhi");
  auto bugMode7CLCT       = spPAParams16.getParameter<bool>("BugMode7CLCT");
  auto bugNegPt           = spPAParams16.getParameter<bool>("BugNegPt");
  auto bugGMTPhi          = spPAParams16.getParameter<bool>("BugGMTPhi");


  try {
    // Configure sector processor LUT
    sector_processor_lut_.read(coordLUTDir);

    // Configure pT assignment engine
    pt_assign_engine_.read(bdtXMLDir);

    // Configure sector processors
    for (int endcap = MIN_ENDCAP; endcap <= MAX_ENDCAP; ++endcap) {
      for (int sector = MIN_TRIGSECTOR; sector <= MAX_TRIGSECTOR; ++sector) {
        const int es = (endcap - MIN_ENDCAP) * (MAX_TRIGSECTOR - MIN_TRIGSECTOR + 1) + (sector - MIN_TRIGSECTOR);

        sector_processors_.at(es).configure(
            &geometry_translator_,
            &sector_processor_lut_,
            &pt_assign_engine_,
            verbose_, endcap, sector,
            minBX, maxBX, bxWindow, bxShiftCSC, bxShiftRPC,
            zoneBoundaries, zoneOverlap, zoneOverlapRPC,
            includeNeighbor, duplicateTheta, fixZonePhi, useNewZones, fixME11Edges,
            pattDefinitions, symPattDefinitions, useSymPatterns,
            thetaWindow, thetaWindowRPC, bugME11Dupes,
            maxRoadsPerZone, maxTracks, useSecondEarliest, bugSameSectorPt0,
            readPtLUTFile, fixMode15HighPt, bug9BitDPhi, bugMode7CLCT, bugNegPt, bugGMTPhi
        );
      }
    }

  } catch (...) {
    throw;
  }
}

void TrackFinder::resetPtLUT(std::shared_ptr<const L1TMuonEndCapForest> ptLUT){

    pt_assign_engine_.load(ptLUT.get());

    // Configure sector processors
    for (int endcap = MIN_ENDCAP; endcap <= MAX_ENDCAP; ++endcap) {
      for (int sector = MIN_TRIGSECTOR; sector <= MAX_TRIGSECTOR; ++sector) {
        const int es = (endcap - MIN_ENDCAP) * (MAX_TRIGSECTOR - MIN_TRIGSECTOR + 1) + (sector - MIN_TRIGSECTOR);

        sector_processors_.at(es).resetPtAssignment(&pt_assign_engine_);

      }
    }
}

TrackFinder::~TrackFinder() {

}

void TrackFinder::process(
    const edm::Event& iEvent, const edm::EventSetup& iSetup,
    EMTFHitCollection& out_hits,
    EMTFTrackCollection& out_tracks
) const {

  // Clear output collections
  out_hits.clear();
  out_tracks.clear();

  // Get the geometry for TP conversions
  geometry_translator_.checkAndUpdateGeometry(iSetup);

  // ___________________________________________________________________________
  // Extract all trigger primitives
  TriggerPrimitiveCollection muon_primitives;

  EMTFSubsystemCollector collector;
  if (useCSC_)
    collector.extractPrimitives(CSCTag(), iEvent, tokenCSC_, muon_primitives);
  if (useRPC_)
    collector.extractPrimitives(RPCTag(), iEvent, tokenRPC_, muon_primitives);

  // Check trigger primitives
  if (verbose_ > 2) {  // debug
    std::cout << "Num of TriggerPrimitive: " << muon_primitives.size() << std::endl;
    for (const auto& p : muon_primitives) {
      p.print(std::cout);
    }
  }

  // ___________________________________________________________________________
  // Run each sector processor

  // MIN/MAX ENDCAP and TRIGSECTOR set in interface/Common.hh
  for (int endcap = MIN_ENDCAP; endcap <= MAX_ENDCAP; ++endcap) {
    for (int sector = MIN_TRIGSECTOR; sector <= MAX_TRIGSECTOR; ++sector) {
      const int es = (endcap - MIN_ENDCAP) * (MAX_TRIGSECTOR - MIN_TRIGSECTOR + 1) + (sector - MIN_TRIGSECTOR);

      sector_processors_.at(es).process(
          iEvent.id().event(),
          muon_primitives,
          out_hits,
          out_tracks
      );
    }
  }

  if (verbose_ > 0) {  // debug
    std::cout << "Num of EMTFHit: " << out_hits.size() << std::endl;
    std::cout << "bx e s ss st vf ql cp wg id bd hs" << std::endl;
    for (const auto& h : out_hits) {
      int bx      = h.BX() + 3;
      int sector  = h.PC_sector();
      int station = (h.PC_station() == 0 && h.Subsector() == 1) ? 1 : h.PC_station();
      int chamber = h.PC_chamber() + 1;
      int strip   = (h.Station() == 1 && h.Ring() == 4) ? h.Strip() + 128 : h.Strip();  // ME1/1a
      std::cout << bx << " " << h.Endcap() << " " << sector << " " << h.Subsector() << " "
          << station << " " << h.Valid() << " " << h.Quality() << " " << h.Pattern() << " "
          << h.Wire() << " " << chamber << " " << h.Bend() << " " << strip << std::endl;
    }

    std::cout << "Converted hits: " << std::endl;
    std::cout << "st ch ph th ph_hit phzvl" << std::endl;
    for (const auto& h : out_hits) {
      std::cout << h.PC_station() << " " << h.PC_chamber() << " " << h.Phi_fp() << " " << h.Theta_fp() << " "
          << (1ul<<h.Ph_hit()) << " " << h.Phzvl() << std::endl;
    }

    std::cout << "Num of EMTFTrack: " << out_tracks.size() << std::endl;
    std::cout << "bx e s a mo et ph cr q pt" << std::endl;
    for (const auto& t : out_tracks) {
      std::cout << t.BX() << " " << t.Endcap() << " " << t.Sector() << " " << t.PtLUT().address << " " << t.Mode() << " "
          << t.GMT_eta() << " " << t.GMT_phi() << " " << t.GMT_charge() << " " << t.GMT_quality() << " " << t.Pt() << std::endl;
    }
  }

  return;
}
