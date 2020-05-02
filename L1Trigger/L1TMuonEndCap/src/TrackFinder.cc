#include "L1Trigger/L1TMuonEndCap/interface/TrackFinder.h"

#include <iostream>
#include <sstream>

#include "L1Trigger/L1TMuonEndCap/interface/EMTFSubsystemCollector.h"

TrackFinder::TrackFinder(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iConsumes)
    : geometry_translator_(),
      condition_helper_(),
      sector_processor_lut_(),
      pt_assign_engine_(),
      sector_processors_(),
      config_(iConfig),
      tokenDTPhi_(iConsumes.consumes<emtf::DTTag::digi_collection>(iConfig.getParameter<edm::InputTag>("DTPhiInput"))),
      tokenDTTheta_(
          iConsumes.consumes<emtf::DTTag::theta_digi_collection>(iConfig.getParameter<edm::InputTag>("DTThetaInput"))),
      tokenCSC_(iConsumes.consumes<emtf::CSCTag::digi_collection>(iConfig.getParameter<edm::InputTag>("CSCInput"))),
      tokenCSCComparator_(iConsumes.consumes<emtf::CSCTag::comparator_digi_collection>(
          iConfig.getParameter<edm::InputTag>("CSCComparatorInput"))),
      tokenRPC_(iConsumes.consumes<emtf::RPCTag::digi_collection>(iConfig.getParameter<edm::InputTag>("RPCInput"))),
      tokenRPCRecHit_(
          iConsumes.consumes<emtf::RPCTag::rechit_collection>(iConfig.getParameter<edm::InputTag>("RPCRecHitInput"))),
      tokenCPPF_(iConsumes.consumes<emtf::CPPFTag::digi_collection>(iConfig.getParameter<edm::InputTag>("CPPFInput"))),
      tokenGEM_(iConsumes.consumes<emtf::GEMTag::digi_collection>(iConfig.getParameter<edm::InputTag>("GEMInput"))),
      tokenME0_(iConsumes.consumes<emtf::ME0Tag::digi_collection>(iConfig.getParameter<edm::InputTag>("ME0Input"))),
      verbose_(iConfig.getUntrackedParameter<int>("verbosity")),
      primConvLUT_(iConfig.getParameter<edm::ParameterSet>("spPCParams16").getParameter<int>("PrimConvLUT")),
      fwConfig_(iConfig.getParameter<bool>("FWConfig")),
      useDT_(iConfig.getParameter<bool>("DTEnable")),
      useCSC_(iConfig.getParameter<bool>("CSCEnable")),
      useRPC_(iConfig.getParameter<bool>("RPCEnable")),
      useIRPC_(iConfig.getParameter<bool>("IRPCEnable")),
      useCPPF_(iConfig.getParameter<bool>("CPPFEnable")),
      useGEM_(iConfig.getParameter<bool>("GEMEnable")),
      useME0_(iConfig.getParameter<bool>("ME0Enable")),
      era_(iConfig.getParameter<std::string>("Era")) {
  if (era_ == "Run2_2016") {
    pt_assign_engine_.reset(new PtAssignmentEngine2016());
  } else if (era_ == "Run2_2017" || era_ == "Run2_2018") {
    pt_assign_engine_.reset(new PtAssignmentEngine2017());
  } else {
    edm::LogError("L1T") << "Cannot recognize the era option: " << era_;
    return;
  }

  auto minBX = iConfig.getParameter<int>("MinBX");
  auto maxBX = iConfig.getParameter<int>("MaxBX");
  auto bxWindow = iConfig.getParameter<int>("BXWindow");
  auto bxShiftCSC = iConfig.getParameter<int>("CSCInputBXShift");
  auto bxShiftRPC = iConfig.getParameter<int>("RPCInputBXShift");
  auto bxShiftGEM = iConfig.getParameter<int>("GEMInputBXShift");
  auto bxShiftME0 = iConfig.getParameter<int>("ME0InputBXShift");

  const auto& spPCParams16 = config_.getParameter<edm::ParameterSet>("spPCParams16");
  auto zoneBoundaries = spPCParams16.getParameter<std::vector<int> >("ZoneBoundaries");
  auto zoneOverlap = spPCParams16.getParameter<int>("ZoneOverlap");
  auto includeNeighbor = spPCParams16.getParameter<bool>("IncludeNeighbor");
  auto duplicateTheta = spPCParams16.getParameter<bool>("DuplicateTheta");
  auto fixZonePhi = spPCParams16.getParameter<bool>("FixZonePhi");
  auto useNewZones = spPCParams16.getParameter<bool>("UseNewZones");
  auto fixME11Edges = spPCParams16.getParameter<bool>("FixME11Edges");

  const auto& spPRParams16 = config_.getParameter<edm::ParameterSet>("spPRParams16");
  auto pattDefinitions = spPRParams16.getParameter<std::vector<std::string> >("PatternDefinitions");
  auto symPattDefinitions = spPRParams16.getParameter<std::vector<std::string> >("SymPatternDefinitions");
  auto useSymPatterns = spPRParams16.getParameter<bool>("UseSymmetricalPatterns");

  const auto& spTBParams16 = config_.getParameter<edm::ParameterSet>("spTBParams16");
  auto thetaWindow = spTBParams16.getParameter<int>("ThetaWindow");
  auto thetaWindowZone0 = spTBParams16.getParameter<int>("ThetaWindowZone0");
  auto useSingleHits = spTBParams16.getParameter<bool>("UseSingleHits");
  auto bugSt2PhDiff = spTBParams16.getParameter<bool>("BugSt2PhDiff");
  auto bugME11Dupes = spTBParams16.getParameter<bool>("BugME11Dupes");
  auto bugAmbigThetaWin = spTBParams16.getParameter<bool>("BugAmbigThetaWin");
  auto twoStationSameBX = spTBParams16.getParameter<bool>("TwoStationSameBX");

  const auto& spGCParams16 = config_.getParameter<edm::ParameterSet>("spGCParams16");
  auto maxRoadsPerZone = spGCParams16.getParameter<int>("MaxRoadsPerZone");
  auto maxTracks = spGCParams16.getParameter<int>("MaxTracks");
  auto useSecondEarliest = spGCParams16.getParameter<bool>("UseSecondEarliest");
  auto bugSameSectorPt0 = spGCParams16.getParameter<bool>("BugSameSectorPt0");

  const auto& spPAParams16 = config_.getParameter<edm::ParameterSet>("spPAParams16");
  auto readPtLUTFile = spPAParams16.getParameter<bool>("ReadPtLUTFile");
  auto fixMode15HighPt = spPAParams16.getParameter<bool>("FixMode15HighPt");
  auto bug9BitDPhi = spPAParams16.getParameter<bool>("Bug9BitDPhi");
  auto bugMode7CLCT = spPAParams16.getParameter<bool>("BugMode7CLCT");
  auto bugNegPt = spPAParams16.getParameter<bool>("BugNegPt");
  auto bugGMTPhi = spPAParams16.getParameter<bool>("BugGMTPhi");
  auto promoteMode7 = spPAParams16.getParameter<bool>("PromoteMode7");
  auto modeQualVer = spPAParams16.getParameter<int>("ModeQualVer");

  // Configure sector processors
  for (int endcap = emtf::MIN_ENDCAP; endcap <= emtf::MAX_ENDCAP; ++endcap) {
    for (int sector = emtf::MIN_TRIGSECTOR; sector <= emtf::MAX_TRIGSECTOR; ++sector) {
      const int es = (endcap - emtf::MIN_ENDCAP) * (emtf::MAX_TRIGSECTOR - emtf::MIN_TRIGSECTOR + 1) +
                     (sector - emtf::MIN_TRIGSECTOR);

      sector_processors_.at(es).configure(&geometry_translator_,
                                          &condition_helper_,
                                          &sector_processor_lut_,
                                          pt_assign_engine_.get(),
                                          verbose_,
                                          endcap,
                                          sector,
                                          minBX,
                                          maxBX,
                                          bxWindow,
                                          bxShiftCSC,
                                          bxShiftRPC,
                                          bxShiftGEM,
                                          bxShiftME0,
                                          era_,
                                          zoneBoundaries,
                                          zoneOverlap,
                                          includeNeighbor,
                                          duplicateTheta,
                                          fixZonePhi,
                                          useNewZones,
                                          fixME11Edges,
                                          pattDefinitions,
                                          symPattDefinitions,
                                          useSymPatterns,
                                          thetaWindow,
                                          thetaWindowZone0,
                                          useRPC_,
                                          useSingleHits,
                                          bugSt2PhDiff,
                                          bugME11Dupes,
                                          bugAmbigThetaWin,
                                          twoStationSameBX,
                                          maxRoadsPerZone,
                                          maxTracks,
                                          useSecondEarliest,
                                          bugSameSectorPt0,
                                          readPtLUTFile,
                                          fixMode15HighPt,
                                          bug9BitDPhi,
                                          bugMode7CLCT,
                                          bugNegPt,
                                          bugGMTPhi,
                                          promoteMode7,
                                          modeQualVer);
    }
  }

}  // End constructor: TrackFinder::TrackFinder()

TrackFinder::~TrackFinder() {}

void TrackFinder::process(const edm::Event& iEvent,
                          const edm::EventSetup& iSetup,
                          EMTFHitCollection& out_hits,
                          EMTFTrackCollection& out_tracks) {
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
  if (useCSC_) {
    collector.extractPrimitives(emtf::CSCTag(), &geometry_translator_, iEvent, tokenCSC_, muon_primitives);
    //collector.extractPrimitives(emtf::CSCTag(), &geometry_translator_, iEvent, tokenCSC_, tokenCSCComparator_, muon_primitives);
  }
  if (useRPC_ && useCPPF_) {
    collector.extractPrimitives(emtf::CPPFTag(), &geometry_translator_, iEvent, tokenCPPF_, muon_primitives);
  } else if (useRPC_) {
    collector.extractPrimitives(emtf::RPCTag(), &geometry_translator_, iEvent, tokenRPC_, muon_primitives);
    //collector.extractPrimitives(emtf::RPCTag(), &geometry_translator_, iEvent, tokenRPC_, tokenRPCRecHit_, muon_primitives);
  }
  if (useIRPC_) {
    collector.extractPrimitives(emtf::IRPCTag(), &geometry_translator_, iEvent, tokenRPC_, muon_primitives);
    //collector.extractPrimitives(emtf::IRPCTag(), &geometry_translator_, iEvent, tokenRPC_, tokenRPCRecHit_, muon_primitives);
  }
  if (useGEM_) {
    collector.extractPrimitives(emtf::GEMTag(), &geometry_translator_, iEvent, tokenGEM_, muon_primitives);
  }
  if (useME0_) {
    collector.extractPrimitives(emtf::ME0Tag(), &geometry_translator_, iEvent, tokenME0_, muon_primitives);
  }
  if (useDT_) {
    collector.extractPrimitives(
        emtf::DTTag(), &geometry_translator_, iEvent, tokenDTPhi_, tokenDTTheta_, muon_primitives);
  }

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
      const int es = (endcap - emtf::MIN_ENDCAP) * (emtf::MAX_TRIGSECTOR - emtf::MIN_TRIGSECTOR + 1) +
                     (sector - emtf::MIN_TRIGSECTOR);

      // Run-dependent configure. This overwrites many of the configurables passed by the python config file.
      if (fwConfig_) {
        sector_processors_.at(es).configure_by_fw_version(condition_helper_.get_fw_version());
      }

      // Process
      sector_processors_.at(es).process(iEvent.id().event(), muon_primitives, out_hits, out_tracks);
    }
  }

  // ___________________________________________________________________________
  if (verbose_ > 0) {  // debug
    std::cout << "Run number: " << iEvent.id().run() << " pc_lut_ver: " << condition_helper_.get_pc_lut_version()
              << " pt_lut_ver: " << condition_helper_.get_pt_lut_version()
              << " pt_lut_ver in engine: " << pt_assign_engine_->get_pt_lut_version()
              << " fw_ver: " << condition_helper_.get_fw_version() << std::endl;
  }

  if (verbose_ > 1) {  // debug
    // Check emulator input and output. They are printed as raw text that is
    // used by the firmware simulator to do comparisons.
    emtf::dump_fw_raw_input(out_hits, out_tracks);
  }

  return;
}
