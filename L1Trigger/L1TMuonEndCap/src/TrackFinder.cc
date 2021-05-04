#include "L1Trigger/L1TMuonEndCap/interface/TrackFinder.h"

#include <iostream>
#include <sstream>

TrackFinder::TrackFinder(const edm::ParameterSet& iConfig, edm::ConsumesCollector&& iConsumes)
    : setup_(iConfig),
      sector_processors_(),
      tokenDTPhi_(iConsumes.consumes<emtf::DTTag::digi_collection>(iConfig.getParameter<edm::InputTag>("DTPhiInput"))),
      tokenDTTheta_(
          iConsumes.consumes<emtf::DTTag::theta_digi_collection>(iConfig.getParameter<edm::InputTag>("DTThetaInput"))),
      tokenCSC_(iConsumes.consumes<emtf::CSCTag::digi_collection>(iConfig.getParameter<edm::InputTag>("CSCInput"))),
      tokenCSCComparator_(iConsumes.consumes<emtf::CSCTag::comparator_digi_collection>(
          iConfig.getParameter<edm::InputTag>("CSCComparatorInput"))),
      tokenRPC_(iConsumes.consumes<emtf::RPCTag::digi_collection>(iConfig.getParameter<edm::InputTag>("RPCInput"))),
      tokenCPPF_(iConsumes.consumes<emtf::CPPFTag::digi_collection>(iConfig.getParameter<edm::InputTag>("CPPFInput"))),
      tokenGEM_(iConsumes.consumes<emtf::GEMTag::digi_collection>(iConfig.getParameter<edm::InputTag>("GEMInput"))),
      tokenME0_(iConsumes.consumes<emtf::ME0Tag::digi_collection>(iConfig.getParameter<edm::InputTag>("ME0Input"))),
      verbose_(iConfig.getUntrackedParameter<int>("verbosity")) {}

TrackFinder::~TrackFinder() {}

void TrackFinder::process(const edm::Event& iEvent,
                          const edm::EventSetup& iSetup,
                          EMTFHitCollection& out_hits,
                          EMTFTrackCollection& out_tracks) {
  // Clear output collections
  out_hits.clear();
  out_tracks.clear();

  // Check and update geometry, conditions, versions, sp LUTs, and pt assignment engine
  setup_.reload(iEvent, iSetup);

  auto tp_geom_ = &(setup_.getGeometryTranslator());

  // Check versions
  if (verbose_ > 0) {  // debug
    std::cout << "Event: " << iEvent.id() << " isData: " << iEvent.isRealData() << " useO2O: " << setup_.useO2O()
              << " era: " << setup_.era() << " fw_ver: " << setup_.get_fw_version()
              << " pt_lut_ver: " << setup_.get_pt_lut_version()
              << " pt_lut_ver in engine: " << setup_.getPtAssignmentEngine()->get_pt_lut_version()
              << " pc_lut_ver: " << setup_.get_pc_lut_version()
              << " pc_lut_ver in cond (i): " << setup_.getConditionHelper().get_pc_lut_version()
              << " pc_lut_ver in cond (ii): " << setup_.getConditionHelper().get_pc_lut_version_unchecked()
              << std::endl;
  }

  // ___________________________________________________________________________
  // Extract all trigger primitives

  TriggerPrimitiveCollection muon_primitives;

  EMTFSubsystemCollector collector;

  auto iConfig = setup_.getConfig();
  auto useDT = iConfig.getParameter<bool>("DTEnable");
  auto useCSC = iConfig.getParameter<bool>("CSCEnable");
  auto useRPC = iConfig.getParameter<bool>("RPCEnable");
  auto useIRPC = iConfig.getParameter<bool>("IRPCEnable");
  auto useCPPF = iConfig.getParameter<bool>("CPPFEnable");
  auto useGEM = iConfig.getParameter<bool>("GEMEnable");
  auto useME0 = iConfig.getParameter<bool>("ME0Enable");

  if (useCSC) {
    collector.extractPrimitives(emtf::CSCTag(), tp_geom_, iEvent, tokenCSC_, muon_primitives);
    //collector.extractPrimitives(emtf::CSCTag(), tp_geom_, iEvent, tokenCSC_, tokenCSCComparator_, muon_primitives);
  }
  if (useRPC && useCPPF) {
    collector.extractPrimitives(emtf::CPPFTag(), tp_geom_, iEvent, tokenCPPF_, muon_primitives);
  } else if (useRPC) {
    collector.extractPrimitives(emtf::RPCTag(), tp_geom_, iEvent, tokenRPC_, muon_primitives);
  }
  if (useIRPC) {
    collector.extractPrimitives(emtf::IRPCTag(), tp_geom_, iEvent, tokenRPC_, muon_primitives);
  }
  if (useGEM) {
    collector.extractPrimitives(emtf::GEMTag(), tp_geom_, iEvent, tokenGEM_, muon_primitives);
  }
  if (useME0) {
    collector.extractPrimitives(emtf::ME0Tag(), tp_geom_, iEvent, tokenME0_, muon_primitives);
  }
  if (useDT) {
    collector.extractPrimitives(emtf::DTTag(), tp_geom_, iEvent, tokenDTPhi_, tokenDTTheta_, muon_primitives);
  }

  // Check trigger primitives. The printout is really verbose.
  if (verbose_ > 2) {  // debug
    std::cout << "Num of TriggerPrimitive: " << muon_primitives.size() << std::endl;
    for (const auto& p : muon_primitives) {
      p.print(std::cout);
    }
  }

  // ___________________________________________________________________________
  // Run the sector processors

  for (int endcap = emtf::MIN_ENDCAP; endcap <= emtf::MAX_ENDCAP; ++endcap) {
    for (int sector = emtf::MIN_TRIGSECTOR; sector <= emtf::MAX_TRIGSECTOR; ++sector) {
      const int es = (endcap - emtf::MIN_ENDCAP) * (emtf::MAX_TRIGSECTOR - emtf::MIN_TRIGSECTOR + 1) +
                     (sector - emtf::MIN_TRIGSECTOR);

      sector_processors_.at(es).configure(&setup_, verbose_, endcap, sector);
      sector_processors_.at(es).process(iEvent.id(), muon_primitives, out_hits, out_tracks);
    }
  }

  // ___________________________________________________________________________
  // Check emulator input and output. They are printed as raw text that is
  // used by the firmware simulator to do comparisons.
  if (verbose_ > 1) {  // debug
    emtf::dump_fw_raw_input(out_hits, out_tracks);
  }

  return;
}
