/*
    EDProducer class for shower emulation in Phase2 DTs.
    Authors: 
        - Carlos Vico Villalba (U. Oviedo)
        - Daniel Estrada Acevedo (U. Oviedo)
*/

// Include CMSSW plugins
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESProducts.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TFile.h"
#include "TTree.h"

// Include Geometry plugins
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

// Phase2 trigger dataformats
#include "L1Trigger/DTTriggerPhase2/interface/constants.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTShower.h"
#include "DataFormats/L1DTTrackFinder/interface/L1Phase2MuDTShowerContainer.h"

// Functionalities
#include "L1Trigger/DTTriggerPhase2/interface/ShowerBuilder.h"
#include "L1Trigger/DTTriggerPhase2/interface/ShowerCandidate.h"

// DT trigger GeomUtils
#include "DQM/DTMonitorModule/interface/DTTrigGeomUtils.h"

// C++ built-ins
#include <fstream>
#include <iostream>
#include <queue>
#include <cmath>

using namespace edm;
using namespace cmsdt;

class DTTrigPhase2ShowerProd : public edm::stream::EDProducer<edm::stream::WatchRuns> {
  /* Declaration of the plugin */

  // Types
  typedef std::map<DTChamberId, DTDigiCollection, std::less<DTChamberId>> DTDigiMap;
  typedef DTDigiMap::iterator DTDigiMap_iterator;
  typedef DTDigiMap::const_iterator DTDigiMap_const_iterator;

public:
  // Public methods/attributes

  //! Constructor
  DTTrigPhase2ShowerProd(const edm::ParameterSet& pset);

  //! Destructor
  ~DTTrigPhase2ShowerProd() override;

  //! Create Trigger Units before starting event processing
  void beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;

  //! Producer: process every event and generates trigger data
  void produce(edm::Event& iEvent, const edm::EventSetup& iEventSetup) override;

  //! endRun: finish things
  void endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // Members
  const DTGeometry* dtGeo_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH;

private:
  // Private methods/attributes
  bool debug_;                                       // Debug flag
  bool dump_digis_;                                  // Dump digis flag
  edm::InputTag digiTag_;                            // Digi collection label
  int showerTaggingAlgo_;                            // Shower tagging algorithm
  edm::EDGetTokenT<DTDigiCollection> dtDigisToken_;  // Digi container
  std::unique_ptr<ShowerBuilder> showerBuilder;      // Shower builder instance

  // TTree to dump digis
  std::unique_ptr<TFile> shower_digis_file_;  // Only used if TFileService is not available
  TTree* m_dumped_digis_tree;
  bool using_tfileservice_;
};

/* Implementation of the plugin */
DTTrigPhase2ShowerProd::DTTrigPhase2ShowerProd(const ParameterSet& pset)
    : debug_(pset.getUntrackedParameter<bool>("debug")),
      dump_digis_(pset.getUntrackedParameter<bool>("dump_digis")),
      digiTag_(pset.getParameter<edm::InputTag>("digiTag")),
      showerTaggingAlgo_(pset.getParameter<int>("showerTaggingAlgo")),
      using_tfileservice_(false) {
  produces<L1Phase2MuDTShowerContainer>();

  if (dump_digis_) {
    // Try to get TFileService
    edm::Service<TFileService> fs;

    if (fs.isAvailable()) {
      // TFileService is available - use it
      m_dumped_digis_tree = fs->make<TTree>("DTShowerDigisTree", "DT Shower Digis Tree");
      using_tfileservice_ = true;
    } else {
      // TFileService not available - create a standalone TFile
      shower_digis_file_ = std::make_unique<TFile>("shower_digis.root", "RECREATE");
      m_dumped_digis_tree = new TTree("DTShowerDigisTree", "DT Shower Digis Tree");
      using_tfileservice_ = false;
    }
  }

  // Fetch consumes
  dtDigisToken_ = consumes<DTDigiCollection>(digiTag_);

  // Algorithm functionalities
  edm::ConsumesCollector consumesColl(consumesCollector());
  // Pass raw pointer directly - no shared_ptr wrapper needed
  showerBuilder = std::make_unique<ShowerBuilder>(pset, consumesColl, dump_digis_ ? m_dumped_digis_tree : nullptr);

  // Load geometry
  dtGeomH = esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();

  if (debug_) {
    showerb::log_debug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: constructor";

    if (showerTaggingAlgo_ == 0)
      showerb::log_debug("DTTrigPhase2ShowerProd") << "Using standalone mode";
    else if (showerTaggingAlgo_ == 1)
      showerb::log_debug("DTTrigPhase2ShowerProd") << "Using firmware emulation mode";
    else
      showerb::log_debug("DTTrigPhase2ShowerProd") << "Unrecognized shower tagging algorithm";
  }
}

DTTrigPhase2ShowerProd::~DTTrigPhase2ShowerProd() {
  if (dump_digis_ && !using_tfileservice_) {
    // Only manage file ourselves if it was created. Not necessary if TFileService was used.
    if (shower_digis_file_) {
      shower_digis_file_->cd();
      m_dumped_digis_tree->Write();
      shower_digis_file_->Close();
      delete m_dumped_digis_tree;  // Clean up our TTree
    }
  }

  if (debug_)
    showerb::log_debug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: destructor";
}

void DTTrigPhase2ShowerProd::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  if (debug_)
    showerb::log_debug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: beginRun started";

  if (auto geom = iEventSetup.getHandle(dtGeomH)) {
    dtGeo_ = &(*geom);
  }
}

void DTTrigPhase2ShowerProd::produce(edm::Event& iEvent, const edm::EventSetup& iEventSetup) {
  showerBuilder->initialise(iEventSetup);

  if (debug_)
    showerb::log_debug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: produce Processing event";

  // Fetch the handle for hits
  edm::Handle<DTDigiCollection> dtdigis;
  iEvent.getByToken(dtDigisToken_, dtdigis);

  // 1. Preprocessing: store digi information by chamber
  DTDigiMap digiMap;
  DTDigiCollection::DigiRangeIterator detUnitIt;
  if (debug_)
    showerb::log_debug("DTTrigPhase2ShowerProd") << "    Preprocessing hits for event " << iEvent.id().event();

  for (const auto& detUnitIt : *dtdigis) {
    const DTLayerId& layId = detUnitIt.first;
    const DTChamberId chambId = layId.superlayerId().chamberId();
    const DTDigiCollection::Range& digi_range = detUnitIt.second;
    digiMap[chambId].put(digi_range, layId);
  }

  if (debug_)
    showerb::log_debug("DTTrigPhase2ShowerProd")
        << "    Hits preprocessed: " << digiMap.size() << " DT chambers to analyze";

  // 2. Look for showers in each chamber
  if (debug_)
    showerb::log_debug("DTTrigPhase2ShowerProd") << "    Building shower candidates for:";

  ShowerCandidatesMap showersMap;
  for (const auto& ich : dtGeo_->chambers()) {
    const DTChamber* chamb = ich;
    DTChamberId chid = chamb->id();
    DTDigiMap_iterator dmit = digiMap.find(chid);

    if (dmit == digiMap.end())
      continue;

    if (debug_)
      showerb::log_debug("DTTrigPhase2ShowerProd") << "      " << chid;

    showerBuilder->run(iEvent, iEventSetup, (*dmit).second, showersMap, chamb);
  }

  if (dump_digis_)
    m_dumped_digis_tree->Fill();

  // 3. Check shower candidates and store them if flagged
  if (debug_)
    showerb::log_debug("DTTrigPhase2ShowerProd")
        << "    Selecting shower candidates (to check: " << showersMap.size() << " )";

  std::vector<L1Phase2MuDTShower> outShower;
  for (auto& sl_showerCandIt : showersMap) {
    auto& showerCandPtrs = sl_showerCandIt.second;
    for (auto& showerCandPtr : showerCandPtrs) {
      if (showerCandPtr->isFlagged()) {
        DTSuperLayerId slId(sl_showerCandIt.first.rawId());

        if (debug_)
          showerb::log_debug("DTTrigPhase2ShowerProd")
              << "      Shower candidate tagged in chamber" << slId.chamberId() << ", SL" << slId.superlayer();

        // 4. Storing results
        outShower.emplace_back(L1Phase2MuDTShower(
            slId.wheel(),                                // Wheel
            slId.sector(),                               // Sector
            slId.station(),                              // Station
            slId.superlayer(),                           // SuperLayer
            showerCandPtr->getNhits(),                   // number of digis
            showerCandPtr->getBX(),                      // BX
            showerCandPtr->getMinWire(),                 // Min wire
            showerCandPtr->getMaxWire(),                 // Max wire
            showerCandPtr->getAvgPos(),                  // Average position
            showerCandPtr->getAvgTime(),                 // Average time
            showerCandPtr->getWiresProfile(),            // Wires profile
            showerCandPtr->getWiresConstituents(),       // Wires constituents
            showerCandPtr->getWiresLayerConstituents(),  // Wires layer constituents
            showerCandPtr->getWiresTdcConstituents()     // Wires tdc constituents
            ));
      }
    }
  }
  if (debug_)
    showerb::log_debug("DTTrigPhase2ShowerProd") << "    Storing results...";

  // 4.1 Storing results
  std::unique_ptr<L1Phase2MuDTShowerContainer> resultShower(new L1Phase2MuDTShowerContainer);
  resultShower->setContainer(outShower);
  iEvent.put(std::move(resultShower));
}

void DTTrigPhase2ShowerProd::endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  if (debug_)
    showerb::log_debug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: endRun";
}

void DTTrigPhase2ShowerProd::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // dtTriggerPhase2Shower
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("CalibratedDigis"));
  desc.add<int>("showerTaggingAlgo", 1);
  desc.add<int>("threshold_for_shower", 6);
  desc.add<int>("nHits_per_bx_phi", 8);
  desc.add<int>("nHits_per_bx_z", 8);
  desc.add<int>("obdt_hits_bxpersistence_phi", 4);
  desc.add<int>("obdt_hits_bxpersistence_z", 4);
  desc.add<int>("obdt_wire_relaxing_time", 2);
  desc.add<int>("bmtl1_hits_bxpersistence", 16);
  desc.add<int>("scenario", 0);
  desc.addUntracked<bool>("debug", false);
  desc.addUntracked<bool>("dump_digis", false);
  descriptions.add("dtTriggerPhase2Shower", desc);
}

DEFINE_FWK_MODULE(DTTrigPhase2ShowerProd);
