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
using namespace std;
using namespace cmsdt;

class DTTrigPhase2ShowerProd : public edm::stream::EDProducer<> {
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
  int showerTaggingAlgo_;                            // Shower tagging algorithm
  edm::InputTag digiTag_;                            // Digi collection label
  edm::EDGetTokenT<DTDigiCollection> dtDigisToken_;  // Digi container
  std::unique_ptr<ShowerBuilder> showerBuilder;      // Shower builder instance
};

/* Implementation of the plugin */
DTTrigPhase2ShowerProd::DTTrigPhase2ShowerProd(const ParameterSet& pset) {
  // Constructor implementation
  produces<L1Phase2MuDTShowerContainer>();
  // Unpack information from pset
  debug_ = pset.getUntrackedParameter<bool>("debug");
  digiTag_ = pset.getParameter<edm::InputTag>("digiTag");
  showerTaggingAlgo_ = pset.getParameter<int>("showerTaggingAlgo");

  // Fetch consumes
  dtDigisToken_ = consumes<DTDigiCollection>(digiTag_);

  // Algorithm functionalities
  edm::ConsumesCollector consumesColl(consumesCollector());
  showerBuilder = std::make_unique<ShowerBuilder>(pset, consumesColl);

  // Load geometry
  dtGeomH = esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();

  if (debug_) {
    LogDebug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: constructor" << endl;
    if (showerTaggingAlgo_ == 0) {
      LogDebug("DTTrigPhase2ShowerProd") << "Using standalone mode" << endl;
    } else if (showerTaggingAlgo_ == 1) {
      LogDebug("DTTrigPhase2ShowerProd") << "Using firmware emulation mode" << endl;
    } else
      LogError("DTTrigPhase2ShowerProd") << "Unrecognized shower tagging algorithm" << endl;
  }
}

DTTrigPhase2ShowerProd::~DTTrigPhase2ShowerProd() {
  // Destructor implementation
  if (debug_)
    LogDebug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: destructor" << endl;
}

void DTTrigPhase2ShowerProd::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  // beginRun implementation
  if (debug_)
    LogDebug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: beginRun started" << endl;

  showerBuilder->initialise(iEventSetup);
  if (auto geom = iEventSetup.getHandle(dtGeomH)) {
    dtGeo_ = &(*geom);
  }
}

void DTTrigPhase2ShowerProd::produce(edm::Event& iEvent, const edm::EventSetup& iEventSetup) {
  // produce implementation
  if (debug_)
    LogDebug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: produce Processing event" << endl;

  // Fetch the handle for hits
  edm::Handle<DTDigiCollection> dtdigis;
  iEvent.getByToken(dtDigisToken_, dtdigis);

  // 1. Preprocessing: store digi information by chamber
  DTDigiMap digiMap;
  DTDigiCollection::DigiRangeIterator detUnitIt;
  if (debug_)
    LogDebug("DTTrigPhase2ShowerProd") << "    Preprocessing hits..." << endl;

  for (const auto& detUnitIt : *dtdigis) {
    const DTLayerId& layId = detUnitIt.first;
    const DTChamberId chambId = layId.superlayerId().chamberId();
    const DTDigiCollection::Range& digi_range = detUnitIt.second;  // This is the digi collection
    digiMap[chambId].put(digi_range, layId);
  }

  if (debug_)
    LogDebug("DTTrigPhase2ShowerProd") << "    Hits preprocessed: " << digiMap.size() << " DT chambers to analyze"
                                       << endl;

  // 2. Look for showers in each chamber
  if (debug_)
    LogDebug("DTTrigPhase2ShowerProd") << "    Building shower candidates for:" << endl;

  std::map<DTSuperLayerId, ShowerCandidatePtr> ShowerCandidates;
  for (const auto& ich : dtGeo_->chambers()) {
    const DTChamber* chamb = ich;
    DTChamberId chid = chamb->id();
    DTDigiMap_iterator dmit = digiMap.find(chid);

    DTSuperLayerId sl1id = chamb->superLayer(1)->id();
    DTSuperLayerId sl3id = chamb->superLayer(3)->id();

    if (dmit == digiMap.end())
      continue;

    if (debug_)
      LogDebug("DTTrigPhase2ShowerProd") << "      " << chid << endl;

    showerBuilder->run(iEvent, iEventSetup, (*dmit).second, ShowerCandidates[sl1id], ShowerCandidates[sl3id]);

    // Save the rawId of these shower candidates
    ShowerCandidates[sl1id]->rawId(sl1id.rawId());
    ShowerCandidates[sl3id]->rawId(sl3id.rawId());
  }

  // 3. Check shower candidates and store them if flagged
  if (debug_)
    LogDebug("DTTrigPhase2ShowerProd") << "    Selecting shower candidates" << endl;

  std::vector<L1Phase2MuDTShower> outShower;  // prepare output container
  for (auto& sl_showerCand : ShowerCandidates) {
    auto showerCandIt = sl_showerCand.second;

    if (showerCandIt->isFlagged()) {
      DTSuperLayerId slId(showerCandIt->getRawId());

      if (debug_) {
        LogDebug("DTTrigPhase2ShowerProd")
            << "      Shower candidate tagged in chamber" << slId.chamberId() << ", SL" << slId.superlayer();
      }
      // 4. Storing results
      outShower.emplace_back(L1Phase2MuDTShower(slId.wheel(),                    // Wheel
                                                slId.sector(),                   // Sector
                                                slId.station(),                  // Station
                                                slId.superlayer(),               // SuperLayer
                                                showerCandIt->getNhits(),        // number of digis
                                                showerCandIt->getBX(),           // BX
                                                showerCandIt->getMinWire(),      // Min wire
                                                showerCandIt->getMaxWire(),      // Max wire
                                                showerCandIt->getAvgPos(),       // Average position
                                                showerCandIt->getAvgTime(),      // Average time
                                                showerCandIt->getWiresProfile()  // Wires profile
                                                ));
    }
  }
  if (debug_)
    LogDebug("DTTrigPhase2ShowerProd") << "    Storing results..." << endl;

  // 4.1 Storing results
  std::unique_ptr<L1Phase2MuDTShowerContainer> resultShower(new L1Phase2MuDTShowerContainer);
  resultShower->setContainer(outShower);
  iEvent.put(std::move(resultShower));
}

void DTTrigPhase2ShowerProd::endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
  // endRun implementation
  if (debug_)
    LogDebug("DTTrigPhase2ShowerProd") << "DTTrigPhase2ShowerProd: endRun" << endl;
}

void DTTrigPhase2ShowerProd::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // dtTriggerPhase2Shower
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("digiTag", edm::InputTag("CalibratedDigis"));
  desc.add<int>("showerTaggingAlgo", 1);
  desc.add<int>("threshold_for_shower", 6);
  desc.add<int>("nHits_per_bx", 8);
  desc.add<int>("obdt_hits_bxpersistence", 4);
  desc.add<int>("obdt_wire_relaxing_time", 2);
  desc.add<int>("bmtl1_hits_bxpersistence", 16);
  desc.add<int>("scenario", 0);
  desc.addUntracked<bool>("debug", false);

  descriptions.add("dtTriggerPhase2Shower", desc);
}

DEFINE_FWK_MODULE(DTTrigPhase2ShowerProd);
