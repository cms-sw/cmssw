/*
    EDProducer class for shower computation in Phase2 DTs.
    Author: Carlos Vico Villalba (U. Oviedo)
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
#include "L1Trigger/DTTriggerPhase2/interface/ShowerGrouping.h"
#include "L1Trigger/DTTriggerPhase2/interface/ShowerBuffer.h"
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
    
        // Members
        const DTGeometry* dtGeo_;
        edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomH;

    private:
        // Private methods/attributes
        bool debug_; // Debug flag
        int nHits_per_bx_; // Number of hits sent by the OBDT per bx
        edm::InputTag digiTag_; // Digi collection label
        edm::EDGetTokenT<DTDigiCollection> dtDigisToken_; // Digi container
        std::unique_ptr<ShowerGrouping> showerBuilder;

        
        void ShowerDebugger(string msg, int vlevel);
    protected:
        // Protected methods/attributes       
};

/* Implementation of the plugin */
DTTrigPhase2ShowerProd::DTTrigPhase2ShowerProd(const ParameterSet& pset) {
    // Constructor implementation
    produces<L1Phase2MuDTShowerContainer>();

    // Unpack information from pset
    debug_ = pset.getUntrackedParameter<bool>("debug");
    digiTag_ = pset.getParameter<edm::InputTag>("digiTag");
    nHits_per_bx_ = pset.getParameter<int>("nHits_per_bx");

    // Fetch consumes
    dtDigisToken_ = consumes<DTDigiCollection>(digiTag_);

    // Algorithm functionalities
    edm::ConsumesCollector consumesColl(consumesCollector());
    showerBuilder = std::make_unique<ShowerGrouping>(pset, consumesColl);

    // Load geometry
    dtGeomH = esConsumes<DTGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();

}

DTTrigPhase2ShowerProd::~DTTrigPhase2ShowerProd() {
    // Destructor implementation
}

void DTTrigPhase2ShowerProd::beginRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
    // beginRun implementation
    if (debug_) ShowerDebugger("DTTrigPhase2ShowerProd::beginRun beginRun started", 1);

    showerBuilder->initialise(iEventSetup);
    if (auto geom = iEventSetup.getHandle(dtGeomH)) {
        dtGeo_ = &(*geom);
    }
}

void DTTrigPhase2ShowerProd::produce(edm::Event& iEvent, const edm::EventSetup& iEventSetup) {
    // produce implementation
    if (debug_) ShowerDebugger("DTTrigPhase2ShowerProd::produce Processing event", 1);

    // Fetch the handle for hits
    edm::Handle<DTDigiCollection> dtdigis;
    iEvent.getByToken(dtDigisToken_, dtdigis);

    // 1. Preprocessing: store digi information by chamber
    DTDigiMap digiMap;
    DTDigiCollection::DigiRangeIterator detUnitIt;
    if (debug_) ShowerDebugger("Preprocessing hits...", 2);
    for (const auto& detUnitIt : *dtdigis) {
        const DTLayerId& layId = detUnitIt.first;
        const DTChamberId chambId = layId.superlayerId().chamberId();
        const DTDigiCollection::Range& digi_range = detUnitIt.second; // This is the digi collection
        digiMap[chambId].put( digi_range, layId );
    }
    
    if (debug_) ShowerDebugger("Hits preprocessed.", 2);

    // 2. Look for showers in each station
    if (debug_) ShowerDebugger("Searching for showers!", 2);
    std::map<int, ShowerBufferPtr> ShowerBuffers;
    for (const auto& ich : dtGeo_->chambers()) {
        const DTChamber* chamb = ich;
        DTChamberId chid = chamb->id();
        DTDigiMap_iterator dmit = digiMap.find(chid);

        if (dmit == digiMap.end())
            continue;
        showerBuilder->run(iEvent, iEventSetup, (*dmit).second, ShowerBuffers[chid.rawId()] );

        // Save the rawId of this shower
        ShowerBuffers[chid.rawId()]->rawId( chid.rawId() );
    }

    // 3. Build shower candidates
    if (debug_) ShowerDebugger("Building shower candidates!", 2);
    
    std::map<int, ShowerCandidatePtr> showerCands;
    for (auto& ch_showerBuf : ShowerBuffers) {
      // In normal conditions, there's just one buffer per event. 
      // when considering delays from OBDT, there might be multiple ones

      auto showerBufIt = ch_showerBuf.second;
      if (showerBufIt->isFlagged()) {
          if (debug_) ShowerDebugger("Building a shower candidate", 3);
          
          DTChamberId chId(showerBufIt->getRawId());

          // Create the candidate
          auto showerCand = std::make_shared<ShowerCandidate>(showerBufIt);
          auto rawId = showerBufIt->getRawId();

          if (debug_) {
             ShowerDebugger("  - Properties of this shower candidate:", 3);
             ShowerDebugger("Wheel:" + std::to_string(chId.wheel()) , 4);
             ShowerDebugger("Sector:" + std::to_string(chId.sector()-1) , 4);
             ShowerDebugger("Station:" + std::to_string(chId.station()) , 4);
             ShowerDebugger("nHits:" + std::to_string(showerCand->getNhits()) , 4);
             ShowerDebugger("avgPos:" + std::to_string(showerCand->getAvgPos()) , 4);
             ShowerDebugger("avgTime:" + std::to_string(showerCand->getAvgTime()) , 4);
          }

          showerCands[rawId] = std::move(showerCand); 
      }
    } 
    if (debug_) ShowerDebugger("Storing results!", 2);

    // 4. Storing results
    std::vector<L1Phase2MuDTShower> outShower;
    for (auto &chamber_showerCand : showerCands) {
        auto showerCandIt = chamber_showerCand.second;
        DTChamberId chId(showerCandIt->getRawId());

        outShower.emplace_back(
          L1Phase2MuDTShower(
            -1, // BX
            chId.wheel(), // Wheel
            chId.sector()-1, // Sector 
            chId.station(), // Station
            showerCandIt->getNhits(), // number of digis
            showerCandIt->getAvgPos(), // Average position
            showerCandIt->getAvgTime()  // AVerage time
          )
        );
    }

    std::unique_ptr<L1Phase2MuDTShowerContainer> resultShower(new L1Phase2MuDTShowerContainer);
    resultShower->setContainer(outShower);
    iEvent.put(std::move(resultShower));

}

void DTTrigPhase2ShowerProd::endRun(edm::Run const& iRun, const edm::EventSetup& iEventSetup) {
    // endRun implementation
}

void DTTrigPhase2ShowerProd::ShowerDebugger(string msg, int vlevel){
    /* Debugger */
    int indent = vlevel*2; // Each vlevel is a 2 space indent
    string indent_txt = string(indent, ' ');
    auto indented_msg = indent_txt + msg;
    if (vlevel == 1) cout << indent_txt << ">> " << msg << endl;
    if (vlevel == 2) cout << indent_txt << "* " << msg << endl;
    if (vlevel == 3) cout << indent_txt << "+ " << msg << endl;
    if (vlevel == 4) cout << indent_txt << "     " << msg << endl;
}


DEFINE_FWK_MODULE(DTTrigPhase2ShowerProd);
