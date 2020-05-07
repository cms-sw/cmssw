// -*- C++ -*-
//
// Package:    UserCode/CalibratedDigis
// Class:      CalibratedDigis
//
/**\class CalibratedDigis CalibratedDigis.cc UserCode/CalibratedDigis/plugins/CalibratedDigis.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Luigi Guiducci
//         Created:  Fri, 11 Jan 2019 12:49:12 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
}  // namespace edm

using namespace std;
using namespace edm;
//
// class declaration
//

class CalibratedDigis : public edm::stream::EDProducer<> {
public:
  explicit CalibratedDigis(const edm::ParameterSet&);
  ~CalibratedDigis();

 
private:
  int timeOffset_;
  int flat_calib_;
  int scenario;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  std::unique_ptr<DTTTrigBaseSync> theSync;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<DTDigiCollection> dtDigisToken;
  edm::Handle<DTDigiCollection> DTDigiHandle;
  edm::InputTag dtDigiTag;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CalibratedDigis::CalibratedDigis(const edm::ParameterSet& iConfig) {
 
  
  //register your products
  /* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
   
   //if you want to put into the Run
   produces<ExampleData2,InRun>();
*/
  dtDigiTag = iConfig.getParameter<InputTag>("dtDigiTag");
  dtDigisToken = consumes<DTDigiCollection>(dtDigiTag);

  theSync = DTTTrigSyncFactory::get()->create(iConfig.getParameter<string>("tTrigMode"),
                                              iConfig.getParameter<ParameterSet>("tTrigModeConfig"));

  flat_calib_ = iConfig.getParameter<int>("flat_calib");
  timeOffset_ = iConfig.getParameter<int>("timeOffset");

  scenario = iConfig.getUntrackedParameter<int>("scenario");

  produces<DTDigiCollection>();
  //now do what ever other initialization is needed
}

CalibratedDigis::~CalibratedDigis() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void CalibratedDigis::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  //  auto cc = setWhatProduced(this);
  
  using namespace edm;
  theSync->setES(iSetup);  
  iEvent.getByToken(dtDigisToken, DTDigiHandle);
  DTDigiCollection mydigis;

  DTDigiCollection::DigiRangeIterator dtLayerIt;

  for (dtLayerIt = DTDigiHandle->begin(); dtLayerIt != DTDigiHandle->end(); ++dtLayerIt) {
    const DTLayerId& layerId = (*dtLayerIt).first;
    for (DTDigiCollection::const_iterator digiIt = ((*dtLayerIt).second).first; digiIt != ((*dtLayerIt).second).second;
         ++digiIt) {
      DTWireId wireId(layerId, (*digiIt).wire());
      float digiTime = (*digiIt).time();
      int wire = (*digiIt).wire();
      int number = (*digiIt).number();
      float newTime = 0;
      if (flat_calib_ != 0)
        newTime = digiTime - 325 + 25.0 * iEvent.eventAuxiliary().bunchCrossing() + float(timeOffset_);
      else {
        if (scenario == 0)  //FIX MC
          newTime = digiTime + 25.0 * 400;
        else if (scenario == 2)  //FIX SliceTest
          newTime = digiTime;
        else
          newTime =
              digiTime - theSync->offset(wireId) + 25.0 * iEvent.eventAuxiliary().bunchCrossing() + float(timeOffset_);
      }
      DTDigi newDigi(wire, newTime, number);
      mydigis.insertDigi(layerId, newDigi);
    }
  }
  auto CorrectedDTDigiCollection = std::make_unique<DTDigiCollection>(mydigis);
  iEvent.put(std::move(CorrectedDTDigiCollection));
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------

// ------------ method called once each stream after processing all runs, lumis and events  ------------

// ------------ method called when starting to processes a run  ------------

// ------------ method called when ending the processing of a run  ------------


// ------------ method called when starting to processes a luminosity block  ------------

// ------------ method called when ending the processing of a luminosity block  ------------

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

//define this as a plug-in
DEFINE_FWK_MODULE(CalibratedDigis);
