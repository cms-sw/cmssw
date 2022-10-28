// -*- C++ -*-
//
// Package:    L1Trigger/DTTriggerPhase2
// Class:      CalibratedDigis
//
/**\class CalibratedDigis CalibratedDigis.cc L1Trigger/DTTriggerPhase2/plugins/CalibratedDigis.cc

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
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "CalibMuon/DTDigiSync/interface/DTTTrigBaseSync.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"

#include "CalibMuon/DTDigiSync/interface/DTTTrigSyncFactory.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/DTTriggerPhase2/interface/constants.h"

namespace edm {
  class ParameterSet;
  class EventSetup;
}  // namespace edm

using namespace std;
using namespace edm;
using namespace cmsdt;
//
// class declaration
//

class CalibratedDigis : public edm::stream::EDProducer<> {
public:
  explicit CalibratedDigis(const edm::ParameterSet&);
  ~CalibratedDigis() override;

private:
  int timeOffset_;
  int flat_calib_;
  int scenario;

  void produce(edm::Event&, const edm::EventSetup&) override;

  std::unique_ptr<DTTTrigBaseSync> theSync;

  // ----------member data ---------------------------
  edm::EDGetTokenT<DTDigiCollection> dtDigisToken;
  edm::Handle<DTDigiCollection> DTDigiHandle;
  edm::InputTag dtDigiTag;

  static constexpr float bxspacing = 25.0;
  static constexpr float timeshift = 400.0;
  static constexpr float flatcalib = 325.0;
};

//
// constructors and destructor
//
CalibratedDigis::CalibratedDigis(const edm::ParameterSet& iConfig) {
  //register your products
  dtDigiTag = iConfig.getParameter<InputTag>("dtDigiTag");
  dtDigisToken = consumes<DTDigiCollection>(dtDigiTag);

  theSync = DTTTrigSyncFactory::get()->create(iConfig.getParameter<string>("tTrigMode"),
                                              iConfig.getParameter<ParameterSet>("tTrigModeConfig"),
                                              consumesCollector());

  flat_calib_ = iConfig.getParameter<int>("flat_calib");
  timeOffset_ = iConfig.getParameter<int>("timeOffset");

  scenario = iConfig.getParameter<int>("scenario");

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
  using namespace edm;
  theSync->setES(iSetup);
  iEvent.getByToken(dtDigisToken, DTDigiHandle);
  DTDigiCollection mydigis;

  for (const auto& dtLayerIt : *DTDigiHandle) {
    const DTLayerId& layerId = dtLayerIt.first;
    for (DTDigiCollection::const_iterator digiIt = dtLayerIt.second.first; digiIt != dtLayerIt.second.second;
         ++digiIt) {
      DTWireId wireId(layerId, (*digiIt).wire());
      float digiTime = (*digiIt).time();
      int wire = (*digiIt).wire();
      int number = (*digiIt).number();
      float newTime = 0;
      if (flat_calib_ != 0)
        newTime = digiTime - flatcalib + bxspacing * iEvent.eventAuxiliary().bunchCrossing() + float(timeOffset_);
      else {
        if (scenario == MC)  //FIX MC
          newTime = digiTime + bxspacing * timeshift;
        else if (scenario == SLICE_TEST)  //FIX SliceTest
          newTime = digiTime;
        else
          newTime = digiTime - theSync->offset(wireId) + bxspacing * iEvent.eventAuxiliary().bunchCrossing() +
                    float(timeOffset_);
      }
      DTDigi newDigi(wire, newTime, number);
      mydigis.insertDigi(layerId, newDigi);
    }
  }
  auto CorrectedDTDigiCollection = std::make_unique<DTDigiCollection>(mydigis);
  iEvent.put(std::move(CorrectedDTDigiCollection));
}

//define this as a plug-in
DEFINE_FWK_MODULE(CalibratedDigis);
