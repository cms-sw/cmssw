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
}

using namespace std;
using namespace edm;
//
// class declaration
//

class CalibratedDigis : public edm::stream::EDProducer<> {
public:
    explicit CalibratedDigis(const edm::ParameterSet&);
    ~CalibratedDigis();

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    int timeOffset_;
    int flat_calib_;
    int scenario;

    virtual void beginStream(edm::StreamID) override;
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    virtual void endStream() override;

    DTTTrigBaseSync *theSync;
    virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
    virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
    //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
    //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

    // ----------member data ---------------------------
    edm::EDGetTokenT< DTDigiCollection > dtDigisToken;
    edm::Handle< DTDigiCollection > DTDigiHandle;
    edm::ESHandle< DTGeometry > DTGeometryHandle;
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
CalibratedDigis::CalibratedDigis(const edm::ParameterSet& iConfig)
{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
   
   //if you want to put into the Run
   produces<ExampleData2,InRun>();
*/
    dtDigiTag = iConfig.getParameter<InputTag>("dtDigiTag");
    std::cout<<"dtDigiTag found:"<<dtDigiTag<<std::endl;
    dtDigisToken = consumes< DTDigiCollection >( dtDigiTag );
    
    flat_calib_ = iConfig.getParameter<int>("flat_calib");
    timeOffset_ = iConfig.getParameter<int>("timeOffset");
    theSync = DTTTrigSyncFactory::get()->create(iConfig.getParameter<string>("tTrigMode"),
					      iConfig.getParameter<ParameterSet>("tTrigModeConfig"));

    scenario = iConfig.getUntrackedParameter<int>("scenario");

    produces<DTDigiCollection>();
    //now do what ever other initialization is needed
  
}


CalibratedDigis::~CalibratedDigis()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CalibratedDigis::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //std::cout<<"Step0"<<std::endl;std::cout.flush();  
  using namespace edm;
  theSync->setES(iSetup);
  //edm::Handle< DTDigiCollection > DTDigiHandle;  
  
  iEvent.getByToken( dtDigisToken, DTDigiHandle );
  ESHandle<DTGeometry> dtGeom;
  iSetup.get<MuonGeometryRecord>().get(dtGeom);
  DTDigiCollection mydigis;

  
  DTDigiCollection::DigiRangeIterator dtLayerIt;

  for (dtLayerIt = DTDigiHandle->begin();dtLayerIt != DTDigiHandle->end();++dtLayerIt){
      const DTLayerId& layerId = (*dtLayerIt).first;
      for (DTDigiCollection::const_iterator digiIt = ((*dtLayerIt).second).first;digiIt!=((*dtLayerIt).second).second; ++digiIt){
	  DTWireId wireId(layerId, (*digiIt).wire());
	  float digiTime=(*digiIt).time();
	  int wire = (*digiIt).wire();
	  int number= (*digiIt).number();
	  float newTime =0;
	  if(flat_calib_!=0)
	      newTime = digiTime - 325                     + 25.0*iEvent.eventAuxiliary().bunchCrossing() + float(timeOffset_);
	  else {
	      if (scenario == 0) //FIX MC 
	          newTime = digiTime + 25.0*400;
	      else if (scenario == 2) //FIX SliceTest 
	          newTime = digiTime;
	      else 
	          newTime = digiTime - theSync->offset(wireId) + 25.0*iEvent.eventAuxiliary().bunchCrossing() + float(timeOffset_); 
	  }
	  DTDigi newDigi(wire, newTime, number);
	  mydigis.insertDigi(layerId,newDigi);
      }
  }
  auto CorrectedDTDigiCollection = std::make_unique<DTDigiCollection>(mydigis);
  iEvent.put(std::move(CorrectedDTDigiCollection)); 
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
CalibratedDigis::beginStream(edm::StreamID)
{
 
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
CalibratedDigis::endStream() {
}

// ------------ method called when starting to processes a run  ------------
void
CalibratedDigis::beginRun(edm::Run const&, edm::EventSetup const&)
{

}
 
// ------------ method called when ending the processing of a run  ------------

void
CalibratedDigis::endRun(edm::Run const&, edm::EventSetup const&)
{
}
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
CalibratedDigis::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
CalibratedDigis::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
CalibratedDigis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(CalibratedDigis);
