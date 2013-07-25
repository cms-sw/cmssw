// -*- C++ -*-
//
// Package:    HcalHitSelection
// Class:      HcalHitSelection
// 
/**\class HcalHitSelection HcalHitSelection.cc RecoLocalCalo/HcalHitSelection/src/HcalHitSelection.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Thu Nov  4 22:17:56 CET 2010
// $Id: HcalHitSelection.cc,v 1.4 2011/03/24 19:41:28 vlimant Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

//
// class declaration
//

class HcalHitSelection : public edm::EDProducer {
   public:
      explicit HcalHitSelection(const edm::ParameterSet&);
      ~HcalHitSelection();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
  
  edm::InputTag hbheTag,hoTag,hfTag;
  int hoSeverityLevel;
  std::vector<edm::InputTag> interestingDetIdCollections;
  
  //hcal severity ES
  edm::ESHandle<HcalChannelQuality> theHcalChStatus;
  edm::ESHandle<HcalSeverityLevelComputer> theHcalSevLvlComputer;
  std::set<DetId> toBeKept;
  template <typename CollectionType> void skim( const edm::Handle<CollectionType> & input, std::auto_ptr<CollectionType> & output,int severityThreshold=0);
  
      // ----------member data ---------------------------
};

template <class CollectionType> void HcalHitSelection::skim( const edm::Handle<CollectionType> & input, std::auto_ptr<CollectionType> & output,int severityThreshold){
  output->reserve(input->size());
  typename CollectionType::const_iterator begin=input->begin();
  typename CollectionType::const_iterator end=input->end();
  typename CollectionType::const_iterator hit=begin;

  for (;hit!=end;++hit){
    //    edm::LogError("HcalHitSelection")<<"the hit pointer is"<<&(*hit);
    const DetId & id = hit->detid();
    const uint32_t & recHitFlag = hit->flags();
    //    edm::LogError("HcalHitSelection")<<"the hit id and flag are "<<id.rawId()<<" "<<recHitFlag;
	
    const uint32_t & dbStatusFlag = theHcalChStatus->getValues(id)->getValue();
    int severityLevel = theHcalSevLvlComputer->getSeverityLevel(id, recHitFlag, dbStatusFlag); 
    //anything that is not "good" goes in
    if (severityLevel>severityThreshold){
      output->push_back(*hit);
    }else{
      //chek on the detid list
      if (toBeKept.find(id)!=toBeKept.end())
	output->push_back(*hit);
    }
  }
}

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
HcalHitSelection::HcalHitSelection(const edm::ParameterSet& iConfig)
{
  hbheTag=iConfig.getParameter<edm::InputTag>("hbheTag");
  hfTag=iConfig.getParameter<edm::InputTag>("hfTag");
  hoTag=iConfig.getParameter<edm::InputTag>("hoTag");

  interestingDetIdCollections = iConfig.getParameter< std::vector<edm::InputTag> >("interestingDetIds");

  hoSeverityLevel=iConfig.getParameter<int>("hoSeverityLevel");

  produces<HBHERecHitCollection>(hbheTag.label());
  produces<HFRecHitCollection>(hfTag.label());
  produces<HORecHitCollection>(hoTag.label());
  
}


HcalHitSelection::~HcalHitSelection()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HcalHitSelection::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iSetup.get<HcalChannelQualityRcd>().get(theHcalChStatus);
  iSetup.get<HcalSeverityLevelComputerRcd>().get(theHcalSevLvlComputer);

  edm::Handle<HBHERecHitCollection> hbhe;
  edm::Handle<HFRecHitCollection> hf;
  edm::Handle<HORecHitCollection> ho;

  iEvent.getByLabel(hbheTag,hbhe);
  iEvent.getByLabel(hfTag,hf);
  iEvent.getByLabel(hoTag,ho);

  toBeKept.clear();
  edm::Handle<DetIdCollection > detId;
  for( unsigned int t = 0; t < interestingDetIdCollections.size(); ++t )
    {
      iEvent.getByLabel(interestingDetIdCollections[t],detId);
      if (!detId.isValid()){
	edm::LogError("MissingInput")<<"the collection of interesting detIds:"<<interestingDetIdCollections[t]<<" is not found.";
	continue;
      }
      toBeKept.insert(detId->begin(),detId->end());
    }

  std::auto_ptr<HBHERecHitCollection> hbhe_out(new HBHERecHitCollection());
  skim(hbhe,hbhe_out);
  iEvent.put(hbhe_out,hbheTag.label());

  std::auto_ptr<HFRecHitCollection> hf_out(new HFRecHitCollection());
  skim(hf,hf_out);
  iEvent.put(hf_out,hfTag.label());

  std::auto_ptr<HORecHitCollection> ho_out(new HORecHitCollection());
  skim(ho,ho_out,hoSeverityLevel);
  iEvent.put(ho_out,hoTag.label());
  
}

// ------------ method called once each job just before starting event loop  ------------
void 
HcalHitSelection::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HcalHitSelection::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalHitSelection);
