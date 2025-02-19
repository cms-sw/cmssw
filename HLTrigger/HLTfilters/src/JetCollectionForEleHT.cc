// -*- C++ -*-
//
// Package:    JetCollectionForEleHT
// Class:      JetCollectionForEleHT
// 
/**\class JetCollectionForEleHT JetCollectionForEleHT.cc HLTrigger/JetCollectionForEleHT/src/JetCollectionForEleHT.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Massimiliano Chiorboli,40 4-A01,+41227671535,
//         Created:  Mon Oct  4 11:57:35 CEST 2010
// $Id: JetCollectionForEleHT.cc,v 1.6 2011/04/28 06:22:41 gruen Exp $
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
#include "HLTrigger/HLTfilters/interface/JetCollectionForEleHT.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"

JetCollectionForEleHT::JetCollectionForEleHT(const edm::ParameterSet& iConfig):
  hltElectronTag(iConfig.getParameter< edm::InputTag > ("HltElectronTag")),
  sourceJetTag(iConfig.getParameter< edm::InputTag > ("SourceJetTag")),
  minDeltaR_(iConfig.getParameter< double > ("minDeltaR"))
{
  produces<reco::CaloJetCollection>();
}



JetCollectionForEleHT::~JetCollectionForEleHT()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}

void
JetCollectionForEleHT::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("HltElectronTag",edm::InputTag("triggerFilterObjectWithRefs"));
  desc.add<edm::InputTag>("SourceJetTag",edm::InputTag("caloJetCollection"));
  desc.add<double>("minDeltaR",0.5);
  descriptions.add("hltJetCollectionForEleHT",desc);
}

//
// member functions
//


// ------------ method called to produce the data  ------------
// template <typename T>
void
JetCollectionForEleHT::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel(hltElectronTag,PrevFilterOutput);
 
  //its easier on the if statement flow if I try everything at once, shouldnt add to timing
  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > clusCands;
  PrevFilterOutput->getObjects(trigger::TriggerCluster,clusCands);
  std::vector<edm::Ref<reco::ElectronCollection> > eleCands;
  PrevFilterOutput->getObjects(trigger::TriggerElectron,eleCands);
  
  //prepare the collection of 3-D vector for electron momenta
  std::vector<TVector3> ElePs;

  if(!clusCands.empty()){ //try trigger cluster
    for(size_t candNr=0;candNr<clusCands.size();candNr++){
      TVector3 positionVector(
          clusCands[candNr]->superCluster()->position().x(),
          clusCands[candNr]->superCluster()->position().y(),
          clusCands[candNr]->superCluster()->position().z());
      ElePs.push_back(positionVector);
    }
  }else if(!eleCands.empty()){ // try trigger electrons
    for(size_t candNr=0;candNr<eleCands.size();candNr++){
      TVector3 positionVector(
          eleCands[candNr]->superCluster()->position().x(),
          eleCands[candNr]->superCluster()->position().y(),
          eleCands[candNr]->superCluster()->position().z());
      ElePs.push_back(positionVector);
    }
  }
  
  edm::Handle<reco::CaloJetCollection> theCaloJetCollectionHandle;
  iEvent.getByLabel(sourceJetTag, theCaloJetCollectionHandle);
  const reco::CaloJetCollection* theCaloJetCollection = theCaloJetCollectionHandle.product();

  std::auto_ptr< reco::CaloJetCollection >  theFilteredCaloJetCollection(new reco::CaloJetCollection);
  
  bool isOverlapping;
 
  for(unsigned int j=0; j<theCaloJetCollection->size(); j++) {

    isOverlapping = false;
    for(unsigned int i=0; i<ElePs.size(); i++) {
      
      TVector3 JetP((*theCaloJetCollection)[j].px(), (*theCaloJetCollection)[j].py(), (*theCaloJetCollection)[j].pz());
      double DR = ElePs[i].DeltaR(JetP);
      
      if(DR<minDeltaR_) {
	      isOverlapping = true;
	      break;
      }
    }
   
    if(!isOverlapping) theFilteredCaloJetCollection->push_back((*theCaloJetCollection)[j]);
  }
  
  //do the filtering

  iEvent.put(theFilteredCaloJetCollection);

  return;

}

// ------------ method called once each job just before starting event loop  ------------
void 
JetCollectionForEleHT::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
JetCollectionForEleHT::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetCollectionForEleHT);

