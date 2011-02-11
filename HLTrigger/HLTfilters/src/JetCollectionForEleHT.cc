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
// $Id: JetCollectionForEleHT.cc,v 1.2 2011/01/14 09:03:18 eperez Exp $
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

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"


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
  desc.add<edm::InputTag>("HltElectronTag",edm::InputTag("hltL1NonIsoHLTNonIsoDoubleElectronEt8HT70PixelMatchFilter"));
  desc.add<edm::InputTag>("SourceJetTag",edm::InputTag("hltMCJetCorJetIcone5HF07"));
  desc.add<double>("minDeltaR",0.5);
  descriptions.add("hltJetCollectionForEleHT",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
JetCollectionForEleHT::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel(hltElectronTag,PrevFilterOutput);

  std::vector<edm::Ref<reco::RecoEcalCandidateCollection> > recoecalcands;
  PrevFilterOutput->getObjects(trigger::TriggerCluster, recoecalcands);
 
  edm::Handle<reco::CaloJetCollection> theCaloJetCollectionHandle;
  iEvent.getByLabel(sourceJetTag, theCaloJetCollectionHandle);
  const reco::CaloJetCollection* theCaloJetCollection = theCaloJetCollectionHandle.product();

  std::auto_ptr< reco::CaloJetCollection >  theFilteredCaloJetCollection(new reco::CaloJetCollection);
  
  bool isOverlapping;
 
  for(unsigned int j=0; j<theCaloJetCollection->size(); j++) {

    isOverlapping = false;
    for(unsigned int i=0; i<recoecalcands.size(); i++) {
      reco::SuperClusterRef theHltEleSC = recoecalcands[i]->superCluster();
      TVector3 EleP(theHltEleSC->position().x(), theHltEleSC->position().y(), theHltEleSC->position().z());
      TVector3 JetP((*theCaloJetCollection)[j].px(), (*theCaloJetCollection)[j].py(), (*theCaloJetCollection)[j].pz());
      double DR = EleP.DeltaR(JetP);
      if(DR<minDeltaR_) {
	isOverlapping = true;
	break;
      }
    }
   
    if(!isOverlapping) theFilteredCaloJetCollection->push_back((*theCaloJetCollection)[j]);
  }


  //do the filtering


  iEvent.put(theFilteredCaloJetCollection);


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
