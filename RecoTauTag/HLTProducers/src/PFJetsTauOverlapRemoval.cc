#include "RecoTauTag/HLTProducers/interface/PFJetsTauOverlapRemoval.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/TauReco/interface/PFTau.h"

//
// class declaration
//
using namespace reco   ;
using namespace std    ;
using namespace edm    ;
using namespace trigger;

PFJetsTauOverlapRemoval::PFJetsTauOverlapRemoval(const edm::ParameterSet& iConfig):
  tauSrc_    ( consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<InputTag>("TauSrc"      ) ) ),
  pfJetSrc_  ( consumes<PFJetCollection>(iConfig.getParameter<InputTag>("PFJetSrc") ) )
{  
  produces<PFJetCollection>();
}
PFJetsTauOverlapRemoval::~PFJetsTauOverlapRemoval(){ }

void PFJetsTauOverlapRemoval::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const
{
    
  unique_ptr<PFJetCollection> cleanedPFJets(new PFJetCollection);
    
  double deltaR2   = 1.0;
  double matchingR2 = 0.25;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> tauJets;
  iEvent.getByToken( tauSrc_, tauJets );
  
  edm::Handle<PFJetCollection> PFJets;
  iEvent.getByToken(pfJetSrc_,PFJets);
                
  trigger::VRpftau taus; 
  tauJets->getObjects(trigger::TriggerTau,taus);

//  if(PFJets->size() == 2){
//    for(unsigned int iJet = 0; iJet < PFJets->size(); iJet++){
//      bool isMatched = false;  
//      for(unsigned int iTau = 0; iTau < taus.size(); iTau++){  
//        const PFJet &  myPFJet = (*PFJets)[iJet];
//        deltaR2 = ROOT::Math::VectorUtil::DeltaR2((taus[iTau])->p4().Vect(), myPFJet.p4().Vect());
//        if(deltaR2 < matchingR2){
//          isMatched = true;
//          break;
//        }
//      if(isMatched == false) cleanedPFJets->push_back(myPFJet);
//      }
//    }
//  }
  
 
  //trying something new - combining both cases in one taking only first two jets for matching
  //and then keeping third or highest pt jet
  
  for(unsigned int iJet = 0; iJet < 3; iJet++){
    bool isMatched = false;  
    for(unsigned int iTau = 0; iTau < taus.size(); iTau++){  
      const PFJet &  myPFJet = (*PFJets)[iJet];
      deltaR2 = ROOT::Math::VectorUtil::DeltaR2((taus[iTau])->p4().Vect(), myPFJet.p4().Vect());
      if(deltaR2 < matchingR2){
        isMatched = true;
        break;
      }
    }
    if(isMatched == false) cleanedPFJets->push_back(myPFJet);
  }

  cleanedPFJets->push_back((*PFJets)[2]);
  
  



//  else if(PFJets->size() == 3){
//    for(unsigned int iJet = 0; iJet < PFJets->size()-1; iJet++){
//      bool isMatched = false;  
//      for(unsigned int iTau = 0; iTau < taus.size(); iTau++){  
//        const PFJet &  myPFJet = (*PFJets)[iJet];
//        deltaR2 = ROOT::Math::VectorUtil::DeltaR2((taus[iTau])->p4().Vect(), myPFJet.p4().Vect());
//        if(deltaR2 < matchingR2){
//          isMatched = true;
//          break;
//        }
//      if(isMatched == false) cleanedPFJets->push_back(myPFJet);
//      }
//    }
//    cleanedPFJets->push_back((*PFJets)[2]);
//  }
// 
//  else if(PFJets->size() == 3){
//    for(unsigned int iJet = 0; iJet < PFJets->size()-1; iJet++){
//      bool isMatched = false;  
//      for(unsigned int iTau = 0; iTau < taus.size(); iTau++){  
//        const PFJet &  myPFJet = (*PFJets)[iJet];
//        deltaR2 = ROOT::Math::VectorUtil::DeltaR2((taus[iTau])->p4().Vect(), myPFJet.p4().Vect());
//        if(deltaR2 < matchingR2){
//          isMatched = true;
//          break;
//        }
//      if(isMatched == false) cleanedPFJets->push_back(myPFJet);
//      }
//    }
//    cleanedPFJets->push_back((*PFJets)[2]);
//  }
  
  iEvent.put(std::move(cleanedPFJets));
}

void PFJetsTauOverlapRemoval::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("PFJetSrc", edm::InputTag("hltAK4PFJetsCorrected"                     ))->setComment("Input collection of PFJets"    );
  desc.add<edm::InputTag>("TauSrc"      , edm::InputTag("hltPFTau20TrackLooseIso"))->setComment("Input collection of PFTaus that have passed ID and isolation requirements");
  descriptions.setComment("This module produces a collection of PFJets that are cross-cleaned with respect to PFTaus passing a HLT filter.");
  descriptions.add       ("PFJetsTauOverlapRemoval",desc);
}
