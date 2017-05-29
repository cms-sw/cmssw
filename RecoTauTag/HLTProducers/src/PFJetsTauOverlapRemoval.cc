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
  tauSrc    ( consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<InputTag>("TauSrc"      ) ) ),
  PFJetSrc( consumes<PFJetCollection>(iConfig.getParameter<InputTag>("PFJetSrc") ) )
{  
  produces<PFJetCollection>();
}
PFJetsTauOverlapRemoval::~PFJetsTauOverlapRemoval(){ }

void PFJetsTauOverlapRemoval::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const
{
    
  unique_ptr<PFJetCollection> cleanedPFJets(new PFJetCollection);
    
  double deltaR    = 1.0;
  double matchingR = 0.5;
  
  edm::Handle<trigger::TriggerFilterObjectWithRefs> tauJets;
  iEvent.getByToken( tauSrc, tauJets );
  
  edm::Handle<PFJetCollection> PFJets;
  iEvent.getByToken(PFJetSrc,PFJets);
                
  trigger::VRpftau taus; 
  tauJets->getObjects(trigger::TriggerTau,taus);

  if(PFJets->size() == 2 || PFJets->size() > 3){
    for(unsigned int iTau = 0; iTau < taus.size(); iTau++){  
      for(unsigned int iJet = 0; iJet < PFJets->size(); iJet++){
        const PFJet &  myPFJet = (*PFJets)[iJet];
        deltaR = ROOT::Math::VectorUtil::DeltaR((taus[iTau])->p4().Vect(), myPFJet.p4().Vect());
        if(deltaR > matchingR) cleanedPFJets->push_back(myPFJet);
        break;
      }
    }
  }
  else if(PFJets->size() == 3){
    for(unsigned int iTau = 0; iTau < taus.size(); iTau++){  
      for(unsigned int iJet = 0; iJet < PFJets->size()-1; iJet++){
        const PFJet &  myPFJet = (*PFJets)[iJet];
        deltaR = ROOT::Math::VectorUtil::DeltaR((taus[iTau])->p4().Vect(), myPFJet.p4().Vect());
        if(deltaR > matchingR) cleanedPFJets->push_back(myPFJet);
        break;
      }
    }
    if(PFJets->size() > 2) cleanedPFJets->push_back((*PFJets)[2]);
  }
 
  
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
