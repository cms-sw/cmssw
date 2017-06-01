#include "RecoTauTag/HLTProducers/interface/L1TCaloJetsMatching.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

//
// function that splits a jet collection into two 
//
std::pair<reco::CaloJetCollection,reco::CaloJetCollection> categorise(const reco::CaloJetCollection& matchedJets, double pt1, double pt2, double m_jj)
{
  std::pair<reco::CaloJetCollection,reco::CaloJetCollection> output;
  
  unsigned int iMax = 0;
  unsigned int jMax = 0;
  double mjjMax = 0;

  if (matchedJets.size() > 1){
    for (unsigned int i = 0; i < matchedJets.size()-1; i++){
      for (unsigned int j = i+1; j < matchedJets.size(); j++){

        const reco::CaloJet &  myJet1 = (matchedJets)[i];
        const reco::CaloJet &  myJet2 = (matchedJets)[j];
        
        double mjj = (myJet1.p4()+myJet2.p4()).M();
        if (mjj > mjjMax){
          mjjMax = mjj;
          iMax = i;
          jMax = j;
        }
      }
    } 
    const reco::CaloJet &  myJet1 = (matchedJets)[iMax];
    const reco::CaloJet &  myJet2 = (matchedJets)[jMax];
        
    if ((myJet1.pt() > pt1) && (myJet2.pt() > pt2) && (mjjMax > m_jj)){
      output.first.push_back(myJet1);
      output.first.push_back(myJet2);
    }
    
    else if ((myJet1.pt() < pt1) && (myJet1.pt() > pt2) && (myJet2.pt() > pt2) && (mjjMax > m_jj)){
      const reco::CaloJet &  myJet3 = (matchedJets)[0];
      if (myJet3.pt()>pt1){
        output.second.push_back(myJet1);
        output.second.push_back(myJet2);
        output.second.push_back(myJet3);
      }
    }    
  }
  return output;    
}

//
// class declaration
//
L1TCaloJetsMatching::L1TCaloJetsMatching(const edm::ParameterSet& iConfig):
  jetSrc_    ( consumes<reco::CaloJetCollection>             (iConfig.getParameter<edm::InputTag>("JetSrc"      ) ) ),
  jetTrigger_( consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("L1JetTrigger") ) ),
  mPt1_Min   ( iConfig.getParameter<double>("Pt1Min")),
  mPt2_Min   ( iConfig.getParameter<double>("Pt2Min")),
  mMjj_Min   ( iConfig.getParameter<double>("MjjMin"))
{  
  produces<reco::CaloJetCollection>("TwoJets");
  produces<reco::CaloJetCollection>("ThreeJets");
}
L1TCaloJetsMatching::~L1TCaloJetsMatching(){ }

void L1TCaloJetsMatching::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const
{    
  std::unique_ptr<reco::CaloJetCollection> caloMatchedJets(new reco::CaloJetCollection);
  
  double deltaR2    = 1.0;
  double matchingR2 = 0.25;
  
  edm::Handle<reco::CaloJetCollection > caloJets;
  iEvent.getByToken(jetSrc_, caloJets);
        
  edm::Handle<trigger::TriggerFilterObjectWithRefs> l1TriggeredJets;
  iEvent.getByToken(jetTrigger_,l1TriggeredJets);
                
  l1t::JetVectorRef jetCandRefVec;
  l1TriggeredJets->getObjects( trigger::TriggerL1Jet,jetCandRefVec);
        
  for(unsigned int iJet = 0; iJet < caloJets->size(); iJet++){
    for(unsigned int iL1Jet = 0; iL1Jet < jetCandRefVec.size(); iL1Jet++){
      const reco::CaloJet &  myJet = (*caloJets)[iJet];
      deltaR2 = ROOT::Math::VectorUtil::DeltaR2(myJet.p4().Vect(), (jetCandRefVec[iL1Jet]->p4()).Vect());
      if(deltaR2 < matchingR2 ) {
        caloMatchedJets->push_back(myJet);
        break;
      }
    }
  } 
   
  std::pair<reco::CaloJetCollection,reco::CaloJetCollection> pairJets = categorise(*caloMatchedJets, mPt1_Min, mPt2_Min, mMjj_Min);
  std::unique_ptr<reco::CaloJetCollection> twoJets(new reco::CaloJetCollection(pairJets.first));
  std::unique_ptr<reco::CaloJetCollection> threeJets(new reco::CaloJetCollection(pairJets.second));
    
  iEvent.put(std::move(twoJets),"TwoJets");
  iEvent.put(std::move(threeJets),"ThreeJets");
}

void L1TCaloJetsMatching::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("L1JetTrigger", edm::InputTag("hltL1VBFDiJet"))->setComment("Name of trigger filter"    );
  desc.add<edm::InputTag>("JetSrc"      , edm::InputTag("hltAK4CaloJetsCorrectedIDPassed"))->setComment("Input collection of CaloJets");
  desc.add<double>       ("Pt1Min",95.0)->setComment("Minimal pT1 of CaloJets to match");
  desc.add<double>       ("Pt2Min",35.0)->setComment("Minimal pT2 of CaloJets to match");
  desc.add<double>       ("MjjMin",650.0)->setComment("Minimal mjj of matched PFjets");
  descriptions.setComment("This module produces collections of CaloJets matched to L1Jets.");
  descriptions.add       ("L1TCaloJetsMatching",desc);
}
