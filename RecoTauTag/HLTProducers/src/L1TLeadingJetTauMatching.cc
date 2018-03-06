#include "RecoTauTag/HLTProducers/interface/L1TLeadingJetTauMatching.h"
#include "Math/GenVector/VectorUtil.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/Math/interface/deltaR.h"

//
// class declaration
//
L1TLeadingJetTauMatching::L1TLeadingJetTauMatching(const edm::ParameterSet& iConfig):
    tauSrc_    ( consumes<reco::PFTauCollection>(iConfig.getParameter<edm::InputTag>("TauSrc"      ) ) ),
    L1JetSrc_  ( consumes<trigger::TriggerFilterObjectWithRefs>(iConfig.getParameter<edm::InputTag>("L1JetSrc") ) ),
    matchingR2_ ( iConfig.getParameter<double>("MatchingdR")*iConfig.getParameter<double>("MatchingdR") ),
    minTauPt_ (iConfig.getParameter<double>("MinTauPt") )
{  
    produces<reco::PFTauCollection>();
}
L1TLeadingJetTauMatching::~L1TLeadingJetTauMatching(){ }

void L1TLeadingJetTauMatching::produce(edm::StreamID iSId, edm::Event& iEvent, const edm::EventSetup& iES) const
{
    std::unique_ptr<reco::PFTauCollection> L1matchedPFTau(new reco::PFTauCollection);
    
    edm::Handle<reco::PFTauCollection> taus;
    iEvent.getByToken(tauSrc_, taus);
  
    edm::Handle<trigger::TriggerFilterObjectWithRefs> L1Jets;
    iEvent.getByToken(L1JetSrc_,L1Jets);
                  
    l1t::JetVectorRef jetCandRefVec;
    L1Jets->getObjects(trigger::TriggerL1Jet,jetCandRefVec);

    for(unsigned int iTau = 0; iTau < taus->size(); iTau++){  

      if ((*taus)[iTau].pt() > minTauPt_){
        if(reco::deltaR2((*taus)[iTau].p4(), jetCandRefVec[0]->p4()) < matchingR2_)
            L1matchedPFTau->push_back((*taus)[iTau]);
      }
    }

    iEvent.put(std::move(L1matchedPFTau));
}

void L1TLeadingJetTauMatching::fillDescriptions(edm::ConfigurationDescriptions& descriptions) 
{
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("L1JetSrc", edm::InputTag("hltL1VBFDiJetOR"))->setComment("Input filter objects passing L1 seed"    );
    desc.add<edm::InputTag>("TauSrc", edm::InputTag("hltSelectedPFTausTrackFindingLooseChargedIsolationAgainstMuon"))->setComment("Input collection of PFTaus");
    desc.add<double>       ("MatchingdR",0.5)->setComment("Maximum dR for matching between PFTaus and L1 filter jets");
    desc.add<double>       ("MinTauPt",20.0)->setComment("PFTaus above this pt will be considered");
    descriptions.setComment("This module produces a collection of PFTaus matched to the leading jet passing the L1 seed filter.");
    descriptions.add       ("L1TLeadingJetTauMatching",desc);
}
