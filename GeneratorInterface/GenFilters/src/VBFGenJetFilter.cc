#include "GeneratorInterface/GenFilters/interface/VBFGenJetFilter.h"

#include "DataFormats/Math/interface/deltaPhi.h"

#include <HepMC/GenVertex.h>

// ROOT includes
#include "TMath.h"

// C++ includes
#include <iostream>

using namespace edm;
using namespace std;


VBFGenJetFilter::VBFGenJetFilter(const edm::ParameterSet& iConfig) :
oppositeHemisphere(iConfig.getUntrackedParameter<bool>  ("oppositeHemisphere",false)),
ptMin             (iConfig.getUntrackedParameter<double>("minPt",                20)),
etaMin            (iConfig.getUntrackedParameter<double>("minEta",             -5.0)),
etaMax            (iConfig.getUntrackedParameter<double>("maxEta",              5.0)),
minInvMass        (iConfig.getUntrackedParameter<double>("minInvMass",          0.0)),
maxInvMass        (iConfig.getUntrackedParameter<double>("maxInvMass",      99999.0)),
minDeltaPhi       (iConfig.getUntrackedParameter<double>("minDeltaPhi",        -1.0)),
maxDeltaPhi       (iConfig.getUntrackedParameter<double>("maxDeltaPhi",     99999.0)),
minDeltaEta       (iConfig.getUntrackedParameter<double>("minDeltaEta",        -1.0)),
maxDeltaEta       (iConfig.getUntrackedParameter<double>("maxDeltaEta",     99999.0))
{
  
  m_inputTag_GenJetCollection = consumes<reco::GenJetCollection>(iConfig.getUntrackedParameter<edm::InputTag>("inputTag_GenJetCollection",edm::InputTag("ak5GenJetsNoNu")));
  
}

VBFGenJetFilter::~VBFGenJetFilter(){
  
}

vector<const reco::GenJet*> VBFGenJetFilter::filterGenJets(const vector<reco::GenJet>* jets){
  
  vector<const reco::GenJet*> out;
  
  for(unsigned i=0; i<jets->size(); i++){
    
    const reco::GenJet* j = &((*jets)[i]);
    
    if(j->p4().pt() >ptMin &&  j->p4().eta()>etaMin && j->p4().eta()<etaMax)
    {
      out.push_back(j);
    }
  }
  
  return out;
}


// ------------ method called to skim the data  ------------
bool VBFGenJetFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  Handle< vector<reco::GenJet> > handleGenJets;
  iEvent.getByToken(m_inputTag_GenJetCollection, handleGenJets);
  const vector<reco::GenJet>* genJets = handleGenJets.product();
  
  // Getting filtered generator jets
  vector<const reco::GenJet*> filGenJets = filterGenJets(genJets);
  
  // If we do not find at least 2 jets veto the event
  if(filGenJets.size()<2){return false;}
  
  for(unsigned a=0; a<filGenJets.size(); a++){
    for(unsigned b=a+1; b<filGenJets.size(); b++){    
      
      const reco::GenJet* pA = filGenJets[a];
      const reco::GenJet* pB = filGenJets[b];
      
      // Getting the dijet vector
      math::XYZTLorentzVector diJet = pA->p4() + pB->p4();
      
      // Testing opposite hemispheres
      double dijetProd = pA->p4().eta()*pB->p4().eta();
      if(oppositeHemisphere && dijetProd>=0){continue;}
      
      // Testing dijet mass
      double invMass = diJet.mass();
      if(invMass<=minInvMass || invMass>maxInvMass){continue;}
      
      // Testing dijet delta eta
      double dEta = fabs(pA->p4().eta()-pB->p4().eta());
      if(dEta<=minDeltaEta || dEta>maxDeltaEta){continue;}

      // Testing dijet delta phi
      double dPhi = fabs(reco::deltaPhi(pA->p4().phi(),pB->p4().phi()));
      if(dPhi<=minDeltaPhi || dPhi>maxDeltaPhi){continue;}
      
      return true;
    }
  }

  return false;
}

//define this as a plug-in
DEFINE_FWK_MODULE(VBFGenJetFilter);
