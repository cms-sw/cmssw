// makes CaloTowerCandidates from CaloTowers
// original author: L.Lista INFN, modifyed by: F.Ratnikov UMd 
// Author for regionality A. Nikitenko
#include <cmath>
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerCreatorForTauHLT.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace edm;
using namespace reco;
using namespace std;
using namespace l1extra ;

CaloTowerCreatorForTauHLT::CaloTowerCreatorForTauHLT( const ParameterSet & p ) 
  :
  mVerbose (p.getUntrackedParameter<int> ("verbose", 0)),
  mtowers (p.getParameter<string> ("towers")),
  mCone (p.getParameter<double> ("UseTowersInCone")),
  mTauTrigger (p.getParameter<string> ("TauTrigger")),
  ml1seeds (p.getParameter<InputTag> ("l1seeds")),
  mEtThreshold (p.getParameter<double> ("minimumEt")),
  mEThreshold (p.getParameter<double> ("minimumE")),
  mTauId (p.getParameter<int> ("TauId"))
{
  produces<CandidateCollection>();
}

CaloTowerCreatorForTauHLT::~CaloTowerCreatorForTauHLT() {
}

void CaloTowerCreatorForTauHLT::produce( Event& evt, const EventSetup& ) {
  Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel( mtowers, caloTowers );

  // imitate L1 seeds
  InputTag tauJetInputTag( ml1seeds.label(), mTauTrigger ) ;
  Handle<L1JetParticleCollection> jetsgen;
  evt.getByLabel(tauJetInputTag, jetsgen);
  //   cout <<"Size of the jetgen "<<jetsgen->size()<<endl;  
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( caloTowers->size() );
  
  int idTau =0;
  L1JetParticleCollection::const_iterator myL1Jet = jetsgen->begin();
  for(;myL1Jet != jetsgen->end();myL1Jet++)
    {
      if(idTau == mTauId)
	{
	  double Sum08 = 0.;
	  
	  if (mVerbose == 3) {
	    std::cout <<" Generated jet et = " << (*myL1Jet).et()
		      <<" eta = " << myL1Jet->eta()
		      <<" phi = " << myL1Jet->phi() << endl;
	  }
	  unsigned idx = 0;
	  for (; idx < caloTowers->size (); idx++) {
	    const CaloTower* cal = &((*caloTowers) [idx]);
	    if (mVerbose == 2) {
	      std::cout << "CaloTwoerCreatorForTauHLT::produce-> " << idx << " tower et/eta/phi/e: " 
			<< cal->et() << '/' << cal->eta() << '/' << cal->phi() << '/' << cal->energy() << " is...";
	    }
	    if (cal->et() >= mEtThreshold && cal->energy() >= mEThreshold ) {
	      
	      math::PtEtaPhiELorentzVector p( cal->et(), cal->eta(), cal->phi(), cal->energy() );
  
	      
	      double delta  = ROOT::Math::VectorUtil::DeltaR((*myL1Jet).p4().Vect(), p);
	      
	      if(delta < mCone) {
		
		RecoCaloTowerCandidate * c = 
		  new RecoCaloTowerCandidate( 0, Candidate::LorentzVector( p ) );
		c->setCaloTower (CaloTowerRef( caloTowers, idx) );
		cands->push_back( c );
		Sum08 += c->pt(); 
		if (mVerbose == 3) std::cout << "accepted: pT/eta/phi:" 
					     << c->pt() << '/' 
					     << c->eta() <<  '/' 
					     << c->phi()
					     <<" emEt()= " << (*caloTowers)[idx].emEt() 
					     <<" ehEt()= " << (*caloTowers)[idx].hadEt() 
					     <<" deltar= " << delta 
					     << " Sum08= " << Sum08 << std::endl;
	      }
	    }
	    else {
	      if (mVerbose == 2) std::cout << "rejected " << std::endl;
	    }
	  }
	  if (mVerbose == 3) {
	    std::cout << "CaloTowerCreatorForTauHLT::produce-> " << cands->size () << " candidates created" << std::endl;
	    std::cout << " Sum08 = " << Sum08 << endl;
	  }
	}
      idTau++;
    }
  
  evt.put( cands );
  
}
