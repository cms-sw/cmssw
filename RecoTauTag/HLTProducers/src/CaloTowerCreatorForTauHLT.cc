// makes CaloTowerCandidates from CaloTowers
// original author: L.Lista INFN, modifyed by: F.Ratnikov UMd 
// Author for regionality A. Nikitenko
// Modified by S. Gennai
#include <cmath>
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTauTag/HLTProducers/interface/CaloTowerCreatorForTauHLT.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
// Math
#include "Math/GenVector/VectorUtil.h"
#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;
using namespace l1extra ;

CaloTowerCreatorForTauHLT::CaloTowerCreatorForTauHLT( const ParameterSet & p ) 
  :
  mVerbose (p.getUntrackedParameter<int> ("verbose", 0)),
  mtowers (p.getParameter<InputTag> ("towers")),
  mCone (p.getParameter<double> ("UseTowersInCone")),
  mTauTrigger (p.getParameter<InputTag> ("TauTrigger")),
//  ml1seeds (p.getParameter<InputTag> ("l1seeds")),
  mEtThreshold (p.getParameter<double> ("minimumEt")),
  mEThreshold (p.getParameter<double> ("minimumE")),
  mTauId (p.getParameter<int> ("TauId"))
{
  produces<CaloTowerCollection>();
}

CaloTowerCreatorForTauHLT::~CaloTowerCreatorForTauHLT() {
}

void CaloTowerCreatorForTauHLT::produce( Event& evt, const EventSetup& ) {
  edm::Handle<CaloTowerCollection> caloTowers;
  evt.getByLabel( mtowers, caloTowers );

  // imitate L1 seeds
  edm::Handle<L1JetParticleCollection> jetsgen;
  evt.getByLabel(mTauTrigger, jetsgen);
  std::auto_ptr<CaloTowerCollection> cands( new CaloTowerCollection );
  cands->reserve( caloTowers->size() );
  
  int idTau =0;
  L1JetParticleCollection::const_iterator myL1Jet = jetsgen->begin();
  for(;myL1Jet != jetsgen->end();myL1Jet++)
    {
      if(idTau == mTauId)
	{
	  double Sum08 = 0.;
	  
	  unsigned idx = 0;
	  for (; idx < caloTowers->size(); idx++) {
	    const CaloTower* cal = &((*caloTowers) [idx]);
	    if (mVerbose == 2) {
	      std::cout << "CaloTowerCreatorForTauHLT::produce-> " << idx << " tower et/eta/phi/e: " 
			<< cal->et() << '/' << cal->eta() << '/' << cal->phi() << '/' << cal->energy() << " is...";
	    }
	    if (cal->et() >= mEtThreshold && cal->energy() >= mEThreshold ) {
	      math::PtEtaPhiELorentzVector p( cal->et(), cal->eta(), cal->phi(), cal->energy() );
  	      double delta  = ROOT::Math::VectorUtil::DeltaR((*myL1Jet).p4().Vect(), p);
	      
	      if(delta < mCone) {
		Sum08 += cal->et(); 
		cands->push_back( *cal );
	      }
	    }
	    else {
	      if (mVerbose == 2) std::cout << "rejected " << std::endl;
	    }
	  }

	}
      idTau++;
    }

  evt.put( cands );
  
}
