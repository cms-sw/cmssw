// makes CaloTowerCandidates from CaloTowers
// original author: M. Sani (UCSD)

#include <cmath>
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTCaloTowerProducer.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;
using namespace l1extra ;

EgammaHLTCaloTowerProducer::EgammaHLTCaloTowerProducer( const ParameterSet & p ) : towers_ (consumes<CaloTowerCollection>(p.getParameter<InputTag> ("towerCollection"))),
										   cone_ (p.getParameter<double> ("useTowersInCone")),
										   l1isoseeds_ (consumes<edm::View<reco::Candidate>>(p.getParameter< edm::InputTag > ("L1IsoCand"))),
										   l1nonisoseeds_ (consumes<edm::View<reco::Candidate>>(p.getParameter< edm::InputTag > ("L1NonIsoCand"))),
										   EtThreshold_ (p.getParameter<double> ("EtMin")),
										   EThreshold_ (p.getParameter<double> ("EMin")) {
  
  produces<CaloTowerCollection>();
}

void EgammaHLTCaloTowerProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  
  desc.add<edm::InputTag>(("towerCollection"), edm::InputTag("hltRecoEcalCandidate"));
  desc.add<edm::InputTag>(("L1IsoCand"), edm::InputTag("hltTowerMakerForAll"));
  desc.add<edm::InputTag>(("L1NonIsoCand"), edm::InputTag("fixedGridRhoFastjetAllCalo"));
  desc.add<double>(("useTowersInCone"), 0.8); 
  desc.add<double>(("EtMin"), 1.0); 
  desc.add<double>(("EMin"), 1.0); 
  descriptions.add(("hltCaloTowerForEgamma"), desc);  
}


void EgammaHLTCaloTowerProducer::produce(edm::StreamID, edm::Event & evt, edm::EventSetup const &) const
{
  edm::Handle<CaloTowerCollection> caloTowers;
  evt.getByToken(towers_, caloTowers);

  edm::Handle<edm::View<reco::Candidate>> emIsolColl;
  evt.getByToken(l1isoseeds_, emIsolColl);
  edm::Handle<edm::View<reco::Candidate> > emNonIsolColl;
  evt.getByToken(l1nonisoseeds_, emNonIsolColl);
  std::auto_ptr<CaloTowerCollection> cands(new CaloTowerCollection);
  cands->reserve(caloTowers->size());

  for (unsigned idx = 0; idx < caloTowers->size(); idx++) {
    const CaloTower* cal = &((*caloTowers) [idx]);
    if (cal->et() >= EtThreshold_ && cal->energy() >= EThreshold_) {
      bool fill = false;
      math::PtEtaPhiELorentzVector p(cal->et(), cal->eta(), cal->phi(), cal->energy());
      for (edm::View<reco::Candidate>::const_iterator emItr = emIsolColl->begin(); emItr != emIsolColl->end() ;++emItr) {
	double delta  = ROOT::Math::VectorUtil::DeltaR((*emItr).p4().Vect(), p);
	if(delta < cone_) {
	  cands->push_back(*cal);
	  fill = true;
	  break;
	}
      }
      
      if (!fill) {
	for(edm::View<reco::Candidate>::const_iterator emItr = emNonIsolColl->begin(); emItr != emNonIsolColl->end() ;++emItr) {
	  double delta  = ROOT::Math::VectorUtil::DeltaR((*emItr).p4().Vect(), p);
	  if(delta < cone_) {
	    cands->push_back(*cal);
	    break;
	  }
	}
      }
    }
  }

  evt.put( cands );  
}
