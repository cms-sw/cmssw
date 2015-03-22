#include "RecoEgamma/EgammaIsolationAlgos/interface/HcalPFClusterIsolation.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include <DataFormats/Math/interface/deltaR.h>

template<typename T1>
HcalPFClusterIsolation<T1>::HcalPFClusterIsolation(double drMax,
						   double drVetoBarrel,
						   double drVetoEndcap,
						   double etaStripBarrel,
						   double etaStripEndcap,
						   double energyBarrel,
						   double energyEndcap,
						   bool useEt):
  drMax_(drMax),
  drVetoBarrel_(drVetoBarrel),
  drVetoEndcap_(drVetoEndcap),
  etaStripBarrel_(etaStripBarrel),
  etaStripEndcap_(etaStripEndcap),
  energyBarrel_(energyBarrel),
  energyEndcap_(energyEndcap),
  useEt_(useEt)
{}

template<typename T1>
HcalPFClusterIsolation<T1>::~HcalPFClusterIsolation() 
{}

template<typename T1>
double HcalPFClusterIsolation<T1>::getSum(const T1Ref candRef, const std::vector<edm::Handle<reco::PFClusterCollection>>& clusterHandles) { 
  
  double etSum = 0.;

  float etaStrip = 0;
  float dRVeto = 0;
  if (fabs(candRef->eta()) < 1.479) {
    dRVeto = drVetoBarrel_;
    etaStrip = etaStripBarrel_;
  } else {
    dRVeto = drVetoEndcap_;
    etaStrip = etaStripEndcap_;
  }
  

  for (unsigned int nHandle=0; nHandle<clusterHandles.size(); nHandle++) {
    for(unsigned i=0; i<clusterHandles[nHandle]->size(); i++) {
      const reco::PFClusterRef pfclu(clusterHandles[nHandle], i);
    
      if (fabs(candRef->eta()) < 1.479) {
	if (fabs(pfclu->pt()) < energyBarrel_)
	  continue;
      } else {
	if (fabs(pfclu->energy()) < energyEndcap_)
	  continue;
      }
      
      float dEta = fabs(candRef->eta() - pfclu->eta());
      if(dEta < etaStrip) 
	continue;
      
      float dR = deltaR(candRef->eta(), candRef->phi(), pfclu->eta(), pfclu->phi());
      if(dR > drMax_ || dR < dRVeto) 
	continue;
      
      if (useEt_)
	etSum += pfclu->pt();
      else
	etSum += pfclu->energy();
    }
  }
  
  return etSum;
}

template class HcalPFClusterIsolation<reco::RecoEcalCandidate>;
template class HcalPFClusterIsolation<reco::RecoChargedCandidate>;
template class HcalPFClusterIsolation<reco::Photon>;
template class HcalPFClusterIsolation<reco::GsfElectron>;
