#include "RecoEgamma/EgammaIsolationAlgos/interface/HcalPFClusterIsolation.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include <DataFormats/Math/interface/deltaR.h>

template <typename T1>
HcalPFClusterIsolation<T1>::HcalPFClusterIsolation(double drMax,
                                                   double drVetoBarrel,
                                                   double drVetoEndcap,
                                                   double etaStripBarrel,
                                                   double etaStripEndcap,
                                                   double energyBarrel,
                                                   double energyEndcap,
                                                   bool useEt)
    : drMax_(drMax),
      drVetoBarrel_(drVetoBarrel),
      drVetoEndcap_(drVetoEndcap),
      etaStripBarrel_(etaStripBarrel),
      etaStripEndcap_(etaStripEndcap),
      energyBarrel_(energyBarrel),
      energyEndcap_(energyEndcap),
      useEt_(useEt) {}

template <typename T1>
HcalPFClusterIsolation<T1>::~HcalPFClusterIsolation() {}

template <typename T1>
double HcalPFClusterIsolation<T1>::getSum(const T1 cand,
                                          const std::vector<edm::Handle<reco::PFClusterCollection>>& clusterHandles) {
  double etSum = 0.;
  double candAbsEta = std::abs(cand.eta());

  float etaStrip = 0;
  float dRVeto = 0;
  if (candAbsEta < 1.479) {
    dRVeto = drVetoBarrel_;
    etaStrip = etaStripBarrel_;
  } else {
    dRVeto = drVetoEndcap_;
    etaStrip = etaStripEndcap_;
  }

  for (unsigned int nHandle = 0; nHandle < clusterHandles.size(); nHandle++) {
    for (unsigned i = 0; i < clusterHandles[nHandle]->size(); i++) {
      const reco::PFClusterRef pfclu(clusterHandles[nHandle], i);

      if (candAbsEta < 1.479) {
        if (std::abs(pfclu->pt()) < energyBarrel_)
          continue;
      } else {
        if (std::abs(pfclu->energy()) < energyEndcap_)
          continue;
      }

      float dEta = std::abs(cand.eta() - pfclu->eta());
      if (dEta < etaStrip)
        continue;

      float dR2 = deltaR2(cand.eta(), cand.phi(), pfclu->eta(), pfclu->phi());
      if (dR2 > (drMax_ * drMax_) || dR2 < (dRVeto * dRVeto))
        continue;

      if (useEt_)
        etSum += pfclu->pt();
      else
        etSum += pfclu->energy();
    }
  }

  return etSum;
}

template <typename T1>
double HcalPFClusterIsolation<T1>::getSum(T1Ref ref,
                                          const std::vector<edm::Handle<reco::PFClusterCollection>>& clusterHandles) {
  return getSum(*ref, clusterHandles);
}

template class HcalPFClusterIsolation<reco::RecoEcalCandidate>;
template class HcalPFClusterIsolation<reco::RecoChargedCandidate>;
template class HcalPFClusterIsolation<reco::Photon>;
template class HcalPFClusterIsolation<reco::GsfElectron>;
