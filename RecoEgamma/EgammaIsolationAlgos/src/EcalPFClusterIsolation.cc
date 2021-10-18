//*****************************************************************************
// File:      EgammaRecHitIsolation.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer, hacked by Sam Harper (ie the ugly stuff is mine)
// Institute: IIHE-VUB, RAL
//=============================================================================
//*****************************************************************************

#include "RecoEgamma/EgammaIsolationAlgos/interface/EcalPFClusterIsolation.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include <DataFormats/Math/interface/deltaR.h>

template <typename T1>
EcalPFClusterIsolation<T1>::EcalPFClusterIsolation(double drMax,
                                                   double drVetoBarrel,
                                                   double drVetoEndcap,
                                                   double etaStripBarrel,
                                                   double etaStripEndcap,
                                                   double energyBarrel,
                                                   double energyEndcap)
    : drMax_(drMax),
      drVetoBarrel_(drVetoBarrel),
      drVetoEndcap_(drVetoEndcap),
      etaStripBarrel_(etaStripBarrel),
      etaStripEndcap_(etaStripEndcap),
      energyBarrel_(energyBarrel),
      energyEndcap_(energyEndcap) {}

template <typename T1>
EcalPFClusterIsolation<T1>::~EcalPFClusterIsolation() {}

template <typename T1>
bool EcalPFClusterIsolation<T1>::computedRVeto(T1Ref candRef, reco::PFClusterRef pfclu) {
  float dR2 = deltaR2(candRef->eta(), candRef->phi(), pfclu->eta(), pfclu->phi());
  if (dR2 > (drMax_ * drMax_))
    return false;

  if (candRef->superCluster().isNonnull()) {
    // Exclude clusters that are part of the candidate
    for (reco::CaloCluster_iterator it = candRef->superCluster()->clustersBegin();
         it != candRef->superCluster()->clustersEnd();
         ++it) {
      if ((*it)->seed() == pfclu->seed()) {
        return false;
      }
    }
  }

  return true;
}

template <>
bool EcalPFClusterIsolation<reco::RecoChargedCandidate>::computedRVeto(T1Ref candRef, reco::PFClusterRef pfclu) {
  float dR2 = deltaR2(candRef->eta(), candRef->phi(), pfclu->eta(), pfclu->phi());
  if (dR2 > (drMax_ * drMax_) || dR2 < drVeto2_)
    return false;
  else
    return true;
}

template <typename T1>
double EcalPFClusterIsolation<T1>::getSum(const T1 cand, edm::Handle<reco::PFClusterCollection> clusterHandle) {
  drVeto2_ = -1.;
  float etaStrip = -1;

  if (std::abs(cand.eta()) < 1.479) {
    drVeto2_ = drVetoBarrel_ * drVetoBarrel_;
    etaStrip = etaStripBarrel_;
  } else {
    drVeto2_ = drVetoEndcap_ * drVetoEndcap_;
    etaStrip = etaStripEndcap_;
  }

  float etSum = 0;
  for (size_t i = 0; i < clusterHandle->size(); i++) {
    reco::PFClusterRef pfclu(clusterHandle, i);

    if (std::abs(cand.eta()) < 1.479) {
      if (std::abs(pfclu->pt()) < energyBarrel_)
        continue;
    } else {
      if (std::abs(pfclu->energy()) < energyEndcap_)
        continue;
    }

    float dEta = std::abs(cand.eta() - pfclu->eta());
    if (dEta < etaStrip)
      continue;
    if (not computedRVeto(cand, pfclu))
      continue;

    etSum += pfclu->pt();
  }

  return etSum;
}

template <typename T1>
double EcalPFClusterIsolation<T1>::getSum(T1Ref ref, edm::Handle<std::vector<reco::PFCluster> > clusts) {
  return getSum(*ref, clusts);
}

template <typename T1>
bool EcalPFClusterIsolation<T1>::computedRVeto(T1 cand, reco::PFClusterRef pfclu) {
  float dR2 = deltaR2(cand.eta(), cand.phi(), pfclu->eta(), pfclu->phi());
  if (dR2 > (drMax_ * drMax_))
    return false;

  if (cand.superCluster().isNonnull()) {
    // Exclude clusters that are part of the candidate
    for (reco::CaloCluster_iterator it = cand.superCluster()->clustersBegin(); it != cand.superCluster()->clustersEnd();
         ++it) {
      if ((*it)->seed() == pfclu->seed()) {
        return false;
      }
    }
  }

  return true;
}

template class EcalPFClusterIsolation<reco::RecoEcalCandidate>;
template class EcalPFClusterIsolation<reco::RecoChargedCandidate>;
template class EcalPFClusterIsolation<reco::Photon>;
template class EcalPFClusterIsolation<reco::GsfElectron>;
