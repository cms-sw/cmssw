// system include files
#include <typeinfo>

// user include files
#include "TH2F.h"
#include "TMath.h"
#include "Fireworks/Calo/plugins/FWHGCalMultiClusterSliceSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDefs.h"

#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

FWHGCalMultiClusterSliceSelector::FWHGCalMultiClusterSliceSelector(TH2F *h, const FWEventItem *i)
    : FWHistSliceSelector(h, i) {}

FWHGCalMultiClusterSliceSelector::~FWHGCalMultiClusterSliceSelector() {}

void FWHGCalMultiClusterSliceSelector::getItemEntryEtaPhi(int itemIdx, float &eta, float &phi) const {
  const std::vector<reco::HGCalMultiCluster> *towers = nullptr;
  m_item->get(towers);
  assert(nullptr != towers);

  std::vector<reco::HGCalMultiCluster>::const_iterator tower = towers->begin();
  std::advance(tower, itemIdx);

  eta = tower->eta();
  phi = tower->phi();
}
