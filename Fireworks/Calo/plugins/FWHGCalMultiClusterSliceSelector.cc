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

FWHGCalMultiClusterSliceSelector::FWHGCalMultiClusterSliceSelector(TH2F *h, const FWEventItem *i) : FWHistSliceSelector(h, i), m_helper(typeid(reco::HGCalMultiCluster))
{
}

FWHGCalMultiClusterSliceSelector::~FWHGCalMultiClusterSliceSelector()
{
}

void FWHGCalMultiClusterSliceSelector::getItemEntryEtaPhi(int itemIdx, float &eta, float &phi) const
{
    const void *modelData = m_item->modelData(itemIdx);
    reco::HGCalMultiCluster tower  = *reinterpret_cast<const reco::HGCalMultiCluster*>(m_helper.offsetObject(modelData));

    eta = tower.eta();
    phi = tower.phi();
}
