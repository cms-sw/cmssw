#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiFwd.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

#include "DataFormats/L1THGCal/interface/ClusterShapes.h"

namespace DataFormats {
  namespace L1THGCal {
    l1t::HGCFETriggerDigi hgcfetd;
    l1t::HGCFETriggerDigiCollection c_hgcfetd;
    edm::Wrapper<l1t::HGCFETriggerDigiCollection> w_c_hgcfetd;

    l1t::HGCFETriggerDigiRef r_hgcfetd;
    l1t::HGCFETriggerDigiPtr p_hgcfetd;

    l1t::HGCFETriggerDigiRefVector v_r_hgcfetd;
    l1t::HGCFETriggerDigiPtrVector v_p_hgcfetd;

    edm::Wrapper<l1t::HGCFETriggerDigiRefVector> w_v_r_hgcfetd;
    edm::Wrapper<l1t::HGCFETriggerDigiPtrVector> w_v_p_hgcfetd;

    l1t::HGCalTowerBxCollection   hgcalTowerBxColl;
    l1t::HGCalTriggerCellBxCollection   hgcalTriggerCellBxColl;
    l1t::HGCalClusterBxCollection hgcalClusterBxColl;
    l1t::HGCalMulticlusterBxCollection hgcalMulticlusterBxColl;
    edm::Ptr<l1t::HGCalTriggerCell> hgcalTriggerCellPtr;
    edm::Ptr<l1t::HGCalCluster> hgcalClusterPtr;
    edm::PtrVector<l1t::HGCalTriggerCell>  hgcalTriggerCellList;
    edm::PtrVector<l1t::HGCalCluster>  hgcalClusterList;

    edm::Wrapper<l1t::HGCalTowerBxCollection>   w_hgcalTowerBxColl;
    edm::Wrapper<l1t::HGCalClusterBxCollection> w_hgcalClusterBxColl;
    edm::Wrapper<l1t::HGCalMulticlusterBxCollection> w_hgcalMulticlusterBxColl;
    edm::Wrapper<l1t::HGCalTriggerCellBxCollection>   w_hgcalTriggerCellBxColl;

    edm::Wrapper<edm::Ptr<l1t::HGCalTriggerCell>> w_hgcalTriggerCellPtr;
    edm::Wrapper<edm::PtrVector<l1t::HGCalTriggerCell>> w_hgcalTriggerCellList;
    edm::Wrapper<edm::Ptr<l1t::HGCalCluster>> w_hgcalClusterPtr;
    edm::Wrapper<edm::PtrVector<l1t::HGCalCluster>> w_hgcalClusterList;
    
    l1t::ClusterShapes clusterShapes;
  }
}
