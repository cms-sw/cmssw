#include "DataFormats/L1THGCal/interface/HGCFETriggerDigi.h"
#include "DataFormats/L1THGCal/interface/HGCFETriggerDigiFwd.h"

#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

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
    l1t::HGCalClusterBxCollection hgcalClusterBxColl;
    l1t::HGCalTriggerCellBxCollection   hgcalTriggerCellBxColl;

    edm::Wrapper<l1t::HGCalTowerBxCollection>   w_hgcalTowerBxColl;
    edm::Wrapper<l1t::HGCalClusterBxCollection> w_hgcalClusterBxColl;
    edm::Wrapper<l1t::HGCalTriggerCellBxCollection>   w_hgcalTriggerCellBxColl;
  }
}
