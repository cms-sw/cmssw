#ifndef RECOCALOTOOLS_NAVIGATION_CALOTOWERNAVIGATOR_H
#define RECOCALOTOOLS_NAVIGATION_CALOTOWERNAVIGATOR_H 1

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

class CaloTowerNavigator : public CaloNavigator<CaloTowerDetId> 
{
 public:
  CaloTowerNavigator(const CaloTowerDetId& home,const CaloSubdetectorTopology* topo) :
    CaloNavigator<CaloTowerDetId>(home,topo)
    {
    };
};

#endif
