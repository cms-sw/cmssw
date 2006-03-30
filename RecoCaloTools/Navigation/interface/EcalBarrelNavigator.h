#ifndef RECOCALOTOOLS_NAVIGATION_ECALBARRELNAVIGATOR_H
#define RECOCALOTOOLS_NAVIGATION_ECALBARRELNAVIGATOR_H 1

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

class EcalBarrelNavigator : public CaloNavigator<EBDetId> 
{
 public:
  EcalBarrelNavigator(const EBDetId& home,const CaloSubdetectorTopology* ebTopology) :
    CaloNavigator<EBDetId>(home,ebTopology)
    {
    };
};

#endif
