#ifndef RECOCALOTOOLS_NAVIGATION_ECALPRESHOWERNAVIGATOR_H
#define RECOCALOTOOLS_NAVIGATION_ECALPRESHOWERNAVIGATOR_H 1

#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

class EcalPreshowerNavigator : public CaloNavigator<ESDetId> {
 public:
  EcalPreshowerNavigator(const ESDetId& home,const CaloSubdetectorTopology* esTopology) :
    CaloNavigator<ESDetId>(home,esTopology)
    {
    };
};

#endif






