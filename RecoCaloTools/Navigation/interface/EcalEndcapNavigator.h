#ifndef RECOCALOTOOLS_NAVIGATION_ECALENDCAPNAVIGATOR_H
#define RECOCALOTOOLS_NAVIGATION_ECALENDCAPNAVIGATOR_H 1

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

class EcalEndcapNavigator : public CaloNavigator<EEDetId> {
 public:
  EcalEndcapNavigator(const EEDetId& home,const CaloSubdetectorTopology* ebTopology) :
    CaloNavigator<EEDetId>(home,ebTopology)
    {
    };
};

#endif
