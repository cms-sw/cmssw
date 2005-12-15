#ifndef DETIDECAL_ECALDETIDCOLLECTIONS_H
#define DETIDECAL_ECALDETIDCOLLECTIONS_H

#include "FWCore/EDProduct/interface/EDCollection.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

typedef edm::EDCollection<EBDetId> EBDetIdCollection;
typedef edm::EDCollection<EcalTrigTowerDetId> EcalTrigTowerDetIdCollection;

#endif
