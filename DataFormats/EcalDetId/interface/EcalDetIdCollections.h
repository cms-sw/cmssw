#ifndef DETIDECAL_ECALDETIDCOLLECTIONS_H
#define DETIDECAL_ECALDETIDCOLLECTIONS_H

#include "DataFormats/Common/interface/EDCollection.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

typedef edm::EDCollection<EBDetId> EBDetIdCollection;
typedef edm::EDCollection<EEDetId> EEDetIdCollection;
typedef edm::EDCollection<EcalElectronicsId> EcalElectronicsIdCollection;
typedef edm::EDCollection<EcalTriggerElectronicsId> EcalTriggerElectronicsIdCollection;
typedef edm::EDCollection<EcalTrigTowerDetId> EcalTrigTowerDetIdCollection;
typedef edm::EDCollection<EcalScDetId> EcalScDetIdCollection;

#endif
