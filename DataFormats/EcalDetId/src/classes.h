#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "FWCore/EDProduct/interface/Wrapper.h"

namespace {
  namespace {
    edm::EDCollection<EBDetId> vEBDI_;
    edm::EDCollection<EcalTrigTowerDetId> vETTDI_;

    EBDetIdCollection theEBDI_;
    EcalTrigTowerDetIdCollection theETTDI_;

    edm::Wrapper<EBDetIdCollection> anotherEBDIw_;
    edm::Wrapper<EcalTrigTowerDetIdCollection> anothertheETTDIw_;


    edm::Wrapper< edm::EDCollection<EBDetId> > theEBDIw_;
    edm::Wrapper< edm::EDCollection<EcalTrigTowerDetId> > theETTDIw_;
 }
}

