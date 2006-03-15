#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    edm::EDCollection<EBDetId> vEBDI_;
    edm::EDCollection<EcalTrigTowerDetId> vETTDI_;
    edm::EDCollection<EcalElectronicsId> vEELI_;

    EBDetIdCollection theEBDI_;
    EcalTrigTowerDetIdCollection theETTDI_;
    EcalElectronicsIdCollection theEELI_;

    edm::Wrapper<EBDetIdCollection> anotherEBDIw_;
    edm::Wrapper<EcalTrigTowerDetIdCollection> anothertheETTDIw_;
    edm::Wrapper<EcalElectronicsIdCollection> anothertheEELIw_;

    edm::Wrapper< edm::EDCollection<EBDetId> > theEBDIw_;
    edm::Wrapper< edm::EDCollection<EcalTrigTowerDetId> > theETTDIw_;
    edm::Wrapper< edm::EDCollection<EcalElectronicsId> > theEELIw_;
 }
}

