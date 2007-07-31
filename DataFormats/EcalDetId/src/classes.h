#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include <boost/cstdint.hpp> 
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    edm::EDCollection<EBDetId> vEBDI_;
    edm::EDCollection<EcalTrigTowerDetId> vETTDI_;
    edm::EDCollection<EcalElectronicsId> vEELI_;
    edm::EDCollection<EcalTriggerElectronicsId> vETELI_;

    EBDetIdCollection theEBDI_;
    EcalTrigTowerDetIdCollection theETTDI_;
    EcalScDetIdCollection theESCDI_;
    EcalElectronicsIdCollection theEELI_;
    EcalTriggerElectronicsIdCollection theETELI_;

    edm::Wrapper<EBDetIdCollection> anotherEBDIw_;
    edm::Wrapper<EcalTrigTowerDetIdCollection> anothertheETTDIw_;
    edm::Wrapper<EcalScDetIdCollection> anothertheESCDIw_;
    edm::Wrapper<EcalElectronicsIdCollection> anothertheEELIw_;
    edm::Wrapper<EcalTriggerElectronicsIdCollection> anothertheETELIw_;

    edm::Wrapper< edm::EDCollection<EBDetId> > theEBDIw_;
    edm::Wrapper< edm::EDCollection<EcalTrigTowerDetId> > theETTDIw_;
    edm::Wrapper< edm::EDCollection<EcalScDetIdCollection> > theESCDIw_;
    edm::Wrapper< edm::EDCollection<EcalElectronicsId> > theEELIw_;
    edm::Wrapper< edm::EDCollection<EcalTriggerElectronicsId> > theETELIw_;
 }
}

