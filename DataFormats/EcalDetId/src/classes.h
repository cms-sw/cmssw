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
  struct dictionary {
    edm::EDCollection<EBDetId> vEBDI_;
    edm::EDCollection<EEDetId> vEEDI_;
    edm::EDCollection<EcalTrigTowerDetId> vETTDI_;
    edm::EDCollection<EcalElectronicsId> vEELI_;
    edm::EDCollection<EcalTriggerElectronicsId> vETELI_;

    EBDetIdCollection theEBDI_;
    EEDetIdCollection theEEDI_;
    EcalTrigTowerDetIdCollection theETTDI_;
    EcalScDetIdCollection theESCDI_;
    EcalElectronicsIdCollection theEELI_;
    EcalTriggerElectronicsIdCollection theETELI_;

    edm::Wrapper<EBDetIdCollection> anotherEBDIw_;
    edm::Wrapper<EEDetIdCollection> anotherEEDIw_;
    edm::Wrapper<EcalTrigTowerDetIdCollection> anothertheETTDIw_;
    edm::Wrapper<EcalScDetIdCollection> anothertheESCDIw_;
    edm::Wrapper<EcalElectronicsIdCollection> anothertheEELIw_;
    edm::Wrapper<EcalTriggerElectronicsIdCollection> anothertheETELIw_;

    edm::Wrapper< edm::EDCollection<EBDetId> > theEBDIw_;
    edm::Wrapper< edm::EDCollection<EEDetId> > theEEDIw_;
    edm::Wrapper< edm::EDCollection<EcalTrigTowerDetId> > theETTDIw_;
    edm::Wrapper< edm::EDCollection<EcalScDetIdCollection> > theESCDIw_;
    edm::Wrapper< edm::EDCollection<EcalElectronicsId> > theEELIw_;
    edm::Wrapper< edm::EDCollection<EcalTriggerElectronicsId> > theETELIw_;

    // needed for channel recovery
    std::set<EBDetId> _ebDetId;
    std::set<EEDetId> _eeDetId;
    std::set<EcalTrigTowerDetId> _TTId;
    std::set<EcalScDetId> _SCId;
    edm::Wrapper< std::set<EBDetId> > _ebDetIdw;
    edm::Wrapper< std::set<EEDetId> > _eeDetIdw;
    edm::Wrapper< std::set<EcalTrigTowerDetId> > _TTIdw;
    edm::Wrapper< std::set<EcalScDetId> > _SCIdw;
 };
}

