#ifndef LogicIDTranslation_H
#define LogicIDTranslation_H

#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

namespace ecaldqm {
  EcalLogicID ecalID();
  EcalLogicID subdetID(EcalSubdetector);
  EcalLogicID crystalID(DetId const &, EcalElectronicsMapping const *);
  EcalLogicID towerID(EcalElectronicsId const &);
  EcalLogicID memChannelID(EcalPnDiodeDetId const &);
  EcalLogicID memTowerID(EcalElectronicsId const &);
  EcalLogicID lmPNID(EcalPnDiodeDetId const &);

  DetId toDetId(EcalLogicID const &);
}  // namespace ecaldqm

#endif
