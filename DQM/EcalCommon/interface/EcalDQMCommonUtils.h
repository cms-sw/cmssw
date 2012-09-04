#ifndef EcalDQMCommonUtils_H
#define EcalDQMCommonUtils_H

#include <iomanip>
#include <algorithm>
#include <cmath>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  enum SMName {
    kEEm07, kEEm08, kEEm09, kEEm01, kEEm02, kEEm03, kEEm04, kEEm05, kEEm06,
    kEBm01, kEBm02, kEBm03, kEBm04, kEBm05, kEBm06, kEBm07, kEBm08, kEBm09,
    kEBm10, kEBm11, kEBm12, kEBm13, kEBm14, kEBm15, kEBm16, kEBm17, kEBm18,
    kEBp01, kEBp02, kEBp03, kEBp04, kEBp05, kEBp06, kEBp07, kEBp08, kEBp09,
    kEBp10, kEBp11, kEBp12, kEBp13, kEBp14, kEBp15, kEBp16, kEBp17, kEBp18,
    kEEp07, kEEp08, kEEp09, kEEp01, kEEp02, kEEp03, kEEp04, kEEp05, kEEp06,
    kEEmLow = kEEm07, kEEmHigh = kEEm06,
    kEEpLow = kEEp07, kEEpHigh = kEEp06,
    kEBmLow = kEBm01, kEBmHigh = kEBm18,
    kEBpLow = kEBp01, kEBpHigh = kEBp18
  };

  extern std::vector<unsigned> const memDCC;

  extern double const etaBound;

  // returns DCC ID (1 - 54)
  unsigned dccId(const DetId&);
  unsigned dccId(const EcalElectronicsId&);

  unsigned memDCCId(unsigned); // convert from dccId skipping DCCs without MEM
  unsigned memDCCIndex(unsigned); // reverse conversion

  // returns TCC ID (1 - 108)
  unsigned tccId(const DetId&);
  unsigned tccId(const EcalElectronicsId&);

  // returns the data tower id - pass only 
  unsigned towerId(const DetId&);
  unsigned towerId(const EcalElectronicsId&);

  unsigned ttId(const DetId&);
  unsigned ttId(const EcalElectronicsId&);

  unsigned rtHalf(DetId const&);

  std::pair<unsigned, unsigned> innerTCCs(unsigned);
  std::pair<unsigned, unsigned> outerTCCs(unsigned);

  std::vector<DetId> scConstituents(EcalScDetId const&);

  std::string smName(unsigned);

  int zside(const DetId&);

  double eta(const EBDetId&);
  double eta(const EEDetId&);
  double phi(const EBDetId&);
  double phi(const EEDetId&);
  double phi(const EcalTrigTowerDetId&);

  bool isForward(const DetId&);

  bool isCrystalId(const DetId&);
  bool isSingleChannelId(const DetId&);
  bool isEcalScDetId(const DetId&);
  bool isEndcapTTId(const DetId&);

  unsigned EEPnDCC(unsigned, unsigned);

  unsigned nCrystals(unsigned);
  unsigned nSuperCrystals(unsigned);

  bool ccuExists(unsigned, unsigned);

  const EcalElectronicsMapping* getElectronicsMap();
  void setElectronicsMap(const EcalElectronicsMapping*);

  const EcalTrigTowerConstituentsMap* getTrigTowerMap();
  void setTrigTowerMap(const EcalTrigTowerConstituentsMap*);

  const CaloGeometry* getGeometry();
  void setGeometry(const CaloGeometry*);

  void checkElectronicsMap();
  void checkTrigTowerMap();
  void checkGeometry();
}

#endif
