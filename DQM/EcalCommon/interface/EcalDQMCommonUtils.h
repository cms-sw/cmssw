#ifndef EcalDQMCommonUtils_H
#define EcalDQMCommonUtils_H

#include <iomanip>
#include <algorithm>
#include <cmath>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

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

  enum Constants {
    nDCC = 54,
    nEBDCC = 36,
    nEEDCC = 18,
    nDCCMEM = 44,
    nEEDCCMEM = 8,

    nTTOuter = 16,
    nTTInner = 28,
    // These lines set the number of TriggerTowers in "outer" and "inner" TCCs,
    // where "outer" := closer to the barrel. These constants are used in
    // setting the binning. There are 16 trigger towers per TCC for "outer" TCCs,
    // and 24 per TCC for "inner" TCCs (but the numbering is from 0 to 27, so
    // 28 bins are required).

    nTCC = 108,
    kEEmTCCLow = 0, kEEmTCCHigh = 35,
    kEEpTCCLow = 72, kEEpTCCHigh = 107,
    kEBTCCLow = 36, kEBTCCHigh = 71,

    nChannels = EBDetId::kSizeForDenseIndexing + EEDetId::kSizeForDenseIndexing,
    nTowers = EcalTrigTowerDetId::kEBTotalTowers + EcalScDetId::kSizeForDenseIndexing
  };

  extern std::vector<unsigned> const memDCC;

  extern double const etaBound;

  // returns DCC ID (1 - 54)
  unsigned dccId(DetId const&);
  unsigned dccId(EcalElectronicsId const&);

  unsigned memDCCId(unsigned); // convert from dccId skipping DCCs without MEM
  unsigned memDCCIndex(unsigned); // reverse conversion

  // returns TCC ID (1 - 108)
  unsigned tccId(DetId const&);
  unsigned tccId(EcalElectronicsId const&);

  // returns the data tower id - pass only 
  unsigned towerId(DetId const&);
  unsigned towerId(EcalElectronicsId const&);

  unsigned ttId(DetId const&);
  unsigned ttId(EcalElectronicsId const&);

  unsigned rtHalf(DetId const&);

  std::pair<unsigned, unsigned> innerTCCs(unsigned);
  std::pair<unsigned, unsigned> outerTCCs(unsigned);

  std::vector<DetId> scConstituents(EcalScDetId const&);

  EcalPnDiodeDetId pnForCrystal(DetId const&, char);

  unsigned dccId(std::string const&);
  std::string smName(unsigned);

  int zside(DetId const&);

  double eta(EBDetId const&);
  double eta(EEDetId const&);
  double phi(EBDetId const&);
  double phi(EEDetId const&);
  double phi(EcalTrigTowerDetId const&);
  double phi(double);

  bool isForward(DetId const&);

  bool isCrystalId(DetId const&);
  bool isSingleChannelId(DetId const&);
  bool isEcalScDetId(DetId const&);
  bool isEndcapTTId(DetId const&);

  unsigned nCrystals(unsigned);
  unsigned nSuperCrystals(unsigned);

  bool ccuExists(unsigned, unsigned);

  bool checkElectronicsMap(bool = true);
  EcalElectronicsMapping const* getElectronicsMap();
  void setElectronicsMap(EcalElectronicsMapping const*);

  bool checkTrigTowerMap(bool = true);
  EcalTrigTowerConstituentsMap const* getTrigTowerMap();
  void setTrigTowerMap(EcalTrigTowerConstituentsMap const*);

  bool checkGeometry(bool = true);
  CaloGeometry const* getGeometry();
  void setGeometry(CaloGeometry const*);

  bool checkTopology(bool = true);
  CaloTopology const* getTopology();
  void setTopology(CaloTopology const*);
}

#endif
