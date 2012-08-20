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

  int zside(const DetId&);

  double eta(const DetId&);
  double phi(const EBDetId&);
  double phi(const EEDetId&);
  double phi(const EcalTrigTowerDetId&);

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

namespace ecaldqm {

  extern const EcalElectronicsMapping* electronicsMap;
  extern const EcalTrigTowerConstituentsMap* trigtowerMap;
  extern const CaloGeometry* geometry;
  extern const std::vector<unsigned> memDCC;
  extern const double etaBound;

  inline unsigned dccId(const DetId &_id){
    checkElectronicsMap();

    unsigned subdet(_id.subdetId());

    if(subdet == EcalBarrel) return electronicsMap->DCCid(EBDetId(_id));
    else if(subdet == EcalTriggerTower) return electronicsMap->DCCid(EcalTrigTowerDetId(_id));
    else if(subdet == EcalEndcap){
      if(isEcalScDetId(_id)) return electronicsMap->getDCCandSC(EcalScDetId(_id)).first;
      else return electronicsMap->getElectronicsId(EEDetId(_id)).dccId();
    }
    else if(subdet == EcalLaserPnDiode) return EcalPnDiodeDetId(_id).iDCCId();

    throw cms::Exception("InvalidDetId") << "EcalDQMCommonUtils::dccId(" << _id.rawId() << ")" << std::endl;

    return 0;
  }

  inline unsigned dccId(const EcalElectronicsId &_id){
    return _id.dccId();
  }

  inline unsigned memDCCId(unsigned _index){
    // for DCCs with no MEM - map the index in an array of DCCs with no MEM to the DCC ID
    if(_index >= memDCC.size()) return 0;
    return memDCC.at(_index) + 1;
  }

  inline unsigned memDCCIndex(unsigned _dccid){
    std::vector<unsigned>::const_iterator itr(std::find(memDCC.begin(), memDCC.end(), _dccid - 1));
    if(itr == memDCC.end()) return -1;

    return (itr - memDCC.begin());
  }

  inline unsigned tccId(const DetId &_id){
    checkElectronicsMap();

    unsigned subdet(_id.subdetId());

    if(subdet == EcalBarrel) return electronicsMap->TCCid(EBDetId(_id));
    else if(subdet == EcalTriggerTower) return electronicsMap->TCCid(EcalTrigTowerDetId(_id));
    else if(subdet == EcalEndcap){
      if(isEcalScDetId(_id)) return 0; // incompatible
      else return electronicsMap->getTriggerElectronicsId(EEDetId(_id)).tccId();
    }

    throw cms::Exception("InvalidDetId") << "EcalDQMCommonUtils::tccId(" << uint32_t(_id) << ")" << std::endl;

    return 0;
  }

  inline unsigned tccId(const EcalElectronicsId &_id){
    checkElectronicsMap();

    return electronicsMap->getTriggerElectronicsId(_id).tccId();
  }


  inline unsigned towerId(const DetId &_id){
    checkElectronicsMap();

    unsigned subdet(_id.subdetId());

    if(subdet == EcalBarrel){
      return EBDetId(_id).tower().iTT();
    }else if(subdet == EcalTriggerTower){
      return EcalTrigTowerDetId(_id).iTT();
    }else if(subdet == EcalEndcap){
      if(isEcalScDetId(_id)) return electronicsMap->getDCCandSC(EcalScDetId(_id)).second;
      else return electronicsMap->getElectronicsId(EEDetId(_id)).towerId();
    }

    throw cms::Exception("InvalidDetId") << "EcalDQMCommonUtils::towerId(" << std::hex << uint32_t(_id) << ")" << std::endl;

    return 0;
  }

  inline unsigned towerId(const EcalElectronicsId &_id){
    return _id.towerId();
  }

  inline unsigned ttId(const DetId& _id){
    checkElectronicsMap();

    unsigned subdet(_id.subdetId());

    if(subdet == EcalBarrel)
      return EBDetId(_id).tower().iTT();
    else if(subdet == EcalTriggerTower)
      return electronicsMap->iTT(EcalTrigTowerDetId(_id));
    else if(subdet == EcalEndcap && !isEcalScDetId(_id))
      return electronicsMap->getTriggerElectronicsId(_id).ttId();

    throw cms::Exception("InvalidDetId") << "EcalDQMCommonUtils::ttId(" << std::hex << uint32_t(_id) << ")" << std::endl;

    return 0;
  }

  inline unsigned ttId(const EcalElectronicsId &_id){
    checkElectronicsMap();
    return electronicsMap->getTriggerElectronicsId(_id).ttId();
  }

  inline int zside(const DetId& _id){
    uint32_t rawId(_id);

    switch(_id.subdetId()){
    case EcalBarrel:
      return (((rawId >> 16) & 0x1) == 1 ? 1 : -1);
    case EcalEndcap:
      return (((rawId >> 14) & 0x1) == 1 ? 1 : -1);
    case EcalTriggerTower:
      return (((rawId >> 15) & 0x1) == 1 ? 1 : -1);
    case EcalLaserPnDiode:
      return (((rawId >> 4) & 0x7f) > kEBpLow ? 1 : -1);
    default:
      throw cms::Exception("InvalidDetId") << "EcalDQMCommonUtils::zside(" << std::hex << uint32_t(_id) << ")" << std::endl;
    }

    return 0;
  }

  inline double eta(const DetId& _id){
    checkGeometry();
    return geometry->getPosition(_id).eta();
  }

  inline double phi(EBDetId const& _ebid){
    const double degToRad(0.0174533);
    return (_ebid.iphi() - 10.5) * degToRad;
  }

  inline double phi(EEDetId const& _eeid){
    const double degToRad(0.0174533);
    double p(std::atan2(_eeid.ix() - 50.5, _eeid.iy() - 50.5));
    if(p < -10. * degToRad) p += 360. * degToRad;
    return p;
  }

  inline double phi(EcalTrigTowerDetId const& _ttid){
    const double degToRad(0.0174533);
    double p((_ttid.iphi() - 0.5) * 5. * degToRad);
    if(p > 350. * degToRad) p -= 360. * degToRad;
    return p;
  }

  inline bool isCrystalId(const DetId& _id){
    return (_id.det() == DetId::Ecal) && ((_id.subdetId() == EcalBarrel) || ((_id.subdetId() == EcalEndcap) && (((_id.rawId() >> 15) &  0x1) == 0)));
  }

  inline bool isSingleChannelId(const DetId& _id){
    return (_id.det() == DetId::Ecal) && (isCrystalId(_id) || (_id.subdetId() == EcalLaserPnDiode));
  }

  inline bool isEcalScDetId(const DetId& _id){
    return (_id.det() == DetId::Ecal) && ((_id.subdetId() == EcalEndcap) && ((_id.rawId() >> 15) &  0x1));
  }

  inline bool isEndcapTTId(const DetId& _id){
    return (_id.det() == DetId::Ecal) && ((_id.subdetId() == EcalTriggerTower) && (((_id.rawId() >> 7) & 0x7f) > 17));
  }

  inline std::string smName(unsigned _dccId){
    std::stringstream ss;

    unsigned iSM(_dccId - 1);

    if(iSM <= kEEmHigh)
      ss << "EE-" << std::setw(2) << std::setfill('0') << (((iSM - kEEmLow + 6) % 9) + 1);
    else if(iSM <= kEBmHigh)
      ss << "EB-" << std::setw(2) << std::setfill('0') << (iSM - kEBmLow + 1);
    else if(iSM <= kEBpHigh)
      ss << "EB+" << std::setw(2) << std::setfill('0') << (iSM - kEBpLow + 1);
    else if(iSM <= kEEpHigh)
      ss << "EE+" << std::setw(2) << std::setfill('0') << (((iSM - kEEpLow + 6) % 9) + 1);

    return ss.str();
  }

  // numbers from CalibCalorimetry/EcalLaserAnalyzer/src/MEEEGeom.cc
  inline unsigned EEPnDCC(unsigned _dee, unsigned _ab){
    switch(_dee){
    case 1: // EE+F -> FEDs 649-653/0
      if(_ab == 0) return 650;
      else return 651;
    case 2: // EE+N -> FEDs 604-608/0
      if(_ab == 0) return 605;
      else return 606;
    case 3: // EE-N -> FEDs 601-603, 608/1, 609
      if(_ab == 0) return 601;
      else return 602;
    case 4: // EE-F -> FEDs 646-648, 653/1, 654
      if(_ab == 0) return 647;
      else return 646;
    default:
      return 600;
    }
  }

  inline unsigned nCrystals(unsigned _dccId){

    unsigned iSM(_dccId - 1);

    if(iSM >= kEBmLow && iSM <= kEBpHigh) return 1700;

    switch(iSM){
    case kEEm05:
    case kEEp05:
      return 810;
    case kEEm07:
    case kEEm03:
    case kEEp07:
    case kEEp03:
      return 830;
    case kEEm09:
    case kEEm01:
    case kEEp09:
    case kEEp01:
      return 815;
    case kEEm08:
    case kEEm02:
    case kEEp08:
    case kEEp02:
      return 791;
    case kEEm04:
    case kEEm06:
    case kEEp04:
    case kEEp06:
      return 821;
    default:
      return 0;
    }

  }

  inline unsigned nSuperCrystals(unsigned _dccId){

    unsigned iSM(_dccId - 1);

    if(iSM >= kEBmLow && iSM <= kEBpHigh) return 68;

    switch(iSM){
    case kEEm05:
    case kEEp05:
      return 41;
    case kEEm07:
    case kEEm03:
    case kEEp07:
    case kEEp03:
      return 34;
    case kEEm09:
    case kEEm01:
    case kEEm04:
    case kEEm06:
    case kEEp09:
    case kEEp01:
    case kEEp04:
    case kEEp06:
      return 33;
    case kEEm08:
    case kEEm02:
    case kEEp08:
    case kEEp02:
      return 32;
    default:
      return 0;
    }

  }

  inline bool ccuExists(unsigned _dccId, unsigned _towerId){
    if(_towerId == 69 || _towerId == 70) return true;
    else if((_dccId == 8 || _dccId == 53) && _towerId >= 18 && _towerId <= 24) return false;
    else if(_dccId <= kEEmHigh + 1 || _dccId >= kEEpLow + 1){
      if(_towerId > nSuperCrystals(_dccId)) return false;
    }
    
    return true;
  }

  inline const EcalElectronicsMapping* getElectronicsMap(){
    checkElectronicsMap();
    return electronicsMap;
  }

  inline void setElectronicsMap(const EcalElectronicsMapping* _map){
    if(electronicsMap) return;
    electronicsMap = _map;
  }

  inline const EcalTrigTowerConstituentsMap* getTrigTowerMap(){
    checkTrigTowerMap();
    return trigtowerMap;
  }

  inline void setTrigTowerMap(const EcalTrigTowerConstituentsMap* _map){
    if(trigtowerMap) return;
    trigtowerMap = _map;
  }

  inline const CaloGeometry* getGeometry(){
    checkGeometry();
    return geometry;
  }

  inline void setGeometry(const CaloGeometry* _geom){
    if(geometry) return;
    geometry = _geom;
  }

  inline void checkElectronicsMap(){
    if(!electronicsMap) throw cms::Exception("InvalidCall") << "ElectronicsMapping not initialized" << std::endl;
  }

  inline void checkTrigTowerMap(){
    if(!trigtowerMap) throw cms::Exception("InvalidCall") << "TrigTowerConstituentsMap not initialized" << std::endl;
  }

  inline void checkGeometry(){
    if(!geometry) throw cms::Exception("InvalidCall") << "CaloGeometry not initialized" << std::endl;
  }
}

#endif
