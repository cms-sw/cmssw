#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include <mutex>

namespace ecaldqm
{
  unsigned memarr[] = {kEEm07, kEEm08, kEEm02, kEEm03,
                       kEBm01, kEBm02, kEBm03, kEBm04, kEBm05, kEBm06, kEBm07, kEBm08, kEBm09,
                       kEBm10, kEBm11, kEBm12, kEBm13, kEBm14, kEBm15, kEBm16, kEBm17, kEBm18,
                       kEBp01, kEBp02, kEBp03, kEBp04, kEBp05, kEBp06, kEBp07, kEBp08, kEBp09,
                       kEBp10, kEBp11, kEBp12, kEBp13, kEBp14, kEBp15, kEBp16, kEBp17, kEBp18,
                       kEEp07, kEEp08, kEEp02, kEEp03};
  const std::vector<unsigned> memDCC(memarr, memarr + 44);

  const double etaBound(1.479);

  unsigned
  dccId(const DetId &_id)
  {
    EcalElectronicsMapping const* map(getElectronicsMap());

    unsigned subdet(_id.subdetId());

    if(subdet == EcalBarrel) return map->DCCid(EBDetId(_id));
    else if(subdet == EcalTriggerTower) return map->DCCid(EcalTrigTowerDetId(_id));
    else if(subdet == EcalEndcap){
      if(isEcalScDetId(_id)) return map->getDCCandSC(EcalScDetId(_id)).first;
      else return map->getElectronicsId(EEDetId(_id)).dccId();
    }
    else if(subdet == EcalLaserPnDiode) return EcalPnDiodeDetId(_id).iDCCId();

    throw cms::Exception("InvalidDetId") << "EcalDQMCommonUtils::dccId(" << _id.rawId() << ")" << std::endl;

    return 0;
  }

  unsigned
  dccId(const EcalElectronicsId &_id)
  {
    return _id.dccId();
  }

  unsigned
  memDCCId(unsigned _index)
  {
    // for DCCs with no MEM - map the index in an array of DCCs with no MEM to the DCC ID
    if(_index >= memDCC.size()) return 0;
    return memDCC.at(_index) + 1;
  }

  unsigned
  memDCCIndex(unsigned _dccid)
  {
    std::vector<unsigned>::const_iterator itr(std::find(memDCC.begin(), memDCC.end(), _dccid - 1));
    if(itr == memDCC.end()) return -1;

    return (itr - memDCC.begin());
  }

  unsigned
  tccId(const DetId &_id)
  {
    EcalElectronicsMapping const* map(getElectronicsMap());

    unsigned subdet(_id.subdetId());

    if(subdet == EcalBarrel) return map->TCCid(EBDetId(_id));
    else if(subdet == EcalTriggerTower) return map->TCCid(EcalTrigTowerDetId(_id));
    else if(subdet == EcalEndcap){
      if(isEcalScDetId(_id)) return 0; // incompatible
      else return map->getTriggerElectronicsId(EEDetId(_id)).tccId();
    }

    throw cms::Exception("InvalidDetId") << "EcalDQMCommonUtils::tccId(" << uint32_t(_id) << ")" << std::endl;

    return 0;
  }

  unsigned
  tccId(const EcalElectronicsId &_id)
  {
    return getElectronicsMap()->getTriggerElectronicsId(_id).tccId();
  }


  unsigned
  towerId(const DetId &_id)
  {
    unsigned subdet(_id.subdetId());

    if(subdet == EcalBarrel) return EBDetId(_id).tower().iTT();
    else if(subdet == EcalTriggerTower) return EcalTrigTowerDetId(_id).iTT();
    else if(subdet == EcalEndcap){
      if(isEcalScDetId(_id)) return getElectronicsMap()->getDCCandSC(EcalScDetId(_id)).second;
      else return getElectronicsMap()->getElectronicsId(EEDetId(_id)).towerId();
    }

    throw cms::Exception("InvalidDetId") << "EcalDQMCommonUtils::towerId(" << std::hex << uint32_t(_id) << ")" << std::endl;

    return 0;
  }

  unsigned
  towerId(const EcalElectronicsId &_id)
  {
    return _id.towerId();
  }

  unsigned
  ttId(const DetId& _id)
  {
    unsigned subdet(_id.subdetId());

    if(subdet == EcalBarrel)
      return EBDetId(_id).tower().iTT();
    else if(subdet == EcalTriggerTower)
      return getElectronicsMap()->iTT(EcalTrigTowerDetId(_id));
    else if(subdet == EcalEndcap && !isEcalScDetId(_id))
      return getElectronicsMap()->getTriggerElectronicsId(_id).ttId();

    throw cms::Exception("InvalidDetId") << "EcalDQMCommonUtils::ttId(" << std::hex << uint32_t(_id) << ")" << std::endl;

    return 0;
  }

  unsigned
  ttId(const EcalElectronicsId &_id)
  {
    return getElectronicsMap()->getTriggerElectronicsId(_id).ttId();
  }

  unsigned
  rtHalf(DetId const& _id)
  {
    if(_id.subdetId() == EcalBarrel){
      int ic(EBDetId(_id).ic());
      if((ic - 1) / 20 > 4 && (ic - 1) % 20 < 10) return 1;
    }
    else{
      unsigned iDCC(dccId(_id) - 1);
      if((iDCC == kEEm05 || iDCC == kEEp05) && EEDetId(_id).ix() > 50) return 1;
    }

    return 0;
  }

  std::pair<unsigned, unsigned>
  innerTCCs(unsigned _dccId)
  {
    int iDCC(_dccId - 1);
    std::pair<unsigned, unsigned> res;
    if(iDCC <= kEEmHigh){
      res.first = (iDCC - kEEmLow) * 2;
      if(res.first == 0) res.first = 18;
      res.second = (iDCC - kEEmLow) * 2 + 1;
    }
    else if(iDCC <- kEBpHigh)
      res.first = res.second = _dccId + 27;
    else{
      res.first = (iDCC - kEEpLow) * 2 + 90;
      if(res.first == 90) res.first = 108;
      res.second = (iDCC - kEEpLow) * 2 + 91;
    }

    return res;
  }

  std::pair<unsigned, unsigned>
  outerTCCs(unsigned _dccId)
  {
    int iDCC(_dccId - 1);
    std::pair<unsigned, unsigned> res;
    if(iDCC <= kEEmHigh){
      res.first = (iDCC - kEEmLow) * 2 + 18;
      if(res.first == 18) res.first = 36;
      res.second = (iDCC - kEEmLow) * 2 + 19;
    }
    else if(iDCC <= kEBpHigh)
      res.first = res.second = _dccId + 27;
    else{
      res.first = (iDCC - kEEpLow) * 2 + 72;
      if(res.first == 72) res.first = 90;
      res.second = (iDCC - kEEpLow) * 2 + 73;
    }

    return res;
  }

  std::vector<DetId>
  scConstituents(EcalScDetId const& _scid)
  {
    std::vector<DetId> res;

    int ixbase((_scid.ix() - 1) * 5);
    int iybase((_scid.iy() - 1) * 5);

    for(int ix(1); ix <= 5; ++ix){
      for(int iy(1); iy <= 5; ++iy){
        if(EEDetId::validDetId(ixbase + ix, iybase + iy, _scid.zside()))
          res.push_back(EEDetId(ixbase + ix, iybase + iy, _scid.zside()));
      }
    }

    return res;
  }

  int
  zside(const DetId& _id)
  {
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

  double
  eta(const EBDetId& _ebid)
  {
    return _ebid.approxEta() + (_ebid.zside() < 0 ? 0.5 : -0.5) * EBDetId::crystalUnitToEta;
  }

  double
  eta(const EEDetId& _id)
  {
    return getGeometry()->getPosition(_id).eta();
  }

  double
  phi(EBDetId const& _ebid)
  {
    const double degToRad(0.0174533);
    return (_ebid.iphi() - 10.5) * degToRad;
  }

  double
  phi(EEDetId const& _eeid)
  {
    const double degToRad(0.0174533);
    double p(std::atan2(_eeid.ix() - 50.5, _eeid.iy() - 50.5));
    if(p < -10. * degToRad) p += 360. * degToRad;
    return p;
  }

  double
  phi(EcalTrigTowerDetId const& _ttid)
  {
    const double degToRad(0.0174533);
    double p((_ttid.iphi() - 0.5) * 5. * degToRad);
    if(p > 350. * degToRad) p -= 360. * degToRad;
    return p;
  }

  double
  phi(double _phi)
  {
    const double degToRad(0.0174533);
    if(_phi < -10. * degToRad) _phi += 360. * degToRad;
    return _phi;
  }

  bool
  isForward(DetId const& _id)
  {
    // the numbers here roughly corresponds to a cut at |eta| > 2.2
    // cf hepwww.rl.ac.uk/CMSecal/Dee-layout.html
    if(_id.subdetId() != EcalEndcap) return false;
    if(isEcalScDetId(_id)){
      EcalScDetId scid(_id);
      return (scid.ix() - 10) * (scid.ix() - 10) + (scid.iy() - 10) * (scid.iy() - 10) < 25.;
    }
    else{
      EEDetId eeid(_id);
      return (eeid.ix() - 50) * (eeid.ix() - 50) + (eeid.iy() - 50) * (eeid.iy() - 50) < 625.;
    }
  }

  bool
  isCrystalId(const DetId& _id)
  {
    return (_id.det() == DetId::Ecal) && ((_id.subdetId() == EcalBarrel) || ((_id.subdetId() == EcalEndcap) && (((_id.rawId() >> 15) &  0x1) == 0)));
  }

  bool
  isSingleChannelId(const DetId& _id)
  {
    return (_id.det() == DetId::Ecal) && (isCrystalId(_id) || (_id.subdetId() == EcalLaserPnDiode));
  }

  bool
  isEcalScDetId(const DetId& _id)
  {
    return (_id.det() == DetId::Ecal) && ((_id.subdetId() == EcalEndcap) && ((_id.rawId() >> 15) &  0x1));
  }

  bool
  isEndcapTTId(const DetId& _id)
  {
    return (_id.det() == DetId::Ecal) && ((_id.subdetId() == EcalTriggerTower) && (((_id.rawId() >> 7) & 0x7f) > 17));
  }

  unsigned
  dccId(std::string const& _smName)
  {
    unsigned smNumber(std::atoi(_smName.substr(3).c_str()));

    if(_smName.find("EE-") == 0)
      return kEEmLow + 1 + ((smNumber + 2) % 9);
    else if(_smName.find("EB-") == 0)
      return kEBmLow + 1 + smNumber;
    else if(_smName.find("EB+") == 0)
      return kEBpLow + 1 + smNumber;
    else if(_smName.find("EE+") == 0)
      return kEEpLow + 1 + ((smNumber + 2) % 9);
    else
      return 0;
  }

  std::string
  smName(unsigned _dccId)
  {
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
  unsigned
  EEPnDCC(unsigned _dee, unsigned _ab)
  {
    switch(_dee){
    case 1: // EE+F -> FEDs 649-653/0
      if(_ab == 0) return 50;
      else return 51;
    case 2: // EE+N -> FEDs 646-648, 653/1, 654
      if(_ab == 0) return 47;
      else return 46;
    case 3: // EE-N -> FEDs 601-603, 608/1, 609
      if(_ab == 0) return 1;
      else return 2;
    case 4: // EE-F -> FEDs 604-608/0
      if(_ab == 0) return 5;
      else return 6;
    default:
      return 0;
    }
  }

  unsigned
  nCrystals(unsigned _dccId)
  {
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

  unsigned
  nSuperCrystals(unsigned _dccId)
  {
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

  bool
  ccuExists(unsigned _dccId, unsigned _towerId)
  {
    if(_towerId == 69 || _towerId == 70) return true;
    else if((_dccId == 8 || _dccId == 53) && _towerId >= 18 && _towerId <= 24) return false;
    else if(_dccId <= kEEmHigh + 1 || _dccId >= kEEpLow + 1) return _towerId <= nSuperCrystals(_dccId);
    else return _towerId <= 68;
  }


  /*
    Note on concurrency compatibility:           
    Call to getters bellow must always happen after
    EcalDQMonitor::ecaldqmGetSetupObjects or equivalent.
    As long as check-set-get are executed in this order
    within each thread, the following functions are
    thread-safe.
  */

  std::mutex mapMutex;
  EcalElectronicsMapping const* electronicsMap(0);
  EcalTrigTowerConstituentsMap const* trigtowerMap(0);
  CaloGeometry const* geometry(0);
  CaloTopology const* topology(0);

  bool
  checkElectronicsMap(bool _throw/* = true*/)
  {
    std::lock_guard<std::mutex> lock(mapMutex);
    if(electronicsMap) return true;
    if(_throw) throw cms::Exception("InvalidCall") << "ElectronicsMapping not initialized";
    return false;
  }

  EcalElectronicsMapping const*
  getElectronicsMap()
  {
    if(!electronicsMap) checkElectronicsMap();
    return electronicsMap;
  }

  void
  setElectronicsMap(EcalElectronicsMapping const* _map)
  {
    std::lock_guard<std::mutex> lock(mapMutex);
    if(electronicsMap) return;
    electronicsMap = _map;
  }

  bool
  checkTrigTowerMap(bool _throw/* = true*/)
  {
    std::lock_guard<std::mutex> lock(mapMutex);
    if(trigtowerMap) return true;
    if(_throw) throw cms::Exception("InvalidCall") << "TrigTowerConstituentsMap not initialized";
    return false;
  }

  EcalTrigTowerConstituentsMap const*
  getTrigTowerMap()
  {
    if(!trigtowerMap) checkTrigTowerMap();
    return trigtowerMap;
  }

  void
  setTrigTowerMap(EcalTrigTowerConstituentsMap const* _map)
  {
    std::lock_guard<std::mutex> lock(mapMutex);
    if(trigtowerMap) return;
    trigtowerMap = _map;
  }

  bool
  checkGeometry(bool _throw/* = true*/)
  {
    std::lock_guard<std::mutex> lock(mapMutex);
    if(geometry) return true;
    if(_throw) throw cms::Exception("InvalidCall") << "CaloGeometry not initialized";
    return false;
  }

  CaloGeometry const*
  getGeometry()
  {
    if(!geometry) checkGeometry();
    return geometry;
  }

  void
  setGeometry(CaloGeometry const* _geom)
  {
    std::lock_guard<std::mutex> lock(mapMutex);
    if(geometry) return;
    geometry = _geom;
  }

  bool
  checkTopology(bool _throw/* = true*/)
  {
    std::lock_guard<std::mutex> lock(mapMutex);
    if(topology) return true;
    if(_throw) throw cms::Exception("InvalidCall") << "CaloTopology not initialized";
    return false;
  }

  CaloTopology const*
  getTopology()
  {
    if(!topology) checkTopology();
    return topology;
  }

  void
  setTopology(CaloTopology const* _geom)
  {
    std::lock_guard<std::mutex> lock(mapMutex);
    if(topology) return;
    topology = _geom;
  }
}
