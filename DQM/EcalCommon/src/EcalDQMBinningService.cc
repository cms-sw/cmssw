#include "DQM/EcalCommon/interface/EcalDQMBinningService.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"

#include <utility>
#include <iomanip>
#include <sstream>

#include "TMath.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TString.h"
#include "TPRegexp.h"

using namespace ecaldqm;

EcalDQMBinningService::EcalDQMBinningService(const edm::ParameterSet& _ps, edm::ActivityRegistry &_registry) :
  verbosity_(_ps.getUntrackedParameter<int>("verbosity"))
{
  _registry.watchPostBeginRun(this, &EcalDQMBinningService::postBeginRun);
}

EcalDQMBinningService::~EcalDQMBinningService()
{
}

void
EcalDQMBinningService::postBeginRun(const edm::Run &, const edm::EventSetup &_es)
{
  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalElectronicsMapping> elecMapHandle;
  _es.get<EcalMappingRcd>().get(elecMapHandle);
  setElectronicsMap(elecMapHandle.product());

  // set up ecaldqm::electronicsMap in EcalDQMCommonUtils
  edm::ESHandle<EcalTrigTowerConstituentsMap> ttMapHandle;
  _es.get<IdealGeometryRecord>().get(ttMapHandle);
  setTrigTowerMap(ttMapHandle.product());

  edm::ESHandle<CaloGeometry> geomHndl;
  _es.get<CaloGeometryRecord>().get(geomHndl);
  setGeometry(geomHndl.product());
}

void
EcalDQMBinningService::postEndRun(const edm::Run &, const edm::EventSetup &)
{
  setElectronicsMap(0);
  setTrigTowerMap(0);
  setGeometry(0);
}

EcalDQMBinningService::AxisSpecs
EcalDQMBinningService::getBinning(ObjectType _otype, BinningType _btype, bool _isMap, int _axis, unsigned _iME) const
{
  if(_otype >= nObjType || _btype >= unsigned(nPresetBinnings))
    return AxisSpecs(); // you are on your own

  switch(_otype){
  case kEB:
    return getBinningEB_(_btype, _isMap, _axis);
  case kEE:
    return getBinningEE_(_btype, _isMap, 0, _axis);
  case kEEm:
    return getBinningEE_(_btype, _isMap, -1, _axis);
  case kEEp:
    return getBinningEE_(_btype, _isMap, 1, _axis);
  case kSM:
    return getBinningSM_(_btype, _isMap, _iME, _axis);
  case kEBSM:
    return getBinningSM_(_btype, _isMap, _iME + 9, _axis);
  case kEESM:
    if(_iME <= kEEmHigh) return getBinningSM_(_btype, _isMap, _iME, _axis);
    else return getBinningSM_(_btype, _isMap, _iME + nEBDCC, _axis);
  case kSMMEM:
    return getBinningSMMEM_(_btype, _isMap, _iME, _axis);
  case kEBSMMEM:
    return getBinningSMMEM_(_btype, _isMap, _iME + nEEDCCMEM / 2, _axis);
  case kEESMMEM: 
    if(_iME <= kEEmHigh) return getBinningSMMEM_(_btype, _isMap, _iME, _axis);
    else return getBinningSMMEM_(_btype, _isMap, _iME + nEBDCC, _axis);
  case kEcal:
    return getBinningEcal_(_btype, _isMap, _axis);
  case kMEM:
    return getBinningMEM_(_btype, _isMap, -1, _axis);
  case kEBMEM:
    return getBinningMEM_(_btype, _isMap, EcalBarrel, _axis);
  case kEEMEM:
    return getBinningMEM_(_btype, _isMap, EcalEndcap, _axis);
  default:
    return AxisSpecs();
  }
}

int
EcalDQMBinningService::findBin1D(ObjectType _otype, BinningType _btype, const DetId& _id) const
{
  switch(_otype){
  case kSM:
  case kEBSM:
  case kEESM:
    if(_btype == kSuperCrystal)
      return towerId(_id);
    else if(_btype == kTriggerTower){
      unsigned tccid(tccId(_id));
      if(tccid <= 36 || tccid >= 73){ // EE
        unsigned bin(ttId(_id));
        bool outer((tccid >= 19 && tccid <= 36) || (tccid >= 73 && tccid <= 90));
        if(outer) bin += 48;
        bin += (tccid % 2) * (outer ? 16 : 24);
        return bin;
      }
      else
        return ttId(_id);
    }
    else
      break;
  case kEcal:
    if(_btype == kDCC)
      return dccId(_id);
    else if(_btype == kTCC)
      return tccId(_id);
    else
      break;
  case kEB:
    if(_btype == kDCC)
      return dccId(_id) - 9;
    else if(_btype == kTCC)
      return tccId(_id) - 36;
    else
      break;
  case kEEm:
    if(_btype == kDCC)
      return dccId(_id);
    else if(_btype == kTCC)
      return tccId(_id);
    else
      break;
  case kEEp:
    if(_btype == kDCC)
      return dccId(_id) - 45;
    else if(_btype == kTCC)
      return tccId(_id) - 72;
    else
      break;
  case kEE:
    if(_btype == kDCC){
      int bin(dccId(_id));
      if(bin >= 46) bin -= 36;
      return bin;
    }
    else if(_btype == kTCC){
      int bin(tccId(_id));
      if(bin >= 72) bin -= 36;
      return bin;
    }
    else
      break;
  case kSMMEM:
  case kEBSMMEM:
  case kEESMMEM:
    if(_btype == kCrystal)
      return EcalPnDiodeDetId(_id).iPnId();
    else
      break;
  default:
    break;
  }

  return 0;
}

int
EcalDQMBinningService::findBin1D(ObjectType _otype, BinningType _btype, const EcalElectronicsId& _id) const
{
  switch(_otype){
  case kSM:
  case kEBSM:
  case kEESM:
    if(_btype == kSuperCrystal)
      return towerId(_id);
    else if(_btype == kTriggerTower){
      unsigned tccid(tccId(_id));
      if(tccid <= 36 || tccid >= 73){ // EE
        unsigned bin(ttId(_id));
        bool outer((tccid >= 19 && tccid <= 36) || (tccid >= 73 && tccid <= 90));
        if(outer) bin += 48;
        bin += (tccid % 2) * (outer ? 16 : 24);
        return bin;
      }
      else
        return ttId(_id);
    }
    else
      break;
  case kEcal:
    if(_btype == kDCC)
      return dccId(_id);
    else if(_btype == kTCC)
      return tccId(_id);
    else
      break;
  case kEB:
    if(_btype == kDCC)
      return dccId(_id) - 9;
    else if(_btype == kTCC)
      return tccId(_id) - 36;
    else
      break;
  case kEEm:
    if(_btype == kDCC)
      return dccId(_id);
    else if(_btype == kTCC)
      return tccId(_id);
    else
      break;
  case kEEp:
    if(_btype == kDCC)
      return dccId(_id) - 45;
    else if(_btype == kTCC)
      return tccId(_id) - 72;
    else
      break;
  case kEE:
    if(_btype == kDCC){
      int bin(dccId(_id));
      if(bin >= 46) bin -= 36;
      return bin;
    }
    else if(_btype == kTCC){
      int bin(tccId(_id));
      if(bin >= 72) bin -= 36;
      return bin;
    }
    else
      break;
  default:
    break;
  }

  return 0;
}

int
EcalDQMBinningService::findBin1D(ObjectType _otype, BinningType _btype, unsigned _dcctccid) const
{
  if(_otype == kEcal && _btype == kDCC)
    return _dcctccid;
  else if(_otype == kEcal && _btype == kTCC)
    return _dcctccid;
  if(_otype == kEB && _btype == kDCC)
    return _dcctccid - 9;
  else if(_otype == kEB && _btype == kTCC)
    return _dcctccid - 36;
  else if(_otype == kEEm && _btype == kDCC)
    return _dcctccid;
  else if(_otype == kEEm && _btype == kTCC)
    return _dcctccid;
  else if(_otype == kEEp && _btype == kDCC)
    return _dcctccid - 45;
  else if(_otype == kEEp && _btype == kTCC)
    return _dcctccid - 72;
  else if(_otype == kEE && _btype == kDCC)
    return _dcctccid <= 9 ? _dcctccid : _dcctccid - 36;
  else if(_otype == kEE && _btype == kTCC)
    return _dcctccid <= 36 ? _dcctccid : _dcctccid - 36;

  return 0;
}

int
EcalDQMBinningService::findBin2D(ObjectType _otype, BinningType _btype, const DetId& _id) const
{
  if(_otype >= nObjType || _btype >= unsigned(nPresetBinnings)) return 0;

  switch(_btype){
  case kCrystal:
    return findBinCrystal_(_otype, _id);
    break;
  case kTriggerTower:
    return findBinTriggerTower_(_otype, _id);
    break;
  case kSuperCrystal:
    return findBinSuperCrystal_(_otype, _id);
    break;
  default :
    return 0;
  }
}

int
EcalDQMBinningService::findBin2D(ObjectType _otype, BinningType _btype, const EcalElectronicsId &_id) const
{
  if(_otype >= nObjType || _btype >= unsigned(nPresetBinnings)) return 0;

  switch(_btype){
  case kCrystal:
    return findBinCrystal_(_otype, _id);
    break;
  case kSuperCrystal:
    return findBinSuperCrystal_(_otype, _id);
    break;
  default :
    return 0;
  }
}

int
EcalDQMBinningService::findBin2D(ObjectType _otype, BinningType _btype, unsigned _dccid) const
{
  if(_otype != kEcal || _btype != kDCC) return 0;

  int nbinsX(9);
  unsigned iDCC(_dccid - 1);
  int xbin(0);
  if(iDCC <= kEEmHigh || iDCC >= kEEpLow) xbin = (iDCC + 6) % nbinsX + 1;
  else xbin = iDCC % nbinsX + 1;
  int ybin(6 - iDCC / nbinsX);

  return (nbinsX + 2) * ybin + xbin;
}

unsigned
EcalDQMBinningService::findPlot(ObjectType _otype, const DetId &_id) const
{
  if(getNObjects(_otype) == 1) return 0;

  switch(_otype){
  case kEcal3P:
    if(_id.subdetId() == EcalBarrel) return 1;
    else if(_id.subdetId() == EcalEndcap && zside(_id) > 0) return 2;
    else if(_id.subdetId() == EcalTriggerTower){
      if(!isEndcapTTId(_id)) return 1;
      else{
        if(zside(_id) > 0) return 2;
        else return 0;
      }
    }
    else return 0;

  case kEcal2P:
    if(_id.subdetId() == EcalBarrel) return 1;
    else if(_id.subdetId() == EcalTriggerTower && !isEndcapTTId(_id)) return 1;
    else return 0;

  case kEE2P:
    if(zside(_id) > 0) return 1;
    else return 0;

  case kMEM2P:
    if(_id.subdetId() == EcalLaserPnDiode){
      unsigned iDCC(dccId(_id) - 1);
      if(iDCC >= kEBmLow && iDCC <= kEBpHigh) return 1;
      else return 0;
    }
    else return -1;

  default:
    return findPlot(_otype, dccId(_id));
  }
}

unsigned
EcalDQMBinningService::findPlot(ObjectType _otype, const EcalElectronicsId &_id) const
{
  if(getNObjects(_otype) == 1) return 0;

  return findPlot(_otype, _id.dccId());
}

unsigned
EcalDQMBinningService::findPlot(ObjectType _otype, unsigned _dcctccid, BinningType _btype/* = kDCC*/) const
{
  if(getNObjects(_otype) == 1) return 0;

  unsigned iSM(_dcctccid - 1);

  switch(_otype){
  case kSM:
    return iSM;

  case kEBSM:
    return iSM - 9;

  case kEESM:
    if(iSM <= kEEmHigh) return iSM;
    else return iSM - nEBDCC;

  case kSMMEM:
    return memDCCIndex(_dcctccid);

  case kEBSMMEM:
    return memDCCIndex(_dcctccid) - nEEDCCMEM / 2;

  case kEESMMEM:
    if(iSM <= kEEmHigh) return memDCCIndex(_dcctccid);
    else return memDCCIndex(_dcctccid) - nEBDCC;

  case kEcal2P:
    if(_btype == kDCC){
      if(iSM <= kEEmHigh || iSM >= kEEpLow) return 0;
      else return 1;
    }
    else if(_btype == kTCC){
      if(iSM <= kEEmTCCHigh || iSM >= kEEpTCCLow) return 0;
      else return 1;
    }
    else{
      if(iSM == kEB) return 1;
      else return 0;
    }

  case kEcal3P:
    if(_btype == kDCC){
      if(iSM <= kEEmHigh) return 0;
      else if(iSM <= kEBpHigh) return 1;
      else return 2;
    }
    else if(_btype == kTCC){
      if(iSM <= kEEmTCCHigh) return 0;
      else if(iSM <= kEBTCCHigh) return 1;
      else return 2;
    }
    else{
      if(iSM == kEEm) return 0;
      else if(iSM == kEB) return 1;
      else return 2;
    }

  case kEE2P:
    if(_btype == kDCC){
      if(iSM >= kEEpLow) return 1;
      else return 0;
    }
    else{
      if(iSM >= kEEpTCCLow) return 1;
      else return 0;
    }

  case kMEM2P:
    if(_btype == kDCC){
      if(iSM <= kEEmHigh || iSM >= kEEpLow) return 0;
      else return 1;
    }
    else if(_btype == kTCC) return -1;
    else{
      if(iSM == kEB) return 1;
      else return 0;
    }
  default:
    return -1;
  }
}

/*static*/
EcalDQMBinningService::ObjectType
EcalDQMBinningService::getObject(ObjectType _otype, unsigned _iObj)
{
  switch(_otype){
  case kEcal3P:
    switch(_iObj){
    case 0: return kEEm;
    case 1: return kEB;
    case 2: return kEEp;
    default: return nObjType;
    }
  case kEcal2P:
    switch(_iObj){
    case 0: return kEE;
    case 1: return kEB;
    default: return nObjType;
    }
  case kEE2P:
    switch(_iObj){
    case 0: return kEEm;
    case 1: return kEEp;
    default: return nObjType;
    }
  case kMEM2P:
    switch(_iObj){
    case 0: return kEEMEM;
    case 1: return kEBMEM;
    default: return nObjType;
    }
  default:
    return _otype;
  }
}

/*static*/
unsigned
EcalDQMBinningService::getNObjects(ObjectType _otype)
{
  switch(_otype){
  case kSM:
    return nDCC;
  case kEBSM:
    return nEBDCC;
  case kEESM:
    return nEEDCC;
  case kSMMEM:
    return nDCCMEM;
  case kEBSMMEM:
    return nEBDCC;
  case kEESMMEM:
    return nEEDCCMEM;
  case kEcal2P:
    return 2;
  case kEcal3P:
    return 3;
  case kEE2P:
    return 2;
  case kMEM2P:
    return 2;
  default:
    return 1;
  }
}

bool
EcalDQMBinningService::isValidIdBin(ObjectType _otype, BinningType _btype, unsigned _iME, int _bin) const
{
  if(_otype == kEEm || _otype == kEEp){
    if(_btype == kCrystal || _btype == kTriggerTower)
      return EEDetId::validDetId(_bin % 102, _bin / 102, 1);
    else if(_btype == kSuperCrystal)
      return EcalScDetId::validDetId(_bin % 22, _bin / 22, 1);
  }
  else if(_otype == kEE){
    if(_btype == kCrystal || _btype == kTriggerTower){
      int ix(_bin % 202);
      if(ix > 100) ix = (ix - 100) % 101;
      return EEDetId::validDetId(ix, _bin / 202, 1);
    }
    else if(_btype == kSuperCrystal){
      int ix(_bin % 42);
      if(ix > 20) ix = (ix - 20) % 21;
      return EcalScDetId::validDetId(ix, _bin / 42, 1);
    }
  }
  else if(_otype == kSM || _otype == kEESM){
    unsigned iSM(_iME);
    if(_otype == kEESM && iSM > kEEmHigh) iSM += nEBDCC;

    if(iSM >= kEBmLow && iSM <= kEBpHigh) return true;
    
    if(_btype == kCrystal || _btype == kTriggerTower){
      int nX(nEESMX);
      if(iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08) nX = nEESMXExt;
      int ix(_bin % (nX + 2) + xlow(iSM));
      int iy(_bin / (nX + 2) + ylow(iSM));
      int z(iSM <= kEEmHigh ? -1 : 1);
      return EEDetId::validDetId(ix, iy, 1) && iSM == dccId(EEDetId(ix, iy, z)) - 1;
    }
    else if(_btype == kSuperCrystal){
      int nX(nEESMX / 5);
      if(iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08) nX = nEESMXExt / 5;
      int ix(_bin % (nX + 2) + xlow(iSM) / 5);
      int iy(_bin / (nX + 2) + ylow(iSM) / 5);
      int z(iSM <= kEEmHigh ? -1 : 1);
      return EcalScDetId::validDetId(ix, iy, z) && iSM == dccId(EcalScDetId(ix, iy, z)) - 1;
    }
  }

  return true;
}

/*static*/
int
EcalDQMBinningService::xlow(int _iSM)
{
  using namespace ecaldqm;

  switch(_iSM){
  case kEEm01: case kEEp01: return 15;
  case kEEm02: case kEEp02: return 0;
  case kEEm03: case kEEp03: return 0;
  case kEEm04: case kEEp04: return 5;
  case kEEm05: case kEEp05: return 30;
  case kEEm06: case kEEp06: return 55;
  case kEEm07: case kEEp07: return 60;
  case kEEm08: case kEEp08: return 55;
  case kEEm09: case kEEp09: return 45;
  default: break;
  }

  if(_iSM >= kEBmLow && _iSM <= kEBpHigh) return 0;

  return 0;
}

/*static*/
int
EcalDQMBinningService::ylow(int _iSM)
{
  using namespace ecaldqm;

  switch(_iSM){
  case kEEm01: case kEEp01: case kEEm09: case kEEp09: return 60;
  case kEEm02: case kEEp02: case kEEm08: case kEEp08: return 50;
  case kEEm03: case kEEp03: case kEEm07: case kEEp07: return 25;
  case kEEm04: case kEEp04: case kEEm06: case kEEp06: return 5;
  case kEEm05: case kEEp05: return 0;
  default: break;
  }

  if(_iSM >= kEBmLow && _iSM <= kEBmHigh) return ((_iSM - kEBmLow) % 18) * 20;
  if(_iSM >= kEBpLow && _iSM <= kEBpHigh) return (-1 - ((_iSM - kEBpLow) % 18)) * 20;

  return 0;
}

std::string
EcalDQMBinningService::channelName(uint32_t _rawId, BinningType _btype/* = kDCC*/) const
{
  // assume the following IDs for respective binning types:
  // Crystal: EcalElectronicsId
  // TriggerTower: EcalTriggerElectronicsId (pstrip and channel ignored)
  // SuperCrystal: EcalElectronicsId (strip and crystal ignored)
  // TCC: TCC ID
  // DCC: DCC ID

  std::stringstream ss;

  switch(_btype){
  case kCrystal:
    {
      // EB-03 DCC 12 CCU 12 strip 3 xtal 1 (EB ieta -13 iphi 60) (TCC 39 TT 12 pstrip 3 chan 1)
      EcalElectronicsId eid(_rawId);
      if(eid.towerId() >= 69)
        ss << smName(eid.dccId()) << " DCC " << eid.dccId() << " CCU " << eid.towerId() << " PN " << eid.xtalId();
      else{
        ss << smName(eid.dccId()) << " DCC " << eid.dccId() << " CCU " << eid.towerId() << " strip " << eid.stripId() << " xtal " << eid.xtalId();

        if(eid.dccId() >= kEBmLow + 1 && eid.dccId() <= kEBpHigh + 1){
          EBDetId ebid(getElectronicsMap()->getDetId(eid));
          ss << " (EB ieta " << std::showpos << ebid.ieta() << std::noshowpos << " iphi " << ebid.iphi() << ")";
        }
        else{
          EEDetId eeid(getElectronicsMap()->getDetId(eid));
          ss << " (EE ix " << eeid.ix() << " iy " << eeid.iy() << ")";
        }
        EcalTriggerElectronicsId teid(getElectronicsMap()->getTriggerElectronicsId(eid));
        ss << " (TCC " << teid.tccId() << " TT " << teid.ttId() << " pstrip " << teid.pseudoStripId() << " chan " << teid.channelId() << ")";
        break;
      }
    }
  case kTriggerTower:
    {
      // EB-03 DCC 12 TCC 18 TT 3
      EcalTriggerElectronicsId teid(_rawId);
      EcalElectronicsId eid(getElectronicsMap()->getElectronicsId(teid));
      ss << smName(eid.dccId()) << " DCC " << eid.dccId() << " TCC " << teid.tccId() << " TT " << teid.ttId();
      break;
    }
  case kSuperCrystal:
    {
      // EB-03 DCC 12 CCU 18 (EBTT ieta -13 iphi 60)
      EcalElectronicsId eid(_rawId);
      ss << smName(eid.dccId()) << " DCC " << eid.dccId() << " CCU " << eid.towerId();
      if(eid.dccId() >= kEBmLow + 1 && eid.dccId() <= kEBpHigh + 1){
	EcalTrigTowerDetId ttid(EBDetId(getElectronicsMap()->getDetId(eid)).tower());
	ss << " (EBTT ieta " << std::showpos << ttid.ieta() << std::noshowpos << " iphi " << ttid.iphi() << ")";
      }
      else{
	EcalScDetId scid(EEDetId(getElectronicsMap()->getDetId(eid)).sc());
	ss << " (EESC ix " << scid.ix() << " iy " << scid.iy() << ")";
      }
      break;
    }
  case kTCC:
    {
      // EB-03 TCC 12
      int tccid(_rawId - nDCC);
      int dccid(getElectronicsMap()->DCCid(getElectronicsMap()->getTrigTowerDetId(tccid, 1)));
      ss << smName(dccid) << " TCC " << (_rawId - nDCC);
      break;
    }
  case kDCC:
    ss << smName(_rawId);
    break;
  default:
    break;
  }

  return ss.str();
}

uint32_t
EcalDQMBinningService::idFromName(std::string const& _name) const
{
  TString name(_name);
  TPRegexp re("(EB|EE)([+-][0-9][0-9])(?: TCC ([0-9]+)| DCC ([0-9]+) (CCU|TCC) ([0-9]+)(?: (TT|strip|PN) ([0-9]+)(?: xtal ([0-9]+)|)|)|)");
  //            1      2                       3             4        5         6           7             8                9
  uint32_t rawId(0);

  TObjArray* matches(re.MatchS(name));
  matches->SetOwner(true);
  if(matches->GetEntries() == 0) return 0;
  else if(matches->GetEntries() == 3){
    TString subdet(static_cast<TObjString*>(matches->At(1))->GetString());
    if(subdet == "EB"){
      int dccid(static_cast<TObjString*>(matches->At(2))->GetString().Atoi());
      unsigned offset(0);
      if(dccid < 0){
	dccid *= -1;
	offset = kEEmLow;
      }
      else offset = kEEpLow;
      rawId = (dccid + 2) % 9 + 1 + offset;
    }
    else{
      int dccid(static_cast<TObjString*>(matches->At(2))->GetString().Atoi());
      if(dccid < 0) dccid *= -1;
      else dccid += 18;
      rawId = kEBmLow + dccid;
    }
  }
  else if(matches->GetEntries() == 4)
    rawId = static_cast<TObjString*>(matches->At(3))->GetString().Atoi() + nDCC;
  else{
    TString subtype(static_cast<TObjString*>(matches->At(5))->GetString());
    if(subtype == "TCC"){
      int tccid(static_cast<TObjString*>(matches->At(6))->GetString().Atoi());
      int ttid(static_cast<TObjString*>(matches->At(8))->GetString().Atoi());
      rawId = EcalTriggerElectronicsId(tccid, ttid, 1, 1).rawId();
    }
    else{
      int dccid(static_cast<TObjString*>(matches->At(4))->GetString().Atoi());
      int towerid(static_cast<TObjString*>(matches->At(6))->GetString().Atoi());
      if(matches->GetEntries() == 7)
	rawId = EcalElectronicsId(dccid, towerid, 1, 1).rawId();
      else{
        TString chType(static_cast<TObjString*>(matches->At(7))->GetString());
	int stripOrPNid(static_cast<TObjString*>(matches->At(8))->GetString().Atoi());
        if(chType == "PN")
          rawId = EcalElectronicsId(dccid, towerid, 1, stripOrPNid).rawId();
        else if(chType == "strip"){
          int xtalid(static_cast<TObjString*>(matches->At(9))->GetString().Atoi());
          rawId = EcalElectronicsId(dccid, towerid, stripOrPNid, xtalid).rawId();
        }
        // case "TT" is already taken care of
      }
    }
  }
  
  delete matches;

  return rawId;
}

uint32_t
EcalDQMBinningService::idFromBin(ObjectType _otype, BinningType _btype, unsigned _iME, int _bin) const
{
  if(_otype == kEB){
    if(_btype == kCrystal){
      int ieta(_bin / 362 - 86);
      if(ieta >= 0) ++ieta;
      return EBDetId(ieta, _bin % 362);
    }
    else if(_btype == kTriggerTower || _btype == kSuperCrystal){
      int ieta(_bin / 74 - 17);
      int z(1);
      if(ieta <= 0){
        z = -1;
        ieta = -ieta + 1;
      }
      return EcalTrigTowerDetId(z , EcalBarrel, ieta, (_bin % 74 + 69) % 72 + 1);
    }
  }
  else if(_otype == kEEm || _otype == kEEp){
    int z(_otype == kEEm ? -1 : 1);
    if(_btype == kCrystal || _btype == kTriggerTower)
      return EEDetId(_bin % 102, _bin / 102, z).rawId();
    else if(_btype == kSuperCrystal)
      return EcalScDetId(_bin % 22, _bin / 22, z).rawId();
  }
  else if(_otype == kEE){
    if(_btype == kCrystal || _btype == kTriggerTower){
      int ix(_bin % 202);
      int z(ix > 100 ? 1 : -1);
      if(z > 0) ix = (ix - 100) % 101;
      return EEDetId(ix, _bin / 202, z).rawId();
    }
    else if(_btype == kSuperCrystal){
      int ix(_bin % 42);
      int z(ix > 20 ? 1 : -1);
      if(z > 0) ix = (ix - 20) % 21;
      return EcalScDetId(ix, _bin / 42, z).rawId();
    }
  }
  else if(_otype == kSM || _otype == kEBSM || _otype == kEESM){
    unsigned iSM(_iME);
    if(_otype == kEBSM) iSM += 9;
    else if(_otype == kEESM && iSM > kEEmHigh) iSM += nEBDCC;

    int z(iSM <= kEBmHigh ? -1 : 1);

    if((iSM >= kEBmLow && iSM <= kEBmHigh) || (iSM >= kEBpLow && iSM <= kEBpHigh)){
      if(_btype == kCrystal){
        int iphi(((iSM - 9) % 18) * 20 + (z < 0 ? _bin / 87 : 21 - _bin / 87));
        int ieta((_bin % 87) * z);
        return EBDetId(ieta, iphi).rawId();
      }
      else if(_btype == kTriggerTower || _btype == kSuperCrystal){
        int iphi((((iSM - 9) % 18) * 4 + (z < 0 ? _bin / 19 : 5 - _bin / 19) + 69) % 72 + 1);
        int ieta(_bin % 19);
        return EcalTrigTowerDetId(z, EcalBarrel, ieta, iphi).rawId();
      }
    }
    else{
      if(_btype == kCrystal || _btype == kTriggerTower){
        int nX(nEESMX);
        if(iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08) nX = nEESMXExt;
        return EEDetId(_bin % (nX + 2) + xlow(iSM), _bin / (nX + 2) + ylow(iSM), z).rawId();
      }
      else if(_btype == kSuperCrystal){
        int nX(nEESMX / 5);
        if(iSM == kEEm02 || iSM == kEEm08 || iSM == kEEp02 || iSM == kEEp08) nX = nEESMXExt / 5;
        return EcalScDetId(_bin % (nX + 2) + xlow(iSM) / 5, _bin / (nX + 2) + ylow(iSM) / 5, z).rawId();
      }
    }
  }

  return 0;
}

/*static*/
EcalDQMBinningService::AxisSpecs const*
EcalDQMBinningService::formAxis(edm::ParameterSet const& _axisParams)
{
  AxisSpecs* axis(new AxisSpecs);

  axis->nbins = _axisParams.getUntrackedParameter<int>("nbins", 0);

  if(_axisParams.existsAs<std::vector<double> >("edges", false)){
    std::vector<double> const& vEdges(_axisParams.getUntrackedParameter<std::vector<double> >("edges"));
    axis->nbins = vEdges.size() - 1;
    axis->edges = new double[vEdges.size()];
    for(unsigned iE(0); iE < vEdges.size(); iE++)
      axis->edges[iE] = vEdges[iE];
  }
  else{
    axis->low = _axisParams.getUntrackedParameter<double>("low", 0.);
    axis->high = _axisParams.getUntrackedParameter<double>("high", 0);
  }

  axis->title = _axisParams.getUntrackedParameter<std::string>("title", "");

  return axis;
}

/*static*/
EcalDQMBinningService::ObjectType
EcalDQMBinningService::getObjectType(std::string const& _otypeName)
{
  if(_otypeName == "EB") return kEB;
  else if(_otypeName == "EE") return kEE;
  else if(_otypeName == "EEm") return kEEm;
  else if(_otypeName == "EEp") return kEEp;
  else if(_otypeName == "SM") return kSM;
  else if(_otypeName == "EBSM") return kEBSM;
  else if(_otypeName == "EESM") return kEESM;
  else if(_otypeName == "SMMEM") return kSMMEM;
  else if(_otypeName == "EBSMMEM") return kEBSMMEM;
  else if(_otypeName == "EESMMEM") return kEESMMEM;
  else if(_otypeName == "Ecal") return kEcal;
  else if(_otypeName == "MEM") return kMEM;
  else if(_otypeName == "EBMEM") return kEBMEM;
  else if(_otypeName == "EEMEM") return kEEMEM;
  else if(_otypeName == "Ecal2P") return kEcal2P;
  else if(_otypeName == "Ecal3P") return kEcal3P;
  else if(_otypeName == "EE2P") return kEE2P;
  else if(_otypeName == "MEM2P") return kMEM2P;
  else if(_otypeName == "Channel") return kChannel;
  else if(_otypeName == "None") return nObjType;

  throw cms::Exception("InvalidConfiguration") << "No object type " << _otypeName << " defined";
}

/*static*/
EcalDQMBinningService::BinningType
EcalDQMBinningService::getBinningType(std::string const& _btypeName)
{
  if(_btypeName == "Crystal") return kCrystal;
  else if(_btypeName == "TriggerTower") return kTriggerTower;
  else if(_btypeName == "SuperCrystal") return kSuperCrystal;
  else if(_btypeName == "TCC") return kTCC;
  else if(_btypeName == "DCC") return kDCC;
  else if(_btypeName == "ProjEta") return kProjEta;
  else if(_btypeName == "ProjPhi") return kProjPhi;
  else if(_btypeName == "User") return kUser;
  else if(_btypeName == "Report") return kReport;
  else if(_btypeName == "Trend") return kTrend;

  throw cms::Exception("InvalidConfiguration") << "No binning type " << _btypeName << " defined";
}
