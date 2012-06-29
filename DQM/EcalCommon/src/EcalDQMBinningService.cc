#include "DQM/EcalCommon/interface/EcalDQMBinningService.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

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
  cacheId_(0),
  cacheOtype_(nObjType),
  cacheBtype_(nBinType),
  etaBound_(1.479),
  geometry_(0),
  initialized_(false),
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
  geometry_ = geomHndl.product();
  if(!geometry_)
    throw cms::Exception("EventSetup") << "CaloGeometry invalid";

  initialized_ = true;
}

std::vector<EcalDQMBinningService::AxisSpecs>
EcalDQMBinningService::getBinning(ObjectType _otype, BinningType _btype, bool _isMap/* = true*/, unsigned _objOffset/* = 0*/) const
{
  if(_otype >= nObjType || _btype >= unsigned(nPresetBinnings))
    return std::vector<AxisSpecs>(0); // you are on your own

  switch(_otype){
  case kEB:
    return getBinningEB_(_btype, _isMap);
  case kEBMEM:
    return getBinningEBMEM_(_btype, _isMap);
  case kEE:
    return getBinningEE_(_btype, _isMap, 0);
  case kEEm:
    return getBinningEE_(_btype, _isMap, -1);
  case kEEp:
    return getBinningEE_(_btype, _isMap, 1);
  case kEEMEM:
    return getBinningEEMEM_(_btype, _isMap);
  case kSM:
    return getBinningSM_(_btype, _isMap, _objOffset);
  case kSMMEM:
    return getBinningSMMEM_(_btype, _isMap, _objOffset);
  case kEcal:
    return getBinningEcal_(_btype, _isMap);
  default:
    return std::vector<AxisSpecs>(0);
  }
}

int
EcalDQMBinningService::xlow(int _iSM) const
{
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

int
EcalDQMBinningService::ylow(int _iSM) const
{
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

const std::vector<int>*
EcalDQMBinningService::getBinMap(ObjectType& _okey, BinningType& _bkey) const
{
  if(unsigned(_okey) >= nPlotType || unsigned(_bkey) >= nPresetBinnings) return 0;

  if((_okey == kEEm || _okey == kEEp) && _bkey == kProjPhi)
    _okey = kEE;

  if(_bkey == kTriggerTower){
    if(_okey == kEB) _bkey = kSuperCrystal;
    else if(_okey == kEE || _okey == kEEm || _okey == kEEp) _bkey = kCrystal;
  }

  if(binMaps_[_okey][_bkey].size() != 0) return &(binMaps_[_okey][_bkey]);

  // Map is not defined yet (or is not going to be)

  const std::vector<int>* binMap(0);

  switch(_okey){
  case kEB:
    binMap = getBinMapEB_(_bkey);
    break;
  case kEBMEM:
    binMap = getBinMapEBMEM_(_bkey);
    break;
  case kEE:
    binMap = getBinMapEE_(_bkey, 0);
    break;
  case kEEm:
    binMap = getBinMapEE_(_bkey, -1);
    break;
  case kEEp:
    binMap = getBinMapEE_(_bkey, 1);
    break;
  case kEEMEM:
    binMap = getBinMapEEMEM_(_bkey);
    break;
  case kSM:
    binMap = getBinMapSM_(_bkey);
    break;
  case kSMMEM:
    binMap = getBinMapSMMEM_(_bkey);
    break;
  case kEcal:
    binMap = getBinMapEcal_(_bkey);
    break;
  default:
    return 0;
  }

  if(verbosity_ > 0){
    std::cout << "EcalDQMBinningService: Booked new binMap for " << int(_okey) << " " << int(_bkey);
    std::cout << " (Current memory usage: ";
    int bytes(0);
    for(unsigned iO(0); iO < nPlotType; iO++)
      for(unsigned iB(0); iB < nPresetBinnings; iB++)
	bytes += binMaps_[iO][iB].size() * sizeof(int);
    std::cout << bytes / 1024. << "kB)" << std::endl;
  }

  return binMap;
}

std::pair<unsigned, std::vector<int> >
EcalDQMBinningService::findBins(ObjectType _otype, BinningType _btype, const DetId &_id) const
{
  using namespace std;

  if(_otype == cacheOtype_ && _btype == cacheBtype_ && _id == cacheId_) return cache_;

  pair<unsigned, std::vector<int> > ret(-1, std::vector<int>(0));

  if(_otype >= nObjType || _btype >= unsigned(nPresetBinnings)) return ret;

  ret.first = findOffset(_otype, _id);
  if(ret.first == unsigned(-1)) return ret;
 
  // bring up the appropriate dictionary
  ObjectType okey(objectFromOffset(_otype, ret.first));
  BinningType bkey(_btype);

  if(okey == nObjType) return ret;

  const std::vector<int>* binMap(getBinMap(okey, bkey));
  if(binMap == 0) return ret;

  switch(bkey){
  case kCrystal:
    findBinsCrystal_(_id, okey, *binMap, ret.second);
    break;
  case kTriggerTower:
    findBinsTriggerTower_(_id, okey, *binMap, ret.second);
    break;
  case kSuperCrystal:
    findBinsSuperCrystal_(_id, okey, *binMap, ret.second);
    break;
  case kDCC:
    findBinsDCC_(_id, okey, *binMap, ret.second);
    break;
  case kTCC:
    findBinsTCC_(_id, okey, *binMap, ret.second);
    break;
  case kProjEta:
    findBinsProjEta_(_id, okey, *binMap, ret.second);
    break;
  case kProjPhi:
    findBinsProjPhi_(_id, okey, *binMap, ret.second);
  default :
    break;
  }

  // binMap value differs from actual bin numbers for SM plots
  if(_otype == kSM || _otype == kSMMEM){
    for(vector<int>::iterator binItr(ret.second.begin()); binItr != ret.second.end(); ++binItr)
      *binItr -= smOffsetBins(_otype, _btype, ret.first);
  }

  cacheId_ = _id;
  cacheOtype_ = _otype;
  cacheBtype_ = _btype;
  cache_ = ret;

  return ret;
}

std::pair<unsigned, std::vector<int> >
EcalDQMBinningService::findBins(ObjectType _otype, BinningType _btype, const EcalElectronicsId &_id) const
{
  return findBins(_otype, _btype, getElectronicsMap()->getDetId(_id));
}

std::pair<unsigned, std::vector<int> >
EcalDQMBinningService::findBins(ObjectType _otype, BinningType _btype, unsigned _dcctccid) const
{
  using namespace std;

  if(_otype == cacheOtype_ && _btype == cacheBtype_ && _dcctccid == cacheId_) return cache_;

  pair<unsigned, std::vector<int> > ret(-1, std::vector<int>(0));

  if(_btype != kTCC && _btype != kDCC) return ret;

  ret.first = findOffset(_otype, _btype, _dcctccid);

  if(ret.first == unsigned(-1)) return ret;

  // bring up the appropriate dictionary
  ObjectType okey(objectFromOffset(_otype, ret.first));
  BinningType bkey(_btype);

  const std::vector<int>* binMap(getBinMap(okey, bkey));
  if(binMap == 0) return ret;

  unsigned index(_dcctccid - 1);

  if(bkey == kDCC){
    if(okey == kEB) index -= kEBmLow;
    else if(okey == kEE && index >= kEEpLow) index -= (kEBpHigh - kEEmHigh);
    else if(okey == kEEp) index -= kEEpLow;
  }
  else{
    if(okey == kEB) index -= kEBTCCLow;
    else if(okey == kEE && index >= kEEpLow) index -= (kEBTCCHigh - kEEmTCCHigh);
    else if(okey == kEEp) index -= kEEpTCCLow;
  }

  ret.second.push_back(binMap->at(index));

  cacheId_ = _dcctccid;
  cacheOtype_ = _otype;
  cacheBtype_ = _btype;
  cache_ = ret;

  return ret;
}

std::pair<unsigned, std::vector<int> >
EcalDQMBinningService::findBinsNoMap(ObjectType _otype, BinningType _btype, const DetId& _id) const
{
  using namespace std;

  // Not yet implemented to scale to general cases
  if(!((_otype == kSM && _btype == kSuperCrystal) ||
       (_otype == kSM && _btype == kTriggerTower) ||
       (_otype == kEcal2P && _btype == kDCC) ||
       (_otype == kEcal2P && _btype == kTCC)))
    throw cms::Exception("NotImplemented") << "1D bin finding only for SM - SC plots or Ecal2P - DCC plots" << std::endl;

  if(_otype == kEcal2P && (_btype == kDCC || _btype == kTCC))
    return findBins(_otype, _btype, _id);

  pair<unsigned, std::vector<int> > ret(-1, std::vector<int>(0));

  ret.first = findOffset(_otype, _id);
  if(ret.first == unsigned(-1)) return ret;

  if(_otype == kSM && _btype == kSuperCrystal)
    ret.second.push_back(towerId(_id));
  else if(_otype == kSM && _btype == kTriggerTower){
    unsigned tccid(tccId(_id));
    if(tccid >= 37 && tccid <= 72) // EB
      ret.second.push_back(ttId(_id));
    else{
      unsigned bin(ttId(_id));
      tccid = (tccid - 1) % 36;
      bool outer(tccid >= 18);
      tccid = (tccid + 1) % 18; // TCC numbering is shifted wrt DCC numbering by one 
      if(outer) bin += 48;
      bin += (tccid % 2) * (outer ? 16 : 24);
      ret.second.push_back(bin);
    }
  }

  return ret;
}

std::pair<unsigned, std::vector<int> >
EcalDQMBinningService::findBinsNoMap(ObjectType _otype, BinningType _btype, const EcalElectronicsId& _id) const
{
  return findBinsNoMap(_otype, _btype, getElectronicsMap()->getDetId(_id));
}

int
EcalDQMBinningService::getBin(ObjectType _otype, BinningType _btype, unsigned _index) const
{
  if(_otype >= nObjType || _btype >= unsigned(nPresetBinnings)) return 0;

  const std::vector<int>* binMap(getBinMap(_otype, _btype));
  if(binMap == 0 || _index >= binMap->size()) return 0;

  return (*binMap)[_index];
}

unsigned
EcalDQMBinningService::findOffset(ObjectType _otype, const DetId &_id) const
{
  if(_otype == cacheOtype_ && _id == cacheId_) return cache_.first;

  unsigned iSM(dccId(_id) - 1);

  if(_otype == kEcal3P){
    if(iSM <= kEEmHigh) return findOffset(kEcal3P, kUser, (unsigned)kEEm + 1);
    else if(iSM <= kEBpHigh) return findOffset(kEcal3P, kUser, (unsigned)kEB + 1);
    else return findOffset(kEcal3P, kUser, (unsigned)kEEp + 1);
  }
  else if(_otype == kEcal2P){
    if(iSM <= kEEmHigh) return findOffset(kEcal2P, kUser, (unsigned)kEE + 1);
    else if(iSM <= kEBpHigh) return findOffset(kEcal2P, kUser, (unsigned)kEB + 1);
    else return findOffset(kEcal2P, kUser, (unsigned)kEE + 1);
  }
  else if(_otype == kEcal)
    return 0;
  else if(_otype == kEcalMEM2P){
    if(iSM <= kEEmHigh) return findOffset(kEcal2P, kUser, (unsigned)kEE + 1);
    else if(iSM <= kEBpHigh) return findOffset(kEcal2P, kUser, (unsigned)kEB + 1);
    else return findOffset(kEcal2P, kUser, (unsigned)kEE + 1);
  }

  return findOffset(_otype, kDCC, iSM + 1);
}

unsigned
EcalDQMBinningService::findOffset(ObjectType _otype, const EcalElectronicsId &_id) const
{
  return findOffset(_otype, getElectronicsMap()->getDetId(_id));
}

unsigned
EcalDQMBinningService::findOffset(ObjectType _otype, BinningType _btype, unsigned _dcctccid) const
{
  unsigned iSM(_dcctccid - 1);

  switch(_otype){
  case kEB:
    if(_btype == kDCC && iSM >= kEBmLow && iSM <= kEBpHigh) return 0;
    else if(_btype == kTCC && iSM >= kEBTCCLow && iSM <= kEBTCCHigh) return 0;
    return -1;
  case kEE:
    if(_btype == kDCC && 
       (iSM <= kEEmHigh ||
	(iSM >= kEEpLow && iSM <= kEEpHigh))) return 0;
    else if(_btype == kTCC && 
	    (iSM <= kEEmTCCHigh ||
	     (iSM >= kEEpTCCLow && iSM <= kEEpTCCHigh))) return 0;
    return -1;
  case kEEm:
    if(_btype == kDCC && iSM <= kEEmHigh) return 0;
    else if(_btype == kTCC && iSM <= kEEmTCCHigh) return 0;
    else return -1;
  case kEEp:
    if(_btype == kDCC && (iSM >= kEEpLow && iSM <= kEEpHigh)) return 0;
    else if(_btype == kTCC && (iSM >= kEEpTCCLow && iSM <= kEEpTCCHigh)) return 0;
    else return -1;
  case kSM:
    if(iSM < nDCC) return iSM;
    else return -1;
  case kSMMEM:
    if(iSM < nDCC && dccNoMEM.find(iSM) == dccNoMEM.end()) return memDCCIndex(_dcctccid);
    else return -1;
  case kEcal:
    if(_btype == kDCC && iSM < nDCC) return 0;
    else if(_btype == kTCC && iSM < nTCC) return 0;
    else if(_btype == kUser) return 0;
    else return -1;
  case kEcal2P:
    {
      int iSubdet(iSM);
      if(_btype == kDCC){
	if(iSM <= kEEmHigh) iSubdet = kEE;
	else if(iSM <= kEBpHigh) iSubdet = kEB;
	else iSubdet = kEE;
      }
      else if(_btype == kTCC){
	if(iSM <= kEEmTCCHigh) iSubdet = kEE;
	else if(iSM <= kEBTCCHigh) iSubdet = kEB;
	else iSubdet = kEE;
      }
      if(iSubdet == kEE || iSubdet == kEEm || iSubdet == kEEp) return 0;
      else if(iSubdet == kEB) return 1;
      else return -1;
    }
  case kEcal3P:
    {
      int iSubdet(iSM);
      if(_btype == kDCC){
	if(iSM <= kEEmHigh) iSubdet = kEEm;
	else if(iSM <= kEBpHigh) iSubdet = kEB;
	else iSubdet = kEEp;
      }
      else if(_btype == kTCC){
	if(iSM <= kEEmTCCHigh) iSubdet = kEEm;
	else if(iSM <= kEBTCCHigh) iSubdet = kEB;
	else iSubdet = kEEp;
      }
      if(iSubdet == kEEm) return 0;
      else if(iSubdet == kEB) return 1;
      else if(iSubdet == kEEp) return 2;
      else return -1;
    }
  case kEcalMEM2P:
    {
      int iSubdet(iSM);
      if(iSubdet == kEE || iSubdet == kEEm || iSubdet == kEEp) return 0;
      else if(iSubdet == kEB) return 1;
      else return -1;
    }
  default:
    return -1;
  }
}

EcalDQMBinningService::ObjectType
EcalDQMBinningService::objectFromOffset(ObjectType _otype, unsigned _offset) const
{
  if(_otype == kEcal3P) {
    switch(_offset){
    case 0: return kEEm;
    case 1: return kEB;
    case 2: return kEEp;
    default: return nObjType;
    }
  }
  else if(_otype == kEcal2P){
    switch(_offset){
    case 0: return kEE;
    case 1: return kEB;
    default: return nObjType;
    }
  }
  else if(_otype == kEcalMEM2P){
    switch(_offset){
    case 0: return kEEMEM;
    case 1: return kEBMEM;
    default: return nObjType;
    }
  }
  else
    return _otype;
}

int
EcalDQMBinningService::smOffsetBins(ObjectType _otype, BinningType _btype, unsigned _offset) const
{
  if(!_offset) return 0;

  switch(_otype) {
  case kSM :
    {
      int bins(0);
      int offset(_offset);

      if(offset > kEEpLow){
	int ext(0);
	if(offset > kEEp02) ext += 1;
	if(offset > kEEp08) ext += 1;
	int offBins(nEESMBins * (offset - kEEpLow) + (nEESMBinsExt - nEESMBins) * ext);
	switch(_btype){
	case kCrystal:
	case kTriggerTower:
	  bins += offBins; break;
	case kSuperCrystal:
	  bins += offBins / 25; break;
	default:
	  break;
	}
	offset = kEEpLow;
      }
      if(offset > kEBmLow){
	int offBins(nEBSMBins * (offset - kEBmLow));
	switch(_btype){
	case kCrystal:
	  bins += offBins; break;
	case kTriggerTower:
	case kSuperCrystal:
	  bins += offBins / 25; break;
	default:
	  break;
	}
	offset = kEBmLow;
      }
      if(offset > kEEmLow){
	int ext(0);
	if(offset > kEEm02) ext += 1;
	if(offset > kEEm08) ext += 1;
	int offBins(nEESMBins * (offset - kEEmLow) + (nEESMBinsExt - nEESMBins) * ext);
	switch(_btype){
	case kCrystal:
	case kTriggerTower:
	  bins += offBins; break;
	case kSuperCrystal:
	  bins += offBins / 25; break;
	default:
	  break;
	}
      }

      return bins;
    }
  case kSMMEM :
    {
      return _offset * 10;
    }
  default :
    break;
  }

  return 0;
}

std::string
EcalDQMBinningService::channelName(uint32_t _rawId, BinningType _btype/* = kDCC*/) const
{
  std::stringstream ss;

  switch(_btype){
  case kCrystal:
    {
      // EB-03 DCC 12 CCU 12 strip 3 xtal 1 (EB ieta -13 iphi 60) (TCC 39 TT 12 pstrip 3 chan 1)
      EcalElectronicsId eid(_rawId);
      ss << smName(eid.dccId()) << " DCC " << eid.dccId() << " CCU " << eid.towerId() << " strip " << eid.stripId() << " xtal " << eid.xtalId();
      if(eid.towerId() >= 69) break;

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
  TPRegexp re("(EB|EE)([+-][0-9][0-9])(?: TCC ([0-9]+)| DCC ([0-9]+) (CCU|TCC) ([0-9]+)(?: (?:TT|strip) ([0-9]+)(?: xtal ([0-9]+)|)|)|)");
  //            1      2                       3             4        5         6                        7                8
  uint32_t rawId(0);

  TObjArray* matches(re.MatchS(name));
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
      int ttid(static_cast<TObjString*>(matches->At(7))->GetString().Atoi());
      rawId = EcalTriggerElectronicsId(tccid, ttid, 1, 1).rawId();
    }
    else{
      int dccid(static_cast<TObjString*>(matches->At(4))->GetString().Atoi());
      int towerid(static_cast<TObjString*>(matches->At(6))->GetString().Atoi());
      if(matches->GetEntries() == 7)
	rawId = EcalElectronicsId(dccid, towerid, 1, 1).rawId();
      else{
	int stripid(static_cast<TObjString*>(matches->At(7))->GetString().Atoi());
	int xtalid(static_cast<TObjString*>(matches->At(8))->GetString().Atoi());
	rawId = EcalElectronicsId(dccid, towerid, stripid, xtalid).rawId();
      }
    }
  }
  
  delete matches;

  return rawId;
}
