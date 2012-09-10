#include "../interface/EcalDQMClientUtils.h"

#include <vector>
#include <fstream>

#include "DQM/EcalCommon/interface/MESet.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/EcalDQMStatusDictionary.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  EcalDQMChannelStatus const* channelStatus(0);
  EcalDQMTowerStatus const* towerStatus(0);
  std::map<uint32_t, uint32_t> pnMaskMap;

  bool
  applyMask(BinService::BinningType _btype, DetId const& _id, uint32_t _mask)
  {
    using namespace std;

    int subdet(_id.subdetId());

    if(subdet == EcalLaserPnDiode){
      std::map<uint32_t, uint32_t>::const_iterator pnItr(pnMaskMap.find(_id.rawId()));
      if(pnItr == pnMaskMap.end()) return false;
      return (pnItr->second & _mask) != 0;
    }

    bool doMask(false);

    // turn off masking for good channel for the time being
    // update the RP then enable again
    if(channelStatus && towerStatus){
      bool searchTower(_btype == BinService::kTriggerTower || _btype == BinService::kSuperCrystal);

      switch(subdet){
      case EcalBarrel:
	if(searchTower){
	  EcalTrigTowerDetId ttid(EBDetId(_id).tower());
	  int tccid(getElectronicsMap()->TCCid(ttid));
	  int itt(getElectronicsMap()->iTT(ttid));
	  vector<DetId> ids(getElectronicsMap()->ttConstituents(tccid, itt));
	  for(vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
	    if(doMask) break;
	    EcalDQMChannelStatus::const_iterator cItr(channelStatus->find(idItr->rawId()));
	    if(cItr != channelStatus->end()) doMask |= cItr->getStatusCode() & _mask;
	  }
	}
	else{
	  EcalDQMChannelStatus::const_iterator cItr(channelStatus->find(_id.rawId()));
	  if(cItr != channelStatus->end()) doMask |= cItr->getStatusCode() & _mask;
	}

	if(!doMask){
	  EcalDQMTowerStatus::const_iterator tItr(towerStatus->find(EBDetId(_id).tower().rawId()));
	  if(tItr != towerStatus->end()) doMask |= tItr->getStatusCode() & _mask;
	}

	break;

      case EcalEndcap:
	if(isEcalScDetId(_id)){
	  EcalScDetId scid(_id);
	  for(int ix(1); ix <= 5; ix++){
	    for(int iy(1); iy <= 5; iy++){
	      if(doMask) break;
	      int iix((scid.ix() - 1) * 5 + ix);
	      int iiy((scid.iy() - 1) * 5 + iy);
	      if(!EEDetId::validDetId(iix, iiy, scid.zside())) continue;
	      EcalDQMChannelStatus::const_iterator cItr(channelStatus->find(EEDetId(iix, iiy, scid.zside()).rawId()));
	      if(cItr != channelStatus->end()) doMask |= cItr->getStatusCode() & _mask;
	    }
	  }

	  if(!doMask){
	    EcalDQMTowerStatus::const_iterator tItr(towerStatus->find(_id.rawId()));
	    if(tItr != towerStatus->end()) doMask |= tItr->getStatusCode() & _mask;
	  }
	}
	else{
	  if(searchTower){
	    EcalScDetId scid(EEDetId(_id).sc());
	    for(int ix(1); ix <= 5; ix++){
	      for(int iy(1); iy <= 5; iy++){
		if(doMask) break;
		int iix((scid.ix() - 1) * 5 + ix);
		int iiy((scid.iy() - 1) * 5 + iy);
		if(!EEDetId::validDetId(iix, iiy, scid.zside())) continue;
		EcalDQMChannelStatus::const_iterator cItr(channelStatus->find(EEDetId(iix, iiy, scid.zside()).rawId()));
		if(cItr != channelStatus->end()) doMask |= cItr->getStatusCode() & _mask;
	      }
	    }
	  }
	  else{
	    EcalDQMChannelStatus::const_iterator cItr(channelStatus->find(_id.rawId()));
	    if(cItr != channelStatus->end()) doMask |= cItr->getStatusCode() & _mask;
	  }

	  if(!doMask){
	    EcalDQMTowerStatus::const_iterator tItr(towerStatus->find(EEDetId(_id).sc().rawId()));
	    if(tItr != towerStatus->end()) doMask |= tItr->getStatusCode() & _mask;
	  }
	}

	break;

      case EcalTriggerTower:
	{
	  EcalTrigTowerDetId ttid(_id);
	  vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
	  for(vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr){
	    if(doMask) break;
	    EcalDQMChannelStatus::const_iterator cItr(channelStatus->find(idItr->rawId()));
	    if(cItr != channelStatus->end()) doMask |= cItr->getStatusCode() & _mask;

	    if(doMask) break;
	    if(idItr->subdetId() == EcalBarrel){
	      if(idItr != ids.begin()) continue;
	      EcalDQMTowerStatus::const_iterator tItr(towerStatus->find(EBDetId(*idItr).tower().rawId()));
	      if(tItr != towerStatus->end()) doMask |= tItr->getStatusCode() & _mask;
	    }
	    else{
	      EcalDQMTowerStatus::const_iterator tItr(towerStatus->find(EEDetId(*idItr).sc().rawId()));
	      if(tItr != towerStatus->end()) doMask |= tItr->getStatusCode() & _mask;
	    }
	  }
	}

	break;

      default:
	break;
      }
    }

    return doMask;
  }

  void
  setStatuses(EcalDQMChannelStatus const* _chStatus, EcalDQMTowerStatus const* _towStatus)
  {
    channelStatus = _chStatus;
    towerStatus = _towStatus;
  }

  void
  readPNMaskMap(std::string const& _fileName)
  {
    std::ifstream maskFile(_fileName);
    if(!maskFile.is_open())
      throw cms::Exception("IOError") << "File " << _fileName << " not found";

    EcalDQMStatusDictionary::init();

    pnMaskMap.clear();

    std::string line;
    std::stringstream ss;
    std::string key;
    int dcc(0);
    int ipn(0);
    std::string type;
    while(std::getline(maskFile, line), maskFile.good()){
      ss.clear();
      ss.str("");
      ss << line;
      ss >> key;
      if(key != "PN") continue;
      ss >> dcc >> ipn >> type;
      int subdet(dcc - 1 <= kEEmHigh || dcc - 1 >= kEEpLow ? EcalEndcap : EcalBarrel);
      EcalPnDiodeDetId pnid(subdet, dcc, ipn);
      pnMaskMap[pnid.rawId()] |= EcalDQMStatusDictionary::getCode(type);
    }

    EcalDQMStatusDictionary::clear();
  }
}
