#include "../interface/EcalDQMClientUtils.h"

#include <vector>

#include "DQM/EcalCommon/interface/MESet.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

namespace ecaldqm {

  EcalDQMChannelStatus const* channelStatus(0);
  EcalDQMTowerStatus const* towerStatus(0);

  float
  maskQuality(BinService::BinningType _btype, DetId const& _id, uint32_t _mask, int _quality)
  {
    using namespace std;

    bool doMask(false);

    // turn off masking for good channel for the time being
    // update the RP then enable again
    if(_quality != 1 && channelStatus && towerStatus){
      bool searchTower(_btype == BinService::kTriggerTower || _btype == BinService::kSuperCrystal);

      switch(_id.subdetId()){
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

    return doMask ? _quality + 3. : _quality;
  }

  void
  setStatuses(EcalDQMChannelStatus const* _chStatus, EcalDQMTowerStatus const* _towStatus)
  {
    channelStatus = _chStatus;
    towerStatus = _towStatus;
  }
}
