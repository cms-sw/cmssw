#include "DQM/EcalCommon/interface/DQWorkerClient.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DQM/EcalCommon/interface/MESetChannel.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  EcalDQMChannelStatus const* DQWorkerClient::channelStatus(0);
  EcalDQMTowerStatus const* DQWorkerClient::towerStatus(0);

  DQWorkerClient::DQWorkerClient(const edm::ParameterSet &_params, const edm::ParameterSet& _paths, std::string const& _name) :
    DQWorker(_params, _paths, _name),
    sources_(0)
  {
  }

  void
  DQWorkerClient::endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &)
  {
    for(std::vector<MESet const*>::iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr){
      MESetChannel const* meset(dynamic_cast<MESetChannel const*>(*sItr));
      if(meset) meset->checkDirectory();
    }
  }

  void
  DQWorkerClient::reset()
  {
    DQWorker::reset();
    for(std::vector<MESet const*>::iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr)
      (*sItr)->clear();
  }

  void
  DQWorkerClient::initialize()
  {
    initialized_ = true;
    for(std::vector<MESet const*>::iterator sItr(sources_.begin()); sItr != sources_.end(); ++sItr)
      initialized_ &= (*sItr)->retrieve();
  }

  void
  DQWorkerClient::source_(unsigned _iS, std::string const& _worker, unsigned _iW, edm::ParameterSet const& _sources)
  {
    if(_iS >= sources_.size()) sources_.resize(_iS + 1, 0);

    std::map<std::string, std::vector<MEData> >::const_iterator dataItr(meData.find(_worker));
    if(dataItr == meData.end())
      throw cms::Exception("InvalidCall") << "DQWorker " << _worker << " is not defined";

    MEData const& data(dataItr->second.at(_iW));

    edm::ParameterSet const& workerPaths(_sources.getUntrackedParameterSet(_worker));

    std::string fullpath(workerPaths.getUntrackedParameter<std::string>(data.pathName));

    sources_.at(_iS) = createMESet_(fullpath, data, true);
  }

  void
  DQWorkerClient::fillQuality_(unsigned _iME, DetId const& _id, uint32_t _mask, float _quality)
  {
    using namespace std;

    bool doMask(false);

    // turn off masking for good channel for the time being
    // update the RP then enable again
    if(_quality != 1. && channelStatus && towerStatus){
      BinService::BinningType btype(MEs_[_iME]->getBinType());
      bool searchTower(btype == BinService::kTriggerTower || btype == BinService::kSuperCrystal);

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

    float quality(doMask ? _quality + 3. : _quality);

    MEs_[_iME]->setBinContent(_id, quality);
  }

}
