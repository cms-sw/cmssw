#include "../interface/DQWorkerClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetChannel.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  EcalDQMChannelStatus const* DQWorkerClient::channelStatus(0);
  EcalDQMTowerStatus const* DQWorkerClient::towerStatus(0);

  DQWorkerClient::DQWorkerClient(const edm::ParameterSet &_params, std::string const& _name) :
    DQWorker(_params, _name),
    sources_(0)
  {
    using namespace std;

    map<string, vector<MEData> >::iterator dItr(meData.find(name_));
    if(dItr == meData.end())
      throw cms::Exception("InvalidCall") << "MonitorElement setup data not found for " << name_ << std::endl;

    edm::ParameterSet const& myParams(_params.getUntrackedParameterSet(name_));

    if(myParams.existsAs<edm::ParameterSet>("sources")){
      edm::ParameterSet const& sources(myParams.getUntrackedParameterSet("sources"));

      vector<MEData> const& vData(dItr->second);

      for(unsigned iME(0); iME < vData.size(); iME++){
        MEData const& nameData(vData[iME]);
        if(nameData.kind != MonitorElement::DQM_KIND_INVALID) continue;

        vector<string> workerAndME(sources.getUntrackedParameter<vector<string> >(nameData.pathName));

        string worker(workerAndME[0]);
        string ME(workerAndME[1]);

        map<string, vector<MEData> >::const_iterator dataItr(meData.find(worker));
        if(dataItr == meData.end())
          throw cms::Exception("InvalidCall") << "DQWorker " << worker << " is not defined";

        MEData data;
        for(vector<MEData>::const_iterator mItr(dataItr->second.begin()); mItr != dataItr->second.end(); ++mItr)
          if(mItr->pathName == ME) data = *mItr;

        if(data.kind == MonitorElement::DQM_KIND_INVALID)
          throw cms::Exception("InvalidCall") << "DQWorker " << worker << " does not have an ME of name " << ME;

        sources_.push_back(createMESet_(data));
      }
    }
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
