#include "DQM/EcalCommon/interface/MESetChannel.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm
{
  MESetChannel::MESetChannel(std::string const& _fullpath, MEData const& _data, bool _readOnly/* = false*/) :
    MESet(_fullpath, _data, _readOnly),
    meTable_()
  {
  }

  MESetChannel::~MESetChannel()
  {
  }

  bool
  MESetChannel::retrieve() const
  {
    active_ = true;
    return true;
  }

  void
  MESetChannel::clear() const
  {
    MESet::clear();
    if(!readOnly_) dqmStore_->rmdir(dir_);
    mes_.clear();
    meTable_.clear();
  }

  void
  MESetChannel::fill(DetId const& _id, double _w/* = 1.*/, double, double)
  {
    uint32_t rawId(getIndex_(_id));

    std::map<uint32_t, unsigned>::iterator tableItr(meTable_.find(rawId));
    if(tableItr == meTable_.end()){
      std::string name(binService_->channelName(rawId, data_->btype));
      tableItr = append_(name, rawId);
    }
    if(tableItr == meTable_.end()) return;

    mes_.at(tableItr->second)->Fill(0.5, _w);
  }

  void
  MESetChannel::fill(EcalElectronicsId const& _id, double _w/* = 1.*/, double, double)
  {
    uint32_t rawId(getIndex_(_id));

    std::map<uint32_t, unsigned>::iterator tableItr(meTable_.find(rawId));
    if(tableItr == meTable_.end()){
      std::string name(binService_->channelName(rawId, data_->btype));
      tableItr = append_(name, rawId);
    }
    if(tableItr == meTable_.end()) return;

    mes_.at(tableItr->second)->Fill(0.5, _w);
  }

  void
  MESetChannel::setBinContent(DetId const& _id, double _content, double _err/* = 0.*/)
  {
    uint32_t rawId(getIndex_(_id));

    std::map<uint32_t, unsigned>::iterator tableItr(meTable_.find(rawId));
    if(tableItr == meTable_.end()){
      std::string name(binService_->channelName(rawId, data_->btype));
      tableItr = append_(name, rawId);
    }
    if(tableItr == meTable_.end()) return;

    mes_.at(tableItr->second)->setBinContent(1, _content);
    mes_.at(tableItr->second)->setBinError(1, _err);
  }
   
  void
  MESetChannel::setBinContent(EcalElectronicsId const& _id, double _content, double _err/* = 0.*/)
  {
    uint32_t rawId(getIndex_(_id));

    std::map<uint32_t, unsigned>::iterator tableItr(meTable_.find(rawId));
    if(tableItr == meTable_.end()){
      std::string name(binService_->channelName(rawId, data_->btype));
      tableItr = append_(name, rawId);
    }
    if(tableItr == meTable_.end()) return;

    mes_.at(tableItr->second)->setBinContent(1, _content);
    mes_.at(tableItr->second)->setBinError(1, _err);
  }

  void
  MESetChannel::reset(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    if(readOnly_) return;

    if(_content == 0. && _entries == 0.){
      mes_.clear();
      meTable_.clear();
      dqmStore_->rmdir(dir_);
      return;
    }

    for(unsigned iME(0); iME < mes_.size(); iME++){
      mes_[iME]->setBinContent(1, _content);
      mes_[iME]->setBinContent(1, _err);
      if(data_->kind == MonitorElement::DQM_KIND_TPROFILE)
	mes_[iME]->setBinEntries(1, _entries);
    }
  }

  double
  MESetChannel::getBinContent(DetId const& _id, int) const
  {
    uint32_t rawId(getIndex_(_id));

    std::map<uint32_t, unsigned>::const_iterator tableItr(meTable_.find(rawId));

    if(tableItr == meTable_.end()) return 0.;

    return mes_.at(tableItr->second)->getBinContent(1);
  }

  double
  MESetChannel::getBinContent(EcalElectronicsId const& _id, int) const
  {
    uint32_t rawId(getIndex_(_id));

    std::map<uint32_t, unsigned>::const_iterator tableItr(meTable_.find(rawId));

    if(tableItr == meTable_.end()) return 0.;

    return mes_.at(tableItr->second)->getBinContent(1);
  }

  void
  MESetChannel::checkDirectory() const
  {
    using namespace std;

    vector<MonitorElement*> storeMEs(dqmStore_->getContents(dir_));
    for(vector<MonitorElement*>::iterator storeItr(storeMEs.begin()); storeItr != storeMEs.end(); ++storeItr){
      if(find(mes_.begin(), mes_.end(), *storeItr) != mes_.end()) continue;

      uint32_t id(binService_->idFromName((*storeItr)->getName()));
      if(id != 0){
	mes_.push_back(*storeItr);
	meTable_[id] = mes_.size() - 1;
      }
    }
  }

  std::map<uint32_t, unsigned>::iterator
  MESetChannel::append_(std::string const& _name, uint32_t _rawId)
  {
    std::string pwd(dqmStore_->pwd());
    dqmStore_->setCurrentFolder(dir_);

    MonitorElement* me(dqmStore_->book1D(_name, _name, 1, 0., 1.));

    dqmStore_->setCurrentFolder(pwd);

    if(!me) return meTable_.end();

    mes_.push_back(me);
    std::pair<std::map<uint32_t, unsigned>::iterator, bool> ins(meTable_.insert(std::pair<uint32_t, unsigned>(_rawId, mes_.size() - 1)));
    
    return ins.first;
  }

  uint32_t
  MESetChannel::getIndex_(DetId const& _id) const
  {
    switch(data_->btype){
    case BinService::kCrystal:
      return getElectronicsMap()->getElectronicsId(_id).rawId();
    case BinService::kTriggerTower:
      {
	if(_id.subdetId() == EcalTriggerTower){
	  EcalTrigTowerDetId ttid(_id);
	  return EcalTriggerElectronicsId(getElectronicsMap()->TCCid(ttid), getElectronicsMap()->iTT(ttid), 1, 1).rawId();
	}
	else{
	  EcalTriggerElectronicsId teid(getElectronicsMap()->getTriggerElectronicsId(_id));
	  return EcalTriggerElectronicsId(teid.tccId(), teid.ttId(), 1, 1).rawId();
	}
      }
    case BinService::kSuperCrystal:
      {
	EcalElectronicsId eid(getElectronicsMap()->getElectronicsId(_id));
	return EcalElectronicsId(eid.dccId(), eid.towerId(), 1, 1).rawId();
      }
    case BinService::kTCC:
      {
	EcalTriggerElectronicsId teid(getElectronicsMap()->getTriggerElectronicsId(_id));
	return BinService::nDCC + teid.tccId();
      }
    case BinService::kDCC:
      {
	EcalElectronicsId eid(getElectronicsMap()->getElectronicsId(_id));
	return eid.dccId();
      }
    default:
      throw cms::Exception("InvalidConfiguration") << "MESetChannel configured with bin type " << data_->btype;
      return 0;
    }
  }

  uint32_t
  MESetChannel::getIndex_(EcalElectronicsId const& _id) const
  {
    switch(data_->btype){
    case BinService::kCrystal:
      return _id.rawId();
    case BinService::kTriggerTower:
      {
	EcalTriggerElectronicsId teid(getElectronicsMap()->getTriggerElectronicsId(_id));
	return EcalTriggerElectronicsId(teid.tccId(), teid.ttId(), 1, 1).rawId();
      }
    case BinService::kSuperCrystal:
      return EcalElectronicsId(_id.dccId(), _id.towerId(), 1, 1).rawId();
    case BinService::kTCC:
      {
	EcalTriggerElectronicsId teid(getElectronicsMap()->getTriggerElectronicsId(_id));
	return BinService::nDCC + teid.tccId();
      }
    case BinService::kDCC:
      return _id.dccId();
    default:
      throw cms::Exception("InvalidConfiguration") << "MESetChannel configured with bin type " << data_->btype;
      return 0;
    }
  }

}
