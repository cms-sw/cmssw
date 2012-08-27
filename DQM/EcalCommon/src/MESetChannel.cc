#include "DQM/EcalCommon/interface/MESetChannel.h"
#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"

#include <limits>
#include <iomanip>

namespace ecaldqm
{
  const unsigned maxTableSize(100);

  MESetChannel::MESetChannel(std::string const& _fullPath, BinService::ObjectType _otype, BinService::BinningType _btype, MonitorElement::Kind _kind) :
    MESet(_fullPath, _otype, _btype, _kind),
    meTable_()
  {
    switch(kind_){
    case MonitorElement::DQM_KIND_TH1F:
    case MonitorElement::DQM_KIND_TPROFILE:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }

    switch(btype_){
    case BinService::kCrystal:
    case BinService::kTriggerTower:
    case BinService::kSuperCrystal:
    case BinService::kTCC:
    case BinService::kDCC:
      break;
    default:
      throw_("MESetChannel configured with wrong binning type");
    }
  }

  MESetChannel::MESetChannel(MESetChannel const& _orig) :
    MESet(_orig),
    meTable_(_orig.meTable_)
  {
  }

  MESetChannel::~MESetChannel()
  {
  }

  MESet&
  MESetChannel::operator=(MESet const& _rhs)
  {
    MESetChannel const* pRhs(dynamic_cast<MESetChannel const*>(&_rhs));
    if(pRhs) meTable_ = pRhs->meTable_;
    return MESet::operator=(_rhs);
  }

  MESet*
  MESetChannel::clone() const
  {
    return new MESetChannel(*this);
  }

  void
  MESetChannel::book()
  {
    clear();
    active_ = true;
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
    dqmStore_->rmdir(dir_);
    meTable_.clear();
  }

  void
  MESetChannel::fill(DetId const& _id, double _w/* = 1.*/, double, double)
  {
    if(!active_) return;

    unsigned iME(preparePlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return;
    checkME_(iME);

    mes_[iME]->Fill(0.5, _w);
  }

  void
  MESetChannel::fill(EcalElectronicsId const& _id, double _w/* = 1.*/, double, double)
  {
    if(!active_) return;

    unsigned iME(preparePlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return;
    checkME_(iME);

    mes_[iME]->Fill(0.5, _w);
  }

  void
  MESetChannel::fill(unsigned, double, double, double)
  {
  }

  void
  MESetChannel::setBinContent(DetId const& _id, double _content)
  {
    if(!active_) return;

    unsigned iME(preparePlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return;
    checkME_(iME);

    mes_[iME]->setBinContent(1, _content);
  }
   
  void
  MESetChannel::setBinContent(EcalElectronicsId const& _id, double _content)
  {
    if(!active_) return;

    unsigned iME(preparePlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return;
    checkME_(iME);

    mes_[iME]->setBinContent(1, _content);
  }

  void
  MESetChannel::setBinContent(unsigned, double)
  {
  }

  void
  MESetChannel::setBinError(DetId const& _id, double _error)
  {
    if(!active_) return;

    unsigned iME(preparePlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return;
    checkME_(iME);

    mes_[iME]->setBinError(1, _error);
  }
   
  void
  MESetChannel::setBinError(EcalElectronicsId const& _id, double _error)
  {
    if(!active_) return;

    unsigned iME(preparePlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return;
    checkME_(iME);

    mes_[iME]->setBinError(1, _error);
  }

  void
  MESetChannel::setBinError(unsigned, double)
  {
  }

  void
  MESetChannel::setBinEntries(DetId const& _id, double _entries)
  {
    if(!active_) return;

    unsigned iME(preparePlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return;
    checkME_(iME);

    mes_[iME]->setBinEntries(1, _entries);
  }
   
  void
  MESetChannel::setBinEntries(EcalElectronicsId const& _id, double _entries)
  {
    if(!active_) return;

    unsigned iME(preparePlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return;
    checkME_(iME);

    mes_[iME]->setBinEntries(1, _entries);
  }

  void
  MESetChannel::setBinEntries(unsigned, double)
  {
  }

  double
  MESetChannel::getBinContent(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(findPlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return 0.;
    checkME_(iME);

    return mes_[iME]->getBinContent(1);
  }

  double
  MESetChannel::getBinContent(EcalElectronicsId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(findPlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return 0.;
    checkME_(iME);

    return mes_[iME]->getBinContent(1);
  }

  double
  MESetChannel::getBinContent(unsigned, int) const
  {
    return 0.;
  }

  double
  MESetChannel::getBinError(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(findPlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return 0.;
    checkME_(iME);

    return mes_[iME]->getBinError(1);
  }

  double
  MESetChannel::getBinError(EcalElectronicsId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(findPlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return 0.;
    checkME_(iME);

    return mes_[iME]->getBinError(1);
  }

  double
  MESetChannel::getBinError(unsigned, int) const
  {
    return 0.;
  }

  double
  MESetChannel::getBinEntries(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(findPlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return 0.;
    checkME_(iME);

    return mes_[iME]->getBinEntries(1);
  }

  double
  MESetChannel::getBinEntries(EcalElectronicsId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(findPlot_(getIndex_(_id)));
    if(iME == unsigned(-1)) return 0.;
    checkME_(iME);

    return mes_[iME]->getBinEntries(1);
  }

  double
  MESetChannel::getBinEntries(unsigned, int) const
  {
    return 0.;
  }

  void
  MESetChannel::reset(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    if(!active_) return;

    if(_content == 0. && _entries == 0.){
      mes_.clear();
      meTable_.clear();
      dqmStore_->rmdir(dir_);
      return;
    }

    unsigned nME(mes_.size());
    for(unsigned iME(0); iME < nME; iME++){
      mes_[iME]->setBinContent(1, _content);
      mes_[iME]->setBinError(1, _err);
      if(kind_ == MonitorElement::DQM_KIND_TPROFILE)
	mes_[iME]->setBinEntries(1, _entries);
    }
  }

  void
  MESetChannel::checkDirectory() const
  {
    using namespace std;

    vector<MonitorElement*> storeMEs(dqmStore_->getContents(dir_));

    unsigned nME(storeMEs.size());
    for(unsigned iME(0); iME < nME; iME++){
      MonitorElement* me(storeMEs[iME]);
      if(find(mes_.begin(), mes_.end(), me) != mes_.end()) continue;

      uint32_t id(binService_->idFromName(me->getName()));
      if(id != 0){
	mes_.push_back(me);
	meTable_[id] = mes_.size() - 1;
      }
    }
  }

  unsigned
  MESetChannel::preparePlot_(uint32_t _rawId) const
  {
    if(_rawId == 0) return -1;

    std::map<uint32_t, unsigned>::iterator tableItr(meTable_.find(_rawId));
    if(tableItr == meTable_.end()){
      if(meTable_.size() == maxTableSize){
        std::cout << "max table size" << std::endl;
        return -1;
      }

      std::string name(binService_->channelName(_rawId, btype_));
      std::string pwd(dqmStore_->pwd());
      dqmStore_->setCurrentFolder(dir_);

      MonitorElement* me(0);
      if(kind_ == MonitorElement::DQM_KIND_TH1F)
        me = dqmStore_->book1D(name, name, 1, 0., 1.);
      else
        me = dqmStore_->bookProfile(name, name, 1, 0., 1., -std::numeric_limits<double>::max(), std::numeric_limits<double>::max());

      dqmStore_->setCurrentFolder(pwd);

      if(me){
        mes_.push_back(me);
        tableItr = meTable_.insert(std::pair<uint32_t, unsigned>(_rawId, mes_.size() - 1)).first;
      }
    }
    if(tableItr == meTable_.end()){
      std::cout << "insertion error" << std::endl;
      return -1;
    }
    
    return tableItr->second;
  }

  unsigned
  MESetChannel::findPlot_(uint32_t _rawId) const
  {
    if(_rawId == 0) return -1;

    std::map<uint32_t, unsigned>::const_iterator tableItr(meTable_.find(_rawId));

    if(tableItr == meTable_.end()) return -1;

    return tableItr->second;
  }

  uint32_t
  MESetChannel::getIndex_(DetId const& _id) const
  {
    int subdet(_id.subdetId());

    switch(btype_){
    case BinService::kCrystal:
      if(isCrystalId(_id)) return getElectronicsMap()->getElectronicsId(_id).rawId();
      break;

    case BinService::kTriggerTower:
      if(subdet == EcalTriggerTower){
        std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
        if(ids.size() > 0)
          return getElectronicsMap()->getTriggerElectronicsId(ids[0]).rawId();
      }
      else if(isCrystalId(_id)) return getElectronicsMap()->getTriggerElectronicsId(_id).rawId();
      break;

    case BinService::kSuperCrystal:
      if(isCrystalId(_id)) return getElectronicsMap()->getElectronicsId(_id).rawId();
      else if(isEcalScDetId(_id)){
        std::pair<int, int> dccsc(getElectronicsMap()->getDCCandSC(EcalScDetId(_id)));
        return EcalElectronicsId(dccsc.first, dccsc.second, 1, 1).rawId();
      }
      else if(subdet == EcalTriggerTower && !isEndcapTTId(_id)){
        std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
        if(ids.size() > 0)
          return getElectronicsMap()->getElectronicsId(ids[0]).rawId();
      }
      break;

    case BinService::kTCC:
      return tccId(_id) + BinService::nDCC;

    case BinService::kDCC:
      return dccId(_id);

    default:
      break;
    }

    return 0;
  }

  uint32_t
  MESetChannel::getIndex_(EcalElectronicsId const& _id) const
  {
    switch(btype_){
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
      return tccId(_id) + BinService::nDCC;

    case BinService::kDCC:
      return _id.dccId();

    default:
      break;
    }

    return 0;
  }
}
