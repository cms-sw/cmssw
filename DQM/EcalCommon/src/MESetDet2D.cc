#include "DQM/EcalCommon/interface/MESetDet2D.h"

namespace ecaldqm
{

  MESetDet2D::MESetDet2D(std::string const& _fullPath, BinService::ObjectType _otype, BinService::BinningType _btype, MonitorElement::Kind _kind, BinService::AxisSpecs const* _zaxis/* = 0*/) :
    MESetEcal(_fullPath, _otype, _btype, _kind, 2, 0, 0, _zaxis)
  {
    switch(kind_){
    case MonitorElement::DQM_KIND_TH2F:
    case MonitorElement::DQM_KIND_TPROFILE2D:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }
  }

  MESetDet2D::MESetDet2D(MESetDet2D const& _orig) :
    MESetEcal(_orig)
  {
  }

  MESetDet2D::~MESetDet2D()
  {
  }

  MESet*
  MESetDet2D::clone() const
  {
    return new MESetDet2D(*this);
  }

  void
  MESetDet2D::fill(DetId const& _id, double _w/* = 1.*/, double, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin;

    if(isEndcapTTId(_id)){
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      unsigned nId(ids.size());
      for(unsigned iId(0); iId < nId; iId++){
        bin = binService_->findBin2D(obj, BinService::kTriggerTower, ids[iId]);
        fill_(iME, bin, _w);
      }
    }
    else{
      bin = binService_->findBin2D(obj, btype_, _id);
      fill_(iME, bin, _w);
    }
  }

  void
  MESetDet2D::fill(EcalElectronicsId const& _id, double _w/* = 1.*/, double, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin(binService_->findBin2D(obj, btype_, _id));
    fill_(iME, bin, _w);
  }

  void
  MESetDet2D::setBinContent(DetId const& _id, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin;

    if(isEndcapTTId(_id)){
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      unsigned nId(ids.size());
      for(unsigned iId(0); iId < nId; iId++){
        bin = binService_->findBin2D(obj, BinService::kTriggerTower, ids[iId]);
        mes_[iME]->setBinContent(bin, _content);
      }
    }
    else{
      bin = binService_->findBin2D(obj, btype_, _id);
      mes_[iME]->setBinContent(bin, _content);
    }
  }

  void
  MESetDet2D::setBinContent(EcalElectronicsId const& _id, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin(binService_->findBin2D(obj, btype_, _id));
    mes_[iME]->setBinContent(bin, _content);
  }

  void
  MESetDet2D::setBinError(DetId const& _id, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin;

    if(isEndcapTTId(_id)){
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      unsigned nId(ids.size());
      for(unsigned iId(0); iId < nId; iId++){
        bin = binService_->findBin2D(obj, BinService::kTriggerTower, ids[iId]);
        mes_[iME]->setBinError(bin, _error);
      }
    }
    else{
      bin = binService_->findBin2D(obj, btype_, _id);
      mes_[iME]->setBinError(bin, _error);
    }
  }

  void
  MESetDet2D::setBinError(EcalElectronicsId const& _id, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin(binService_->findBin2D(obj, btype_, _id));
    mes_[iME]->setBinError(bin, _error);
  }

  void
  MESetDet2D::setBinEntries(DetId const& _id, double _entries)
  {
    if(!active_) return;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin;

    if(isEndcapTTId(_id)){
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      unsigned nId(ids.size());
      for(unsigned iId(0); iId < nId; iId++){
        bin = binService_->findBin2D(obj, BinService::kTriggerTower, ids[iId]);
        mes_[iME]->setBinEntries(bin, _entries);
      }
    }
    else{
      bin = binService_->findBin2D(obj, btype_, _id);
      mes_[iME]->setBinEntries(bin, _entries);
    }
  }

  void
  MESetDet2D::setBinEntries(EcalElectronicsId const& _id, double _entries)
  {
    if(!active_) return;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin(binService_->findBin2D(obj, btype_, _id));
    mes_[iME]->setBinEntries(bin, _entries);
  }

  double
  MESetDet2D::getBinContent(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin;

    if(isEndcapTTId(_id)){
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      bin = binService_->findBin2D(obj, BinService::kTriggerTower, ids[0]);
    }
    else{
      bin = binService_->findBin2D(obj, btype_, _id);
    }
    
    return mes_[iME]->getBinContent(bin);
  }
  
  double
  MESetDet2D::getBinContent(EcalElectronicsId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin(binService_->findBin2D(obj, btype_, _id));
    
    return mes_[iME]->getBinContent(bin);
  }


  double
  MESetDet2D::getBinError(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin;

    if(isEndcapTTId(_id)){
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      bin = binService_->findBin2D(obj, BinService::kTriggerTower, ids[0]);
    }
    else{
      bin = binService_->findBin2D(obj, btype_, _id);
    }
    
    return mes_[iME]->getBinError(bin);
  }
  
  double
  MESetDet2D::getBinError(EcalElectronicsId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin(binService_->findBin2D(obj, btype_, _id));
    
    return mes_[iME]->getBinError(bin);
  }


  double
  MESetDet2D::getBinEntries(DetId const& _id, int) const
  {
    if(!active_) return 0.;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin;

    if(isEndcapTTId(_id)){
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      bin = binService_->findBin2D(obj, BinService::kTriggerTower, ids[0]);
    }
    else{
      bin = binService_->findBin2D(obj, btype_, _id);
    }
    
    return mes_[iME]->getBinEntries(bin);
  }
  
  double
  MESetDet2D::getBinEntries(EcalElectronicsId const& _id, int) const
  {
    if(!active_) return 0.;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    int bin(binService_->findBin2D(obj, btype_, _id));
    
    return mes_[iME]->getBinEntries(bin);
  }

  int
  MESetDet2D::findBin(DetId const& _id) const
  {
    if(!active_) return 0;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    if(isEndcapTTId(_id)){
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      return binService_->findBin2D(obj, BinService::kTriggerTower, ids[0]);
    }
    else
      return binService_->findBin2D(obj, btype_, _id);
  }

  int
  MESetDet2D::findBin(EcalElectronicsId const& _id) const
  {
    if(!active_) return 0;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(otype_, iME));

    return binService_->findBin2D(obj, btype_, _id);
  }

  void
  MESetDet2D::reset(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    unsigned nME(binService_->getNObjects(otype_));

    bool isProfile(kind_ == MonitorElement::DQM_KIND_TPROFILE || kind_ == MonitorElement::DQM_KIND_TPROFILE2D);

    for(unsigned iME(0); iME < nME; iME++) {
      MonitorElement* me(mes_[iME]);

      BinService::ObjectType obj(binService_->getObject(otype_, iME));

      int nbinsX(me->getTH1()->GetNbinsX());
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int ix(1); ix <= nbinsX; ix++){
        if(nbinsY == 1){
          me->setBinContent(ix, _content);
          me->setBinError(ix, _err);
          if(isProfile) me->setBinEntries(ix, _entries);
        }
        else{
          for(int iy(1); iy <= nbinsY; iy++){
            int bin((nbinsX + 2) * iy + ix);
            if(!binService_->isValidIdBin(obj, btype_, iME, bin)) continue;
            me->setBinContent(bin, _content);
            me->setBinError(bin, _err);
            if(isProfile) me->setBinEntries(bin, _entries);
          }
        }
      }
    }
  }

  void
  MESetDet2D::fill_(unsigned _iME, int _bin, double _w)
  {
    if(kind_ == MonitorElement::DQM_KIND_TPROFILE2D){
      MonitorElement* me(mes_.at(_iME));
      if(me->getBinEntries(_bin) < 0.){
        me->setBinContent(_bin, 0.);
        me->setBinEntries(_bin, 0.);
      }
    }

    MESet::fill_(_iME, _bin, _w);
  }

  void
  MESetDet2D::fill_(unsigned _iME, int _bin, double _y, double _w)
  {
    if(kind_ == MonitorElement::DQM_KIND_TPROFILE2D){
      MonitorElement* me(mes_.at(_iME));
      if(me->getBinEntries(_bin) < 0.){
        me->setBinContent(_bin, 0.);
        me->setBinEntries(_bin, 0.);
      }
    }

    MESet::fill_(_iME, _bin, _y, _w);
  }

  void
  MESetDet2D::fill_(unsigned _iME, double _x, double _wy, double _w)
  {
    if(kind_ == MonitorElement::DQM_KIND_TPROFILE2D){
      MonitorElement* me(mes_.at(_iME));
      int bin(me->getTProfile2D()->FindBin(_x, _wy));
      if(me->getBinEntries(bin) < 0.){
        me->setBinContent(bin, 0.);
        me->setBinEntries(bin, 0.);
      }
    }

    MESet::fill_(_iME, _x, _wy, _w);
  }
}
