#include "DQM/EcalCommon/interface/MESetProjection.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm
{
  MESetProjection::MESetProjection(std::string const& _fullPath, BinService::ObjectType _otype, BinService::BinningType _btype, MonitorElement::Kind _kind, BinService::AxisSpecs const* _yaxis/* = 0*/) :
    MESetEcal(_fullPath, _otype, _btype, _kind, 1, 0, _yaxis)
  {
    switch(kind_){
    case MonitorElement::DQM_KIND_TH1F:
    case MonitorElement::DQM_KIND_TPROFILE:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }

    switch(btype_){
    case BinService::kProjEta:
    case BinService::kProjPhi:
      break;
    default:
      throw_("Unsupported binning");
    }
  }

  MESetProjection::MESetProjection(MESetProjection const& _orig) :
    MESetEcal(_orig)
  {
  }

  MESetProjection::~MESetProjection()
  {
  }

  MESet*
  MESetProjection::clone() const
  {
    return new MESetProjection(*this);
  }

  void
  MESetProjection::fill(DetId const& _id, double _w/* = 1.*/, double, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    int subdet(_id.subdetId());

    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(btype_ == BinService::kProjEta)
        fill_(iME, eta(ebid), _w, 0.);
      else if(btype_ == BinService::kProjPhi)
        fill_(iME, phi(ebid), _w, 0.);
    }
    else if(subdet == EcalEndcap){
      EEDetId eeid(_id);
      if(btype_ == BinService::kProjEta)
        fill_(iME, eta(eeid), _w, 0.);
      if(btype_ == BinService::kProjPhi){
        fill_(iME, phi(eeid), _w, 0.);
      }
    }
    else if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      unsigned nIds(ids.size());
      if(btype_ == BinService::kProjEta){
        for(unsigned iId(0); iId < nIds; iId++)
          fill_(iME, eta(EEDetId(ids[iId])), _w / nIds, 0.);
      }
      else if(btype_ == BinService::kProjPhi){
        for(unsigned iId(0); iId < nIds; iId++)
          fill_(iME, phi(EEDetId(ids[iId])), _w / nIds, 0.);
      }
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(btype_ == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          fill_(iME, (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta, _w, 0.);
        else if(ieta > -18 && ieta < 0)
          fill_(iME, (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta, _w, 0.);
      }
      else if(btype_ == BinService::kProjPhi)
        fill_(iME, phi(ttid), _w, 0.);
    }
  }

  void
  MESetProjection::fill(double _x, double _w/* = 1.*/, double)
  {
    if(!active_) return;
    if(btype_ != BinService::kProjEta) return;

    unsigned iME;
    if(_x < -etaBound) iME = 0;
    else if(_x < etaBound) iME = 1;
    else iME = 2;

    checkME_(iME);

    mes_[iME]->Fill(_x, _w);
  }

  void
  MESetProjection::setBinContent(DetId const& _id, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      unsigned nIds(ids.size());
      std::set<int> bins;
      if(btype_ == BinService::kProjEta){
        for(unsigned iId(0); iId < nIds; iId++){
          int bin(me->getTH1()->FindBin(eta(EEDetId(ids[iId]))));
          if(bins.find(bin) != bins.end()) continue;
          me->setBinContent(bin, _content);
        }
      }
      else if(btype_ == BinService::kProjPhi){
        for(unsigned iId(0); iId < nIds; iId++){
          int bin(me->getTH1()->FindBin(phi(EEDetId(ids[iId]))));
          if(bins.find(bin) != bins.end()) continue;
          me->setBinContent(bin, _content);
        }
      }
      return;
    }

    double x(0.);
    int subdet(_id.subdetId());
    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(btype_ == BinService::kProjEta)
        x = eta(ebid);
      else if(btype_ == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(btype_ == BinService::kProjEta)
        x = eta(EEDetId(_id));
      else if(btype_ == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(btype_ == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(btype_ == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    me->setBinContent(bin, _content);
  }

  void
  MESetProjection::setBinError(DetId const& _id, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      unsigned nIds(ids.size());
      std::set<int> bins;
      if(btype_ == BinService::kProjEta){
        for(unsigned iId(0); iId < nIds; iId++){
          int bin(me->getTH1()->FindBin(eta(EEDetId(ids[iId]))));
          if(bins.find(bin) != bins.end()) continue;
          me->setBinError(bin, _error);
        }
      }
      else if(btype_ == BinService::kProjPhi){
        for(unsigned iId(0); iId < nIds; iId++){
          int bin(me->getTH1()->FindBin(phi(EEDetId(ids[iId]))));
          if(bins.find(bin) != bins.end()) continue;
          me->setBinError(bin, _error);
        }
      }
      return;
    }

    double x(0.);
    int subdet(_id.subdetId());
    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(btype_ == BinService::kProjEta)
        x = eta(ebid);
      else if(btype_ == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(btype_ == BinService::kProjEta)
        x = eta(EEDetId(_id));
      else if(btype_ == BinService::kProjPhi)
        x = phi(EEDetId(_id));
     }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(btype_ == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(btype_ == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    me->setBinError(bin, _error);
  }

  void
  MESetProjection::setBinEntries(DetId const& _id, double _entries)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      unsigned nIds(ids.size());
      std::set<int> bins;
      if(btype_ == BinService::kProjEta){
        for(unsigned iId(0); iId < nIds; iId++){
          int bin(me->getTH1()->FindBin(eta(EEDetId(ids[iId]))));
          if(bins.find(bin) != bins.end()) continue;
          me->setBinEntries(bin, _entries);
        }
      }
      else if(btype_ == BinService::kProjPhi){
        for(unsigned iId(0); iId < nIds; iId++){
          int bin(me->getTH1()->FindBin(phi(EEDetId(ids[iId]))));
          if(bins.find(bin) != bins.end()) continue;
          me->setBinEntries(bin, _entries);
        }
      }
      return;
    }

    double x(0.);
    int subdet(_id.subdetId());
    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(btype_ == BinService::kProjEta)
        x = eta(ebid);
      else if(btype_ == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(btype_ == BinService::kProjEta)
        x = eta(EEDetId(_id));
      else if(btype_ == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(btype_ == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(btype_ == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    me->setBinEntries(bin, _entries);
  }

  double
  MESetProjection::getBinContent(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      if(btype_ == BinService::kProjEta){
        int bin(me->getTH1()->FindBin(eta(EEDetId(ids[0]))));
        return me->getBinContent(bin);
      }
      else if(btype_ == BinService::kProjPhi){
        int bin(me->getTH1()->FindBin(phi(EEDetId(ids[0]))));
        return me->getBinContent(bin);
      }
      return 0.;
    }

    double x(0.);
    int subdet(_id.subdetId());
    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(btype_ == BinService::kProjEta)
        x = eta(ebid);
      else if(btype_ == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(btype_ == BinService::kProjEta)
        x = eta(EEDetId(_id));
      else if(btype_ == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(btype_ == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(btype_ == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    return me->getBinContent(bin);
  }

  double
  MESetProjection::getBinError(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      if(btype_ == BinService::kProjEta){
        int bin(me->getTH1()->FindBin(eta(EEDetId(ids[0]))));
        return me->getBinError(bin);
      }
      else if(btype_ == BinService::kProjPhi){
        int bin(me->getTH1()->FindBin(phi(EEDetId(ids[0]))));
        return me->getBinError(bin);
      }
      return 0.;
    }

    double x(0.);
    int subdet(_id.subdetId());
    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(btype_ == BinService::kProjEta)
        x = eta(ebid);
      else if(btype_ == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(btype_ == BinService::kProjEta)
        x = eta(EEDetId(_id));
      else if(btype_ == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(btype_ == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(btype_ == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    return me->getBinError(bin);
  }

  double
  MESetProjection::getBinEntries(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      if(btype_ == BinService::kProjEta){
        int bin(me->getTH1()->FindBin(eta(EEDetId(ids[0]))));
        return me->getBinEntries(bin);
      }
      else if(btype_ == BinService::kProjPhi){
        int bin(me->getTH1()->FindBin(phi(EEDetId(ids[0]))));
        return me->getBinEntries(bin);
      }
      return 0.;
    }

    double x(0.);
    int subdet(_id.subdetId());
    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(btype_ == BinService::kProjEta)
        x = eta(ebid);
      else if(btype_ == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(btype_ == BinService::kProjEta)
        x = eta(EEDetId(_id));
      else if(btype_ == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(btype_ == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(btype_ == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    return me->getBinEntries(bin);
  }

}
