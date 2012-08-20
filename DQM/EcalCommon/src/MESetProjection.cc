#include "DQM/EcalCommon/interface/MESetProjection.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm
{
  MESetProjection::MESetProjection(MEData const& _data) :
    MESetEcal(_data, 1)
  {
    switch(data_->kind){
    case MonitorElement::DQM_KIND_TH1F:
    case MonitorElement::DQM_KIND_TPROFILE:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }

    switch(data_->btype){
    case BinService::kProjEta:
    case BinService::kProjPhi:
      break;
    default:
      throw_("Unsupported binning");
    }
  }

  MESetProjection::~MESetProjection()
  {
  }

  void
  MESetProjection::fill(DetId const& _id, double _w/* = 1.*/, double, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    int subdet(_id.subdetId());

    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(data_->btype == BinService::kProjEta)
        fill_(iME, ebid.approxEta(), _w, 0.);
      else if(data_->btype == BinService::kProjPhi)
        fill_(iME, phi(ebid), _w, 0.);
    }
    else if(subdet == EcalEndcap){
      if(data_->btype == BinService::kProjEta)
        fill_(iME, eta(_id), _w, 0.);
      if(data_->btype == BinService::kProjPhi){
        EEDetId eeid(_id);
        fill_(iME, phi(eeid), _w, 0.);
      }
    }
    else if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      unsigned nIds(ids.size());
      if(data_->btype == BinService::kProjEta){
        for(unsigned iId(0); iId < nIds; iId++)
          fill_(iME, eta(ids[iId]), _w / nIds, 0.);
      }
      else if(data_->btype == BinService::kProjPhi){
        for(unsigned iId(0); iId < nIds; iId++){
          EEDetId eeid(ids[iId]);
          fill_(iME, phi(eeid), _w / nIds, 0.);
        }
      }
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(data_->btype == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          fill_(iME, (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta, _w, 0.);
        else if(ieta > -18 && ieta < 0)
          fill_(iME, (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta, _w, 0.);
      }
      else if(data_->btype == BinService::kProjPhi)
        fill_(iME, phi(ttid), _w, 0.);
    }
  }

  void
  MESetProjection::fill(double _x, double _w/* = 1.*/, double)
  {
    if(!active_) return;
    if(data_->btype != BinService::kProjEta) return;

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

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      unsigned nIds(ids.size());
      std::set<int> bins;
      if(data_->btype == BinService::kProjEta){
        for(unsigned iId(0); iId < nIds; iId++){
          int bin(me->getTH1()->FindBin(eta(ids[iId])));
          if(bins.find(bin) != bins.end()) continue;
          me->setBinContent(bin, _content);
        }
      }
      else if(data_->btype == BinService::kProjPhi){
        for(unsigned iId(0); iId < nIds; iId++){
          EEDetId eeid(ids[iId]);
          int bin(me->getTH1()->FindBin(phi(eeid)));
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
      if(data_->btype == BinService::kProjEta)
        x = ebid.approxEta();
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(data_->btype == BinService::kProjEta)
        x = eta(_id);
      else if(data_->btype == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(data_->btype == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    me->setBinContent(bin, _content);
  }

  void
  MESetProjection::setBinError(DetId const& _id, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      unsigned nIds(ids.size());
      std::set<int> bins;
      if(data_->btype == BinService::kProjEta){
        for(unsigned iId(0); iId < nIds; iId++){
          int bin(me->getTH1()->FindBin(eta(ids[iId])));
          if(bins.find(bin) != bins.end()) continue;
          me->setBinError(bin, _error);
        }
      }
      else if(data_->btype == BinService::kProjPhi){
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
      if(data_->btype == BinService::kProjEta)
        x = ebid.approxEta();
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(data_->btype == BinService::kProjEta)
        x = eta(_id);
      else if(data_->btype == BinService::kProjPhi)
        x = phi(EEDetId(_id));
     }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(data_->btype == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    me->setBinError(bin, _error);
  }

  void
  MESetProjection::setBinEntries(DetId const& _id, double _entries)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      unsigned nIds(ids.size());
      std::set<int> bins;
      if(data_->btype == BinService::kProjEta){
        for(unsigned iId(0); iId < nIds; iId++){
          int bin(me->getTH1()->FindBin(eta(ids[iId])));
          if(bins.find(bin) != bins.end()) continue;
          me->setBinEntries(bin, _entries);
        }
      }
      else if(data_->btype == BinService::kProjPhi){
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
      if(data_->btype == BinService::kProjEta)
        x = ebid.approxEta();
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(data_->btype == BinService::kProjEta)
        x = eta(_id);
      else if(data_->btype == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(data_->btype == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    me->setBinEntries(bin, _entries);
  }

  double
  MESetProjection::getBinContent(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      if(data_->btype == BinService::kProjEta){
        int bin(me->getTH1()->FindBin(eta(ids[0])));
        return me->getBinContent(bin);
      }
      else if(data_->btype == BinService::kProjPhi){
        int bin(me->getTH1()->FindBin(phi(EEDetId(ids[0]))));
        return me->getBinContent(bin);
      }
      return 0.;
    }

    double x(0.);
    int subdet(_id.subdetId());
    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(data_->btype == BinService::kProjEta)
        x = ebid.approxEta();
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(data_->btype == BinService::kProjEta)
        x = eta(_id);
      else if(data_->btype == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(data_->btype == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    return me->getBinContent(bin);
  }

  double
  MESetProjection::getBinError(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      if(data_->btype == BinService::kProjEta){
        int bin(me->getTH1()->FindBin(eta(ids[0])));
        return me->getBinError(bin);
      }
      else if(data_->btype == BinService::kProjPhi){
        int bin(me->getTH1()->FindBin(phi(EEDetId(ids[0]))));
        return me->getBinError(bin);
      }
      return 0.;
    }

    double x(0.);
    int subdet(_id.subdetId());
    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(data_->btype == BinService::kProjEta)
        x = ebid.approxEta();
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(data_->btype == BinService::kProjEta)
        x = eta(_id);
      else if(data_->btype == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(data_->btype == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    return me->getBinError(bin);
  }

  double
  MESetProjection::getBinEntries(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    if(isEndcapTTId(_id)){
      EcalTrigTowerDetId ttid(_id);
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(ttid));
      if(data_->btype == BinService::kProjEta){
        int bin(me->getTH1()->FindBin(eta(ids[0])));
        return me->getBinEntries(bin);
      }
      else if(data_->btype == BinService::kProjPhi){
        int bin(me->getTH1()->FindBin(phi(EEDetId(ids[0]))));
        return me->getBinEntries(bin);
      }
      return 0.;
    }

    double x(0.);
    int subdet(_id.subdetId());
    if(subdet == EcalBarrel){
      EBDetId ebid(_id);
      if(data_->btype == BinService::kProjEta)
        x = ebid.approxEta();
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ebid);
    }
    else if(subdet == EcalEndcap){
      if(data_->btype == BinService::kProjEta)
        x = eta(_id);
      else if(data_->btype == BinService::kProjPhi)
        x = phi(EEDetId(_id));
    }
    else if(subdet == EcalTriggerTower){
      EcalTrigTowerDetId ttid(_id);
      if(data_->btype == BinService::kProjEta){
        int ieta(ttid.ieta());
        if(ieta < 18 && ieta > 0)
          x = (ieta * 5 - 2.5) * EBDetId::crystalUnitToEta;
        else if(ieta > -18 && ieta < 0)
          x = (ieta * 5 + 2.5) * EBDetId::crystalUnitToEta;
      }
      else if(data_->btype == BinService::kProjPhi)
        x = phi(ttid);
    }

    int bin(me->getTH1()->FindBin(x));
    return me->getBinEntries(bin);
  }

}
