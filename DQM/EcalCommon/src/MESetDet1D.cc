#include "DQM/EcalCommon/interface/MESetDet1D.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

namespace ecaldqm
{

  MESetDet1D::MESetDet1D(MEData const& _data) :
    MESetEcal(_data, 1)
  {
    switch(data_->kind){
    case MonitorElement::DQM_KIND_TH1F:
    case MonitorElement::DQM_KIND_TPROFILE:
    case MonitorElement::DQM_KIND_TH2F:
    case MonitorElement::DQM_KIND_TPROFILE2D:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }
  }

  MESetDet1D::~MESetDet1D()
  {
  }

  void
  MESetDet1D::fill(DetId const& _id, double _wy/* = 1.*/, double _w/* = 1.*/, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D)
      fill_(iME, xbin, _wy, _w);
    else
      fill_(iME, xbin, _wy);
  }

  void
  MESetDet1D::fill(EcalElectronicsId const& _id, double _wy/* = 1.*/, double _w/* = 1.*/, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D)
      fill_(iME, xbin, _wy, _w);
    else
      fill_(iME, xbin, _wy);
  }

  void
  MESetDet1D::fill(unsigned _dcctccid, double _wy/* = 1.*/, double _w/* = 1.*/, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D)
      fill_(iME, xbin, _wy, _w);
    else
      fill_(iME, xbin, _wy);
  }

  void
  MESetDet1D::setBinContent(DetId const& _id, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int iY(1); iY <= nbinsY; iY++)
        me->setBinContent(xbin, iY, _content);
    }
    else
      me->setBinContent(xbin, _content);
  }

  void
  MESetDet1D::setBinContent(EcalElectronicsId const& _id, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int iY(1); iY <= nbinsY; iY++)
        me->setBinContent(xbin, iY, _content);
    }
    else
      me->setBinContent(xbin, _content);
  }

  void
  MESetDet1D::setBinContent(unsigned _dcctccid, double _content)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int iY(1); iY <= nbinsY; iY++)
        me->setBinContent(xbin, iY, _content);
    }
    else
      me->setBinContent(xbin, _content);
  }

  void
  MESetDet1D::setBinContent(DetId const& _id, int _ybin, double _content)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TH2F && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    me->setBinContent(xbin, _ybin, _content);
  }

  void
  MESetDet1D::setBinContent(EcalElectronicsId const& _id, int _ybin, double _content)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TH2F && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    me->setBinContent(xbin, _ybin, _content);
  }

  void
  MESetDet1D::setBinContent(unsigned _dcctccid, int _ybin, double _content)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TH2F && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));
    me->setBinContent(xbin, _ybin, _content);
  }

  void
  MESetDet1D::setBinError(DetId const& _id, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int iY(1); iY <= nbinsY; iY++)
        me->setBinError(xbin, iY, _error);
    }
    else
      me->setBinError(xbin, _error);
  }

  void
  MESetDet1D::setBinError(EcalElectronicsId const& _id, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int iY(1); iY <= nbinsY; iY++)
        me->setBinError(xbin, iY, _error);
    }
    else
      me->setBinError(xbin, _error);
  }

  void
  MESetDet1D::setBinError(unsigned _dcctccid, double _error)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));

    if(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int iY(1); iY <= nbinsY; iY++)
        me->setBinError(xbin, iY, _error);
    }
    else
      me->setBinError(xbin, _error);
  }

  void
  MESetDet1D::setBinError(DetId const& _id, int _ybin, double _error)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TH2F && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    me->setBinError(xbin, _ybin, _error);
  }

  void
  MESetDet1D::setBinError(EcalElectronicsId const& _id, int _ybin, double _error)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TH2F && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    me->setBinError(xbin, _ybin, _error);
  }

  void
  MESetDet1D::setBinError(unsigned _dcctccid, int _ybin, double _error)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TH2F && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));
    me->setBinError(xbin, _ybin, _error);
  }

  void
  MESetDet1D::setBinEntries(DetId const& _id, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));

    if(data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsX(me->getTH1()->GetNbinsX());
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int iY(1); iY <= nbinsY; iY++)
        me->setBinEntries((nbinsX + 2) * iY + xbin, _entries);
    }
    else
      me->setBinEntries(xbin, _entries);
  }

  void
  MESetDet1D::setBinEntries(EcalElectronicsId const& _id, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));

    if(data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsX(me->getTH1()->GetNbinsX());
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int iY(1); iY <= nbinsY; iY++)
        me->setBinEntries((nbinsX + 2) * iY + xbin, _entries);
    }
    else
      me->setBinEntries(xbin, _entries);
  }

  void
  MESetDet1D::setBinEntries(unsigned _dcctccid, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));

    if(data_->kind == MonitorElement::DQM_KIND_TPROFILE2D){
      int nbinsX(me->getTH1()->GetNbinsX());
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int iY(1); iY <= nbinsY; iY++)
        me->setBinEntries((nbinsX + 2) * iY + xbin, _entries);
    }
    else
      me->setBinEntries(xbin, _entries);
  }

  void
  MESetDet1D::setBinEntries(DetId const& _id, int _ybin, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());
    me->setBinEntries((nbinsX + 2) * _ybin + xbin, _entries);
  }

  void
  MESetDet1D::setBinEntries(EcalElectronicsId const& _id, int _ybin, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());
    me->setBinEntries((nbinsX + 2) * _ybin + xbin, _entries);
  }

  void
  MESetDet1D::setBinEntries(unsigned _dcctccid, int _ybin, double _entries)
  {
    if(!active_) return;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));
    int nbinsX(me->getTH1()->GetNbinsX());
    me->setBinEntries((nbinsX + 2) * _ybin + xbin, _entries);
  }

  double
  MESetDet1D::getBinContent(DetId const& _id, int _ybin/* = 0*/) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());

    return me->getBinContent((nbinsX + 2) * _ybin + xbin);
  }

  double
  MESetDet1D::getBinContent(EcalElectronicsId const& _id, int _ybin/* = 0*/) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());

    return me->getBinContent((nbinsX + 2) * _ybin + xbin);
  }

  double
  MESetDet1D::getBinContent(unsigned _dcctccid, int _ybin/* = 0*/) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));
    int nbinsX(me->getTH1()->GetNbinsX());

    return me->getBinContent((nbinsX + 2) * _ybin + xbin);
  }

  double
  MESetDet1D::getBinError(DetId const& _id, int _ybin/* = 0*/) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());

    return me->getBinError((nbinsX + 2) * _ybin + xbin);
  }

  double
  MESetDet1D::getBinError(EcalElectronicsId const& _id, int _ybin/* = 0*/) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());

    return me->getBinError((nbinsX + 2) * _ybin + xbin);
  }

  double
  MESetDet1D::getBinError(unsigned _dcctccid, int _ybin/* = 0*/) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));
    int nbinsX(me->getTH1()->GetNbinsX());

    return me->getBinError((nbinsX + 2) * _ybin + xbin);
  }

  double
  MESetDet1D::getBinEntries(DetId const& _id, int _ybin/* = 0*/) const
  {
    if(!active_) return 0.;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());

    return me->getBinEntries((nbinsX + 2) * _ybin + xbin);
  }

  double
  MESetDet1D::getBinEntries(EcalElectronicsId const& _id, int _ybin/* = 0*/) const
  {
    if(!active_) return 0.;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());

    return me->getBinEntries((nbinsX + 2) * _ybin + xbin);
  }

  double
  MESetDet1D::getBinEntries(unsigned _dcctccid, int _ybin/* = 0*/) const
  {
    if(!active_) return 0.;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));
    int nbinsX(me->getTH1()->GetNbinsX());

    return me->getBinEntries((nbinsX + 2) * _ybin + xbin);
  }

  int
  MESetDet1D::findBin(DetId const& _id) const
  {
    if(!active_) return -1;
    if(data_->kind == MonitorElement::DQM_KIND_TPROFILE || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D) return -1;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    return binService_->findBin1D(obj, data_->btype, _id);
  }

  int
  MESetDet1D::findBin(EcalElectronicsId const& _id) const
  {
    if(!active_) return -1;
    if(data_->kind == MonitorElement::DQM_KIND_TPROFILE || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D) return -1;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    return binService_->findBin1D(obj, data_->btype, _id);
  }

  int
  MESetDet1D::findBin(unsigned _dcctccid) const
  {
    if(!active_) return -1;
    if(data_->kind == MonitorElement::DQM_KIND_TPROFILE || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D) return -1;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid));
    checkME_(iME);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    return binService_->findBin1D(obj, data_->btype, _dcctccid);
  }

  int
  MESetDet1D::findBin(DetId const& _id, double _y, double) const
  {
    if(!active_) return -1;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return -1;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());
    return xbin + (nbinsX + 2) * me->getTH1()->GetYaxis()->FindBin(_y);
  }

  int
  MESetDet1D::findBin(EcalElectronicsId const& _id, double _y, double) const
  {
    if(!active_) return -1;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return -1;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _id));
    int nbinsX(me->getTH1()->GetNbinsX());
    return xbin + (nbinsX + 2) * me->getTH1()->GetYaxis()->FindBin(_y);
  }

  int
  MESetDet1D::findBin(unsigned _dcctccid, double _y, double) const
  {
    if(!active_) return -1;
    if(data_->kind != MonitorElement::DQM_KIND_TPROFILE && data_->kind != MonitorElement::DQM_KIND_TPROFILE2D) return -1;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid));
    checkME_(iME);

    MonitorElement* me(mes_[iME]);

    BinService::ObjectType obj(binService_->getObject(data_->otype, iME));
    int xbin(binService_->findBin1D(obj, data_->btype, _dcctccid));
    int nbinsX(me->getTH1()->GetNbinsX());
    return xbin + (nbinsX + 2) * me->getTH1()->GetYaxis()->FindBin(_y);
  }

  void
  MESetDet1D::reset(double _content/* = 0.*/, double _err/* = 0.*/, double _entries/* = 0.*/)
  {
    unsigned nME(binService_->getNObjects(data_->otype));

    bool isProfile(data_->kind == MonitorElement::DQM_KIND_TPROFILE || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D);
    bool is2D(data_->kind == MonitorElement::DQM_KIND_TH2F || data_->kind == MonitorElement::DQM_KIND_TPROFILE2D);

    for(unsigned iME(0); iME < nME; iME++) {
      MonitorElement* me(mes_[iME]);

      int nbinsX(me->getTH1()->GetNbinsX());
      int nbinsY(me->getTH1()->GetNbinsY());
      for(int ix(1); ix <= nbinsX; ix++){
        for(int iy(1); iy <= nbinsY; iy++){
          int bin(is2D ? (nbinsX + 2) * iy + ix : ix);
          me->setBinContent(bin, _content);
          me->setBinError(bin, _err);
          if(isProfile) me->setBinEntries(bin, _entries);
        }
      }
    }
  }

}

