#include "DQM/EcalCommon/interface/MESetTrend.h"

namespace ecaldqm {

  MESetTrend::MESetTrend(std::string const& _fullPath, BinService::ObjectType _otype, BinService::BinningType _btype, MonitorElement::Kind _kind, BinService::AxisSpecs const* _yaxis/* = 0*/) :
    MESetEcal(_fullPath, _otype, _btype, _kind, 1, 0, _yaxis),
    t0_(0),
    minutely_(false),
    tLow_(0)
  {
    switch(kind_){
    case MonitorElement::DQM_KIND_TH1F:
    case MonitorElement::DQM_KIND_TPROFILE:
    case MonitorElement::DQM_KIND_TH2F:
    case MonitorElement::DQM_KIND_TPROFILE2D:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }
  }

  MESetTrend::MESetTrend(MESetTrend const& _orig) :
    MESetEcal(_orig),
    t0_(_orig.t0_),
    minutely_(_orig.minutely_),
    tLow_(_orig.tLow_)
  {
  }

  MESetTrend::~MESetTrend()
  {
  }

  MESet&
  MESetTrend::operator=(MESet const& _rhs)
  {
    MESetTrend const* pRhs(dynamic_cast<MESetTrend const*>(&_rhs));
    if(pRhs){
      t0_ = pRhs->t0_;
      minutely_ = pRhs->minutely_;
      tLow_ = pRhs->tLow_;
    }

    return MESetEcal::operator=(_rhs);
  }

  MESet*
  MESetTrend::clone() const
  {
    return new MESetTrend(*this);
  }

  void
  MESetTrend::book()
  {
    if(t0_ == 0) return;

    int conversion(minutely_ ? 60 : 600);

    BinService::AxisSpecs xaxis;
    xaxis.nbins = 60;
    xaxis.low = t0_;
    xaxis.high = t0_ + xaxis.nbins * conversion;

    xaxis_ = new BinService::AxisSpecs(xaxis);

    MESetEcal::book();

    tLow_ = t0_;
  }

  void
  MESetTrend::fill(DetId const& _id, double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    if(shift_(time_t(_t)))
      fill_(iME, _t, _wy, _w);
  }

  void
  MESetTrend::fill(EcalElectronicsId const& _id, double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    if(shift_(time_t(_t)))
      fill_(iME, _t, _wy, _w);
  }

  void
  MESetTrend::fill(unsigned _dcctccid, double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    if(shift_(time_t(_t)))
      fill_(iME, _t, _wy, _w);
  }

  void
  MESetTrend::fill(double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;
    if(mes_.size() != 1)
      throw_("MESet type incompatible");

    if(shift_(time_t(_t)))
      fill_(0, _t, _wy, _w);
  }

  int
  MESetTrend::findBin(DetId const& _id, double _t, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_t, _y);
  }

  int
  MESetTrend::findBin(EcalElectronicsId const& _id, double _t, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binService_->findPlot(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_t, _y);
  }

  int
  MESetTrend::findBin(unsigned _dcctccid, double _t, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binService_->findPlot(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_t, _y);
  }

  int
  MESetTrend::findBin(double _t, double _y/* = 0.*/) const
  {
    if(!active_) return -1;
    if(mes_.size() != 1)
      throw_("MESet type incompatible");

    return mes_[0]->getTH1()->FindBin(_t, _y);
  }

  bool
  MESetTrend::shift_(time_t _t)
  {
    int conversion(minutely_ ? 60 : 600);
    int nbinsX(xaxis_->nbins);
    time_t tHigh(tLow_ + nbinsX * conversion);

    if(kind_ != MonitorElement::DQM_KIND_TH1F && kind_ != MonitorElement::DQM_KIND_TPROFILE && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return false;

    int dBin(0);

    if(_t >= tLow_ && _t < tHigh) return true;
    else if(_t >= tHigh) dBin = (_t - tHigh) / conversion + 1;
    else if(_t < tLow_){
      int maxBin(0);

      for(unsigned iME(0); iME < mes_.size(); iME++){
	MonitorElement* me(mes_[iME]);

	bool filled(false);
	int iMax(nbinsX + 1);
	while(--iMax > 0 && !filled){
	  switch(kind_){
	  case  MonitorElement::DQM_KIND_TH1F:
	    if(me->getBinContent(iMax) != 0) filled = true;
	    break;
	  case MonitorElement::DQM_KIND_TPROFILE:
	    if(me->getBinEntries(iMax) != 0) filled = true;
	    break;
	  case MonitorElement::DQM_KIND_TH2F:
	    for(int iy(1); iy <= me->getNbinsY(); iy++)
	      if(me->getBinContent(iMax, iy) != 0) filled = true;
	    break;
	  default:
            break;
	  }
	}

	if(iMax > maxBin) maxBin = iMax;
      }

      if(_t < tLow_ - (nbinsX - maxBin) * conversion) return false;

      dBin = (_t - tLow_) / conversion - 1;
    }

    int start(dBin > 0 ? dBin + 1 : nbinsX + dBin);
    int end(dBin > 0 ? nbinsX + 1 : 0);
    int step(dBin > 0 ? 1 : -1);

    tLow_ += dBin * conversion;
    tHigh += dBin * conversion;

    for(unsigned iME(0); iME < mes_.size(); iME++){
      MonitorElement* me(mes_[iME]);

      me->setEntries(0.);

      double entries(0.);

      for(int ix(start); (dBin > 0 ? (ix < end) : (ix > end)); ix += step){
	switch(kind_){
	case  MonitorElement::DQM_KIND_TH1F:
	  entries += me->getBinContent(ix);
  	  me->setBinContent(ix - dBin, me->getBinContent(ix));
	  me->setBinError(ix - dBin, me->getBinError(ix));
	  me->setBinContent(ix, 0.);
	  me->setBinError(ix, 0.);
	  break;
	case MonitorElement::DQM_KIND_TPROFILE:
	  entries += me->getBinEntries(ix);
	  me->setBinEntries(ix - dBin, me->getBinEntries(ix));
	  me->setBinContent(ix - dBin, me->getBinContent(ix) * me->getBinEntries(ix));
	  if(me->getBinEntries(ix) > 0){
	    double rms(me->getBinError(ix) * std::sqrt(me->getBinEntries(ix)));
	    double sumw2((rms * rms + me->getBinContent(ix) * me->getBinContent(ix)) * me->getBinEntries(ix));
	    me->setBinError(ix - dBin, std::sqrt(sumw2));
	  }
	  me->setBinEntries(ix, 0.);
	  me->setBinContent(ix, 0.);
	  me->setBinError(ix, 0.);
	  break;
	case MonitorElement::DQM_KIND_TH2F:
	  for(int iy(1); iy <= me->getNbinsY(); iy++){
	    int orig(me->getTH1()->GetBin(ix, iy));
	    int dest(me->getTH1()->GetBin(ix - dBin, iy));
	    entries += me->getBinContent(orig);
	    me->setBinContent(dest, me->getBinContent(orig));
	    me->setBinError(dest, me->getBinError(orig));
	    me->setBinContent(orig, 0.);
	    me->setBinError(orig, 0.);
	  }
	  break;
	default:
	  break;
	}
      }

      me->setEntries(entries);
      me->getTH1()->GetXaxis()->SetLimits(tLow_, tHigh);
    }

    return true;
  }

}
