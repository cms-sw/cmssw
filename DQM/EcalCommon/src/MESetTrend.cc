#include "DQM/EcalCommon/interface/MESetTrend.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace ecaldqm {

  MESetTrend::MESetTrend(std::string const& _fullpath, MEData const& _data, bool _readOnly/* = false*/) :
    MESetEcal(_fullpath, _data, 1, _readOnly),
    t0_(0),
    minutely_(false),
    tLow_(0)
  {
    if(!_data.xaxis || _data.xaxis->edges)
      throw cms::Exception("InvalidConfiguration") << "MESetTrend";
  }

  MESetTrend::~MESetTrend()
  {
  }

  void
  MESetTrend::book()
  {
    int conversion(minutely_ ? 60 : 3600);
    time_t width((data_->xaxis->high - data_->xaxis->low) * conversion);

    tLow_ = t0_;

    MEData const* temp(data_);
    BinService::AxisSpecs xaxis(*temp->xaxis);
    xaxis.low = tLow_;
    xaxis.high = tLow_ + width;

    data_ = new MEData(temp->pathName, temp->otype, temp->btype, temp->kind, &xaxis, temp->yaxis, temp->zaxis);

    MESetEcal::book();

    delete data_;
    data_ = temp;

    // if yaxis was variable bin size, xaxis will be booked as variable too

    for(unsigned iME(0); iME < mes_.size(); iME++){
      TAxis* axis(mes_[iME]->getTH1()->GetXaxis());
      if(axis->IsVariableBinSize())
	axis->Set(data_->xaxis->nbins, data_->xaxis->low, data_->xaxis->high);
    }
  }

  void
  MESetTrend::fill(DetId const& _id, double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned offset(binService_->findOffset(data_->otype, _id));

    if(shift_(time_t(_t)))
      MESet::fill_(offset, _t, _wy, _w);
  }

  void
  MESetTrend::fill(unsigned _dcctccid, double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned offset(binService_->findOffset(data_->otype, data_->btype, _dcctccid));

    if(shift_(time_t(_t)))
      MESet::fill_(offset, _t, _wy, _w);
  }

  void
  MESetTrend::fill(double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;
    if(mes_.size() != 1)
      throw cms::Exception("InvalidCall") << "MESet type incompatible" << std::endl;

    if(shift_(time_t(_t)))
      MESet::fill_(0, _t, _wy, _w);
  }

  bool
  MESetTrend::shift_(time_t _t)
  {
    int conversion(minutely_ ? 60 : 3600);
    time_t width((data_->xaxis->high - data_->xaxis->low) * conversion);

    time_t tHigh(tLow_ + width);
    int nbinsX(data_->xaxis->nbins);
    MonitorElement::Kind kind(data_->kind);

    int dtPerBin(width / nbinsX);
    int dBin(0);

    if(_t >= tLow_ && _t < tHigh)
      return true;
    else if(_t >= tHigh)
      dBin = (_t - tHigh) / dtPerBin + 1;
    else if(_t < tLow_){
      int maxBin(0);

      for(unsigned iME(0); iME < mes_.size(); iME++){
	MonitorElement* me(mes_[iME]);

	bool filled(false);
	int iMax(nbinsX + 1);
	while(--iMax > 0 && !filled){
	  switch(kind){
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
	    return false;
	  }
	}

	if(iMax > maxBin) maxBin = iMax;
      }

      if(_t < tLow_ - (nbinsX - maxBin) * dtPerBin) return false;

      dBin = (_t - dtPerBin - tLow_) / dtPerBin;
    }

    int start(dBin > 0 ? dBin + 1 : nbinsX + dBin);
    int end(dBin > 0 ? nbinsX + 1 : 0);
    int step(dBin > 0 ? 1 : -1);

    tLow_ += dBin * dtPerBin;
    tHigh += dBin * dtPerBin;

    for(unsigned iME(0); iME < mes_.size(); iME++){
      MonitorElement* me(mes_[iME]);

      me->setEntries(0.);

      double entries(0.);

      for(int ix(start); (dBin > 0 ? (ix < end) : (ix > end)); ix += step){
	switch(kind){
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
