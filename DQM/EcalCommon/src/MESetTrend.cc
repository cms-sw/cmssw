#include "DQM/EcalCommon/interface/MESetTrend.h"

#include <ctime>

namespace ecaldqm {

  MESetTrend::MESetTrend(std::string const& _fullPath, binning::ObjectType _otype, binning::BinningType _btype, MonitorElement::Kind _kind, binning::AxisSpecs const* _xaxis/* = 0*/, binning::AxisSpecs const* _yaxis/* = 0*/) :
    MESetEcal(_fullPath, _otype, _btype, _kind, 1, _xaxis, _yaxis),
    minutely_(false),
    shiftAxis_(false),
    currentBin_(-1)
  {
    switch(kind_){
    case MonitorElement::DQM_KIND_TH1F:
    case MonitorElement::DQM_KIND_TH2F:
    case MonitorElement::DQM_KIND_TPROFILE:
    case MonitorElement::DQM_KIND_TPROFILE2D:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }
  }

  MESetTrend::MESetTrend(MESetTrend const& _orig) :
    MESetEcal(_orig),
    minutely_(_orig.minutely_),
    shiftAxis_(_orig.shiftAxis_),
    currentBin_(_orig.currentBin_)
  {
  }

  MESet&
  MESetTrend::operator=(MESet const& _rhs)
  {
    MESetTrend const* pRhs(dynamic_cast<MESetTrend const*>(&_rhs));
    if(pRhs){
      minutely_ = pRhs->minutely_;
      shiftAxis_ = pRhs->shiftAxis_;
      currentBin_ = pRhs->currentBin_;
    }

    return MESetEcal::operator=(_rhs);
  }

  MESet*
  MESetTrend::clone(std::string const& _path/* = ""*/) const
  {
    std::string path(path_);
    if(_path != "") path_ = _path;
    MESet* copy(new MESetTrend(*this));
    path_ = path;
    return copy;
  }

  void
  MESetTrend::book(DQMStore& _dqmStore)
  {
    doBook_(_dqmStore);
  }

  void
  MESetTrend::book(DQMStore::IBooker& _ibooker)
  {
    doBook_(_ibooker);
  }

  void
  MESetTrend::fill(DetId const& _id, double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    if(shift_(unsigned(_t)))
      fill_(iME, _t + 0.5, _wy, _w);
  }

  void
  MESetTrend::fill(EcalElectronicsId const& _id, double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    if(shift_(unsigned(_t)))
      fill_(iME, _t + 0.5, _wy, _w);
  }

  void
  MESetTrend::fill(int _dcctccid, double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    if(shift_(unsigned(_t)))
      fill_(iME, _t + 0.5, _wy, _w);
  }

  void
  MESetTrend::fill(double _t, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;
    if(mes_.size() != 1)
      throw_("MESet type incompatible");

    if(shift_(unsigned(_t)))
      fill_(0, _t + 0.5, _wy, _w);
  }

  int
  MESetTrend::findBin(DetId const& _id, double _t, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_t + 0.5, _y);
  }

  int
  MESetTrend::findBin(EcalElectronicsId const& _id, double _t, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_t + 0.5, _y);
  }

  int
  MESetTrend::findBin(int _dcctccid, double _t, double _y/* = 0.*/) const
  {
    if(!active_) return -1;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_t + 0.5, _y);
  }

  int
  MESetTrend::findBin(double _t, double _y/* = 0.*/) const
  {
    if(!active_) return -1;
    if(mes_.size() != 1)
      throw_("MESet type incompatible");

    return mes_[0]->getTH1()->FindBin(_t + 0.5, _y);
  }

  void
  MESetTrend::setCumulative()
  {
    if(kind_ == MonitorElement::DQM_KIND_TPROFILE || kind_ == MonitorElement::DQM_KIND_TPROFILE2D)
      throw_("Cumulative flag set for a profile plot");
    currentBin_ = 1;
  }

  template<class Bookable>
  void
  MESetTrend::doBook_(Bookable& _booker)
  {
    binning::AxisSpecs xaxis;
    if(xaxis_) xaxis = *xaxis_;
    else{
      xaxis.nbins = 200;
      xaxis.low = 0.;
      xaxis.high = 2000.;
    }

    if(minutely_){
      time_t localTime(time(0));
      struct tm timeBuffer;
      gmtime_r(&localTime, &timeBuffer); // gmtime() is not thread safe
      unsigned utcTime(mktime(&timeBuffer));

      xaxis.low = utcTime;
      if(xaxis_) xaxis.high = utcTime + xaxis_->high - xaxis_->low;
      else xaxis.high = xaxis.low + 200 * 60.;
    }

    binning::AxisSpecs const* xaxisTemp(xaxis_);
    xaxis_ = &xaxis;

    MESetEcal::book(_booker);

    xaxis_ = xaxisTemp;

    if(minutely_){
      for(unsigned iME(0); iME < mes_.size(); ++iME)
        mes_[iME]->getTH1()->GetXaxis()->SetTimeDisplay(1);
      setAxisTitle("UTC");
    }
    else
      setAxisTitle("LumiSections");
  }

  bool
  MESetTrend::shift_(unsigned _t)
  {
    TAxis* tAxis(mes_[0]->getTH1()->GetXaxis());
    int nbinsX(tAxis->GetNbins());
    unsigned tLow(tAxis->GetBinLowEdge(1));
    unsigned tHigh(tAxis->GetBinUpEdge(nbinsX));

    if(!shiftAxis_) return _t >= tLow && _t < tHigh;

    int dBin(0);
    int unitsPerBin((tHigh - tLow) / nbinsX);

    if(_t >= tLow && _t < tHigh){
      if(currentBin_ > 0){
        int thisBin(tAxis->FindBin(_t + 0.5));
        if(thisBin < currentBin_) return false;
        else if(thisBin > currentBin_){
          for(unsigned iME(0); iME < mes_.size(); iME++){
            MonitorElement* me(mes_[iME]);
            int nbinsY(me->getTH1()->GetNbinsY());
            for(int iy(1); iy <= nbinsY; ++iy){
              int orig(me->getTH1()->GetBin(currentBin_, iy));
              double currentContent(me->getBinContent(orig));
              double currentError(me->getBinError(orig));
              for(int ix(currentBin_); ix <= thisBin; ++ix){
                int dest(me->getTH1()->GetBin(ix, iy));
                me->setBinContent(dest, currentContent);
                me->setBinError(dest, currentError);
              }
            }
          }
        }
      }

      return true;
    }
    else if(_t >= tHigh){
      dBin = (_t - tHigh) / unitsPerBin + 1;
      if(currentBin_ > 0) currentBin_ = nbinsX;
    }
    else if(_t < tLow){
      if(currentBin_ > 0) return false; // no going back in time in case of cumulative history

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
	    for(int iy(1); iy <= me->getNbinsY(); iy++){
	      if(me->getBinContent(me->getTH1()->GetBin(iMax, iy)) != 0){
                filled = true;
                break;
              }
            }
	    break;
	  case MonitorElement::DQM_KIND_TPROFILE2D:
	    for(int iy(1); iy <= me->getNbinsY(); iy++){
	      if(me->getBinEntries(me->getTH1()->GetBin(iMax, iy)) != 0){
                filled = true;
                break;
              }
            }
	    break;
	  default:
            break;
	  }
	}

	if(iMax > maxBin) maxBin = iMax;
      }

      if(_t < tLow - (nbinsX - maxBin) * unitsPerBin) return false;

      dBin = (_t - tLow) / unitsPerBin - 1;
    }

    int start(dBin > 0 ? dBin + 1 : nbinsX + dBin);
    int end(dBin > 0 ? nbinsX + 1 : 0);
    int step(dBin > 0 ? 1 : -1);

    tLow += dBin * unitsPerBin;
    tHigh += dBin * unitsPerBin;

    for(unsigned iME(0); iME < mes_.size(); iME++){
      MonitorElement* me(mes_[iME]);

      me->getTH1()->GetXaxis()->SetLimits(tLow, tHigh);

      if((end - start) / step < 0){
        me->Reset();
        continue;
      }

      me->setEntries(0.);
      double entries(0.);

      switch(kind_){
      case MonitorElement::DQM_KIND_TH1F:
        {
          int ix(start);
          for(; ix != end; ix += step){
            double binContent(me->getBinContent(ix));
            entries += binContent;
            me->setBinContent(ix - dBin, binContent);
            me->setBinError(ix - dBin, me->getBinError(ix));
          }
          ix = end - dBin - 1 * step;
          double lastContent(currentBin_ > 0 ? me->getBinContent(ix) : 0.);
          double lastError(currentBin_ > 0 ? me->getBinContent(ix) : 0.);
          for(ix += step; ix != end; ix += step){
            me->setBinContent(ix, lastContent);
            me->setBinError(ix, lastError);
          }
        }
        break;
      case MonitorElement::DQM_KIND_TPROFILE:
        {
          int ix(start);
          for(; ix != end; ix += step){
            double binEntries(me->getBinEntries(ix));
            double binContent(me->getBinContent(ix));
            entries += binEntries;
            me->setBinEntries(ix - dBin, binEntries);
            me->setBinContent(ix - dBin, binContent * binEntries);
            if(binEntries > 0){
              double rms(me->getBinError(ix) * std::sqrt(binEntries));
              double sumw2((rms * rms + binContent * binContent) * binEntries);
              me->setBinError(ix - dBin, std::sqrt(sumw2));
            }
            else
              me->setBinError(ix - dBin, 0.);
          }
          ix = end - dBin;
          for(; ix != end; ix += step){
            me->setBinEntries(ix, 0.);
            me->setBinContent(ix, 0.);
            me->setBinError(ix, 0.);
          }
        }
        break;
      case MonitorElement::DQM_KIND_TH2F:
        {
          int ix(start);
          int nbinsY(me->getNbinsY());
          for(; ix != end; ix += step){
            for(int iy(1); iy <= nbinsY; iy++){
              int orig(me->getTH1()->GetBin(ix, iy));
              int dest(me->getTH1()->GetBin(ix - dBin, iy));
              double binContent(me->getBinContent(orig));
              entries += binContent;
              me->setBinContent(dest, binContent);
              me->setBinError(dest, me->getBinError(orig));

              me->setBinContent(orig, 0.);
              me->setBinError(orig, 0.);
            }
          }
          ix = end - dBin - 1 * step;
          std::vector<double> lastContent;
          std::vector<double> lastError;
          for(int iy(1); iy <= nbinsY; iy++){
            lastContent.push_back(currentBin_ > 0 ? me->getBinContent(ix, iy) : 0.);
            lastError.push_back(currentBin_ > 0 ? me->getBinError(ix, iy) : 0.);
          }
          for(ix += step; ix != end; ix += step){
            for(int iy(1); iy <= nbinsY; iy++){
              int bin(me->getTH1()->GetBin(ix, iy));
              me->setBinContent(bin, lastContent[iy - 1]);
              me->setBinError(bin, lastError[iy - 1]);
            }
          }
        }
        break;
      case MonitorElement::DQM_KIND_TPROFILE2D:
        {
          int ix(start);
          int nbinsY(me->getNbinsY());
          for(; ix != end; ix += step){
            for(int iy(1); iy <= nbinsY; iy++){
              int orig(me->getTH1()->GetBin(ix, iy));
              int dest(me->getTH1()->GetBin(ix - dBin, iy));
              double binEntries(me->getBinEntries(orig));
              double binContent(me->getBinContent(orig));
              entries += binEntries;
              me->setBinEntries(dest, binEntries);
              me->setBinContent(dest, binContent * binEntries);
              if(binEntries > 0){
                double rms(me->getBinError(orig) * std::sqrt(binEntries));
                double sumw2((rms * rms + binContent * binContent) * binEntries);
                me->setBinError(dest, std::sqrt(sumw2));
              }
              else
                me->setBinError(dest, 0.);
            }
          }
          ix = end - dBin;
          for(; ix != end; ix += step){
            for(int iy(1); iy <= nbinsY; iy++){
              int bin(me->getTH1()->GetBin(ix, iy));
              me->setBinEntries(bin, 0.);
              me->setBinContent(bin, 0.);
              me->setBinError(bin, 0.);
            }
          }
        }
        break;
      default:
        break;
      }

      me->setEntries(entries);
    }

    return true;
  }

}
