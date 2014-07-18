#include "DQM/EcalCommon/interface/MESetNonObject.h"

namespace ecaldqm
{
  MESetNonObject::MESetNonObject(std::string const& _fullPath, binning::ObjectType _otype, binning::BinningType _btype, MonitorElement::Kind _kind, binning::AxisSpecs const* _xaxis/* = 0*/, binning::AxisSpecs const* _yaxis/* = 0*/, binning::AxisSpecs const* _zaxis/* = 0*/) :
    MESet(_fullPath, _otype, _btype, _kind),
    xaxis_(_xaxis ? new binning::AxisSpecs(*_xaxis) : 0),
    yaxis_(_yaxis ? new binning::AxisSpecs(*_yaxis) : 0),
    zaxis_(_zaxis ? new binning::AxisSpecs(*_zaxis) : 0)
  {
  }

  MESetNonObject::MESetNonObject(MESetNonObject const& _orig) :
    MESet(_orig),
    xaxis_(_orig.xaxis_ ? new binning::AxisSpecs(*_orig.xaxis_) : 0),
    yaxis_(_orig.yaxis_ ? new binning::AxisSpecs(*_orig.yaxis_) : 0),
    zaxis_(_orig.zaxis_ ? new binning::AxisSpecs(*_orig.zaxis_) : 0)
  {
  }

  MESetNonObject::~MESetNonObject()
  {
    delete xaxis_;
    delete yaxis_;
    delete zaxis_;
  }

  MESet&
  MESetNonObject::operator=(MESet const& _rhs)
  {
    delete xaxis_;
    delete yaxis_;
    delete zaxis_;
    xaxis_ = 0;
    yaxis_ = 0;
    zaxis_ = 0;

    MESetNonObject const* pRhs(dynamic_cast<MESetNonObject const*>(&_rhs));
    if(pRhs){
      if(pRhs->xaxis_) xaxis_ = new binning::AxisSpecs(*pRhs->xaxis_);
      if(pRhs->yaxis_) yaxis_ = new binning::AxisSpecs(*pRhs->yaxis_);
      if(pRhs->zaxis_) zaxis_ = new binning::AxisSpecs(*pRhs->zaxis_);
    }
    return MESet::operator=(_rhs);
  }

  MESet*
  MESetNonObject::clone(std::string const& _path/* = ""*/) const
  {
    std::string path(path_);
    if(_path != "") path_ = _path;
    MESet* copy(new MESetNonObject(*this));
    path_ = path;
    return copy;
  }

  void
  MESetNonObject::book(DQMStore& _dqmStore)
  {
    doBook_(_dqmStore);
  }

  void
  MESetNonObject::book(DQMStore::IBooker& _ibooker)
  {
    doBook_(_ibooker);
  }

  bool
  MESetNonObject::retrieve(DQMStore const& _store, std::string* _failedPath/* = 0*/) const
  {
    mes_.clear();

    MonitorElement* me(_store.get(path_));
    if(!me){
      if(_failedPath) *_failedPath = path_;
      return false;
    }

    mes_.push_back(me);

    active_ = true;
    return true;
  }

  void
  MESetNonObject::fill(double _x, double _wy/* = 1.*/, double _w/* = 1.*/)
  {
    if(!active_) return;

    if(mes_.size() == 0 || !mes_[0]) return;

    switch(kind_) {
    case MonitorElement::DQM_KIND_REAL :
      mes_[0]->Fill(_x);
      break;
    case MonitorElement::DQM_KIND_TH1F :
    case MonitorElement::DQM_KIND_TPROFILE :
      mes_[0]->Fill(_x, _wy);
      break;
    case MonitorElement::DQM_KIND_TH2F :
    case MonitorElement::DQM_KIND_TPROFILE2D :
      mes_[0]->Fill(_x, _wy, _w);
      break;
    default :
      break;
    }
  }

  void
  MESetNonObject::setBinContent(int _bin, double _content)
  {
    if(!active_) return;
    if(kind_ == MonitorElement::DQM_KIND_REAL) return;

    if(mes_.size() == 0 || !mes_[0]) return;

    mes_[0]->setBinContent(_bin, _content);
  }

  void
  MESetNonObject::setBinError(int _bin, double _error)
  {
    if(!active_) return;
    if(kind_ == MonitorElement::DQM_KIND_REAL) return;

    if(mes_.size() == 0 || !mes_[0]) return;

    mes_[0]->setBinError(_bin, _error);
  }

  void
  MESetNonObject::setBinEntries(int _bin, double _entries)
  {
    if(!active_) return;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return;

    if(mes_.size() == 0 || !mes_[0]) return;

    mes_[0]->setBinEntries(_bin, _entries);
  }

  double
  MESetNonObject::getBinContent(int _bin, int) const
  {
    if(!active_) return 0.;
    if(kind_ == MonitorElement::DQM_KIND_REAL) return 0.;

    if(mes_.size() == 0 || !mes_[0]) return 0.;

    return mes_[0]->getBinContent(_bin);
  }

  double
  MESetNonObject::getBinError(int _bin, int) const
  {
    if(!active_) return 0.;
    if(kind_ == MonitorElement::DQM_KIND_REAL) return 0.;

    if(mes_.size() == 0 || !mes_[0]) return 0.;

    return mes_[0]->getBinError(_bin);
  }

  double
  MESetNonObject::getBinEntries(int _bin, int) const
  {
    if(!active_) return 0.;
    if(kind_ != MonitorElement::DQM_KIND_TPROFILE && kind_ != MonitorElement::DQM_KIND_TPROFILE2D) return 0.;

    if(mes_.size() == 0 || !mes_[0]) return 0.;

    return mes_[0]->getBinEntries(_bin);
  }

  int
  MESetNonObject::findBin(double _x, double _y/* = 0.*/) const
  {
    if(!active_) return 0;

    if(mes_.size() == 0 || !mes_[0]) return 0;

    if(kind_ == MonitorElement::DQM_KIND_TH1F || kind_ == MonitorElement::DQM_KIND_TPROFILE)
      return mes_[0]->getTH1()->FindBin(_x);
    else if(kind_ == MonitorElement::DQM_KIND_TH2F || kind_ == MonitorElement::DQM_KIND_TPROFILE2D)
      return mes_[0]->getTH1()->FindBin(_x, _y);
    else
      return 0;
  }

  bool
  MESetNonObject::isVariableBinning() const
  {
    return (xaxis_ && xaxis_->edges) || (yaxis_ && yaxis_->edges) || (zaxis_ && zaxis_->edges);
  }

  template<class Bookable>
  void
  MESetNonObject::doBook_(Bookable& _booker)
  {
    using namespace std;

    clear();

    if(path_.find('%') != string::npos)
      throw_("book() called with incompletely formed path");

    size_t slashPos(path_.find_last_of('/'));
    string name(path_.substr(slashPos + 1));
    _booker.setCurrentFolder(path_.substr(0, slashPos));

    MonitorElement* me(0);

    switch(kind_) {
    case MonitorElement::DQM_KIND_REAL :
      me = _booker.bookFloat(name);
      break;

    case MonitorElement::DQM_KIND_TH1F :
      {
        if(!xaxis_)
          throw_("No xaxis found for MESetNonObject");

        if(!xaxis_->edges)
          me = _booker.book1D(name, name, xaxis_->nbins, xaxis_->low, xaxis_->high);
        else{
          float* edges(new float[xaxis_->nbins + 1]);
          for(int i(0); i < xaxis_->nbins + 1; i++)
            edges[i] = xaxis_->edges[i];
          me = _booker.book1D(name, name, xaxis_->nbins, edges);
          delete [] edges;
        }
      }
      break;

    case MonitorElement::DQM_KIND_TPROFILE :
      {
        if(!xaxis_)
          throw_("No xaxis found for MESetNonObject");

        double ylow, yhigh;
        if(!yaxis_){
          ylow = -numeric_limits<double>::max();
          yhigh = numeric_limits<double>::max();
        }
        else{
          ylow = yaxis_->low;
          yhigh = yaxis_->high;
        }
        if(xaxis_->edges)
          me = _booker.bookProfile(name, name, xaxis_->nbins, xaxis_->edges, ylow, yhigh, "");
        else
          me = _booker.bookProfile(name, name, xaxis_->nbins, xaxis_->low, xaxis_->high, ylow, yhigh, "");

      }
      break;

    case MonitorElement::DQM_KIND_TH2F :
      {
        if(!xaxis_ || !yaxis_)
          throw_("No x/yaxis found for MESetNonObject");

        if(!xaxis_->edges || !yaxis_->edges)
          me = _booker.book2D(name, name, xaxis_->nbins, xaxis_->low, xaxis_->high, yaxis_->nbins, yaxis_->low, yaxis_->high);
        else{
          float* xedges(new float[xaxis_->nbins + 1]);
          for(int i(0); i < xaxis_->nbins + 1; i++)
            xedges[i] = xaxis_->edges[i];
          float* yedges(new float[yaxis_->nbins + 1]);
          for(int i(0); i < yaxis_->nbins + 1; i++)
            yedges[i] = yaxis_->edges[i];
          me = _booker.book2D(name, name, xaxis_->nbins, xedges, yaxis_->nbins, yedges);
          delete [] xedges;
          delete [] yedges;
        }
      }
      break;

    case MonitorElement::DQM_KIND_TPROFILE2D :
      {
        if(!xaxis_ || !yaxis_)
          throw_("No x/yaxis found for MESetNonObject");
        double high(0.), low(0.);
        if(zaxis_){
          low = zaxis_->low;
          high = zaxis_->high;
        }
        else{
          low = -numeric_limits<double>::max();
          high = numeric_limits<double>::max();
        }

        me = _booker.bookProfile2D(name, name, xaxis_->nbins, xaxis_->low, xaxis_->high, yaxis_->nbins, yaxis_->low, yaxis_->high, low, high, "");
      }
      break;

    default :
      throw_("Unsupported MonitorElement kind");
    }

    if(xaxis_){
      me->setAxisTitle(xaxis_->title, 1);
      if(xaxis_->labels){
        for(int iBin(1); iBin <= xaxis_->nbins; ++iBin)
          me->setBinLabel(iBin, xaxis_->labels[iBin - 1], 1);
      }
    }
    if(yaxis_){
      me->setAxisTitle(yaxis_->title, 2);
      if(yaxis_->labels){
        for(int iBin(1); iBin <= yaxis_->nbins; ++iBin)
          me->setBinLabel(iBin, yaxis_->labels[iBin - 1], 2);
      }
    }
    if(zaxis_){
      me->setAxisTitle(zaxis_->title, 3);
      if(zaxis_->labels){
        for(int iBin(1); iBin <= zaxis_->nbins; ++iBin)
          me->setBinLabel(iBin, zaxis_->labels[iBin - 1], 3);
      }
    }

    if(lumiFlag_) me->setLumiFlag();

    mes_.push_back(me);

    active_ = true;
  }
}
