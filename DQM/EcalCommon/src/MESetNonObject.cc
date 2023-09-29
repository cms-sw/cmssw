#include "DQM/EcalCommon/interface/MESetNonObject.h"

namespace ecaldqm {
  MESetNonObject::MESetNonObject(std::string const &_fullPath,
                                 binning::ObjectType _otype,
                                 binning::BinningType _btype,
                                 MonitorElement::Kind _kind,
                                 binning::AxisSpecs const *_xaxis /* = 0*/,
                                 binning::AxisSpecs const *_yaxis /* = 0*/,
                                 binning::AxisSpecs const *_zaxis /* = 0*/)
      : MESet(_fullPath, _otype, _btype, _kind),
        xaxis_(_xaxis ? new binning::AxisSpecs(*_xaxis) : nullptr),
        yaxis_(_yaxis ? new binning::AxisSpecs(*_yaxis) : nullptr),
        zaxis_(_zaxis ? new binning::AxisSpecs(*_zaxis) : nullptr) {}

  MESetNonObject::MESetNonObject(MESetNonObject const &_orig)
      : MESet(_orig),
        xaxis_(_orig.xaxis_ ? new binning::AxisSpecs(*_orig.xaxis_) : nullptr),
        yaxis_(_orig.yaxis_ ? new binning::AxisSpecs(*_orig.yaxis_) : nullptr),
        zaxis_(_orig.zaxis_ ? new binning::AxisSpecs(*_orig.zaxis_) : nullptr) {}

  MESetNonObject::~MESetNonObject() {
    delete xaxis_;
    delete yaxis_;
    delete zaxis_;
  }

  MESet &MESetNonObject::operator=(MESet const &_rhs) {
    delete xaxis_;
    delete yaxis_;
    delete zaxis_;
    xaxis_ = nullptr;
    yaxis_ = nullptr;
    zaxis_ = nullptr;

    MESetNonObject const *pRhs(dynamic_cast<MESetNonObject const *>(&_rhs));
    if (pRhs) {
      if (pRhs->xaxis_)
        xaxis_ = new binning::AxisSpecs(*pRhs->xaxis_);
      if (pRhs->yaxis_)
        yaxis_ = new binning::AxisSpecs(*pRhs->yaxis_);
      if (pRhs->zaxis_)
        zaxis_ = new binning::AxisSpecs(*pRhs->zaxis_);
    }
    return MESet::operator=(_rhs);
  }

  MESet *MESetNonObject::clone(std::string const &_path /* = ""*/) const {
    std::string path(path_);
    if (!_path.empty())
      path_ = _path;
    MESet *copy(new MESetNonObject(*this));
    path_ = path;
    return copy;
  }

  void MESetNonObject::book(DQMStore::IBooker &_ibooker, EcalElectronicsMapping const *electronicsMap) {
    using namespace std;

    clear();

    if (path_.find('%') != string::npos)
      throw_("book() called with incompletely formed path");

    size_t slashPos(path_.find_last_of('/'));
    string name(path_.substr(slashPos + 1));
    _ibooker.setCurrentFolder(path_.substr(0, slashPos));
    auto oldscope = MonitorElementData::Scope::RUN;
    if (lumiFlag_)
      oldscope = _ibooker.setScope(MonitorElementData::Scope::LUMI);

    MonitorElement *me(nullptr);

    switch (kind_) {
      case MonitorElement::Kind::REAL:
        me = _ibooker.bookFloat(name);
        break;

      case MonitorElement::Kind::TH1F: {
        if (!xaxis_)
          throw_("No xaxis found for MESetNonObject");

        if (xaxis_->edges.empty())
          me = _ibooker.book1D(name, name, xaxis_->nbins, xaxis_->low, xaxis_->high);
        else
          me = _ibooker.book1D(name, name, xaxis_->nbins, &(xaxis_->edges[0]));
      } break;

      case MonitorElement::Kind::TPROFILE: {
        if (!xaxis_)
          throw_("No xaxis found for MESetNonObject");

        double ylow, yhigh;
        if (!yaxis_) {
          ylow = -numeric_limits<double>::max();
          yhigh = numeric_limits<double>::max();
        } else {
          ylow = yaxis_->low;
          yhigh = yaxis_->high;
        }
        if (xaxis_->edges.empty()) {
          me = _ibooker.bookProfile(name, name, xaxis_->nbins, xaxis_->low, xaxis_->high, ylow, yhigh, "");
        } else {
          // DQMStore bookProfile interface uses double* for bin edges
          double *edges(new double[xaxis_->nbins + 1]);
          std::copy(xaxis_->edges.begin(), xaxis_->edges.end(), edges);
          me = _ibooker.bookProfile(name, name, xaxis_->nbins, edges, ylow, yhigh, "");
          delete[] edges;
        }
      } break;

      case MonitorElement::Kind::TH2F: {
        if (!xaxis_ || !yaxis_)
          throw_("No x/yaxis found for MESetNonObject");

        if (xaxis_->edges.empty() && yaxis_->edges.empty())  // unlike MESetEcal, if either of X or Y is not set as
                                                             // variable, binning will be fixed
          me = _ibooker.book2D(
              name, name, xaxis_->nbins, xaxis_->low, xaxis_->high, yaxis_->nbins, yaxis_->low, yaxis_->high);
        else
          me = _ibooker.book2D(name, name, xaxis_->nbins, &(xaxis_->edges[0]), yaxis_->nbins, &(yaxis_->edges[0]));
      } break;

      case MonitorElement::Kind::TPROFILE2D: {
        if (!xaxis_ || !yaxis_)
          throw_("No x/yaxis found for MESetNonObject");
        if (!(xaxis_->edges.empty() && yaxis_->edges.empty()))
          throw_("Variable bin size for 2D profile not implemented");

        double high(0.), low(0.);
        if (zaxis_) {
          low = zaxis_->low;
          high = zaxis_->high;
        } else {
          low = -numeric_limits<double>::max();
          high = numeric_limits<double>::max();
        }

        me = _ibooker.bookProfile2D(name,
                                    name,
                                    xaxis_->nbins,
                                    xaxis_->low,
                                    xaxis_->high,
                                    yaxis_->nbins,
                                    yaxis_->low,
                                    yaxis_->high,
                                    low,
                                    high,
                                    "");
      } break;

      default:
        throw_("Unsupported MonitorElement kind");
    }

    if (xaxis_) {
      me->setAxisTitle(xaxis_->title, 1);
      if (!xaxis_->labels.empty()) {
        for (int iBin(1); iBin <= xaxis_->nbins; ++iBin)
          me->setBinLabel(iBin, xaxis_->labels[iBin - 1], 1);
      }
    }
    if (yaxis_) {
      me->setAxisTitle(yaxis_->title, 2);
      if (!yaxis_->labels.empty()) {
        for (int iBin(1); iBin <= yaxis_->nbins; ++iBin)
          me->setBinLabel(iBin, yaxis_->labels[iBin - 1], 2);
      }
    }
    if (zaxis_) {
      me->setAxisTitle(zaxis_->title, 3);
      if (!zaxis_->labels.empty()) {
        for (int iBin(1); iBin <= zaxis_->nbins; ++iBin)
          me->setBinLabel(iBin, zaxis_->labels[iBin - 1], 3);
      }
    }

    if (lumiFlag_)
      _ibooker.setScope(oldscope);

    mes_.push_back(me);

    active_ = true;
  }

  bool MESetNonObject::retrieve(EcalElectronicsMapping const *electronicsMap,
                                DQMStore::IGetter &_igetter,
                                std::string *_failedPath /* = 0*/) const {
    mes_.clear();

    MonitorElement *me(_igetter.get(path_));
    if (!me) {
      if (_failedPath)
        *_failedPath = path_;
      return false;
    }

    mes_.push_back(me);

    active_ = true;
    return true;
  }

  void MESetNonObject::fill(EcalDQMSetupObjects const edso, double _x, double _wy /* = 1.*/, double _w /* = 1.*/) {
    if (!active_)
      return;

    if (mes_.empty() || !mes_[0])
      return;

    switch (kind_) {
      case MonitorElement::Kind::REAL:
        mes_[0]->Fill(_x);
        break;
      case MonitorElement::Kind::TH1F:
      case MonitorElement::Kind::TPROFILE:
        mes_[0]->Fill(_x, _wy);
        break;
      case MonitorElement::Kind::TH2F:
      case MonitorElement::Kind::TPROFILE2D:
        mes_[0]->Fill(_x, _wy, _w);
        break;
      default:
        break;
    }
  }

  void MESetNonObject::setBinContent(EcalDQMSetupObjects const edso, int _bin, double _content) {
    if (!active_)
      return;
    if (kind_ == MonitorElement::Kind::REAL)
      return;

    if (mes_.empty() || !mes_[0])
      return;

    mes_[0]->setBinContent(_bin, _content);
  }

  void MESetNonObject::setBinError(EcalDQMSetupObjects const edso, int _bin, double _error) {
    if (!active_)
      return;
    if (kind_ == MonitorElement::Kind::REAL)
      return;

    if (mes_.empty() || !mes_[0])
      return;

    mes_[0]->setBinError(_bin, _error);
  }

  void MESetNonObject::setBinEntries(EcalDQMSetupObjects const edso, int _bin, double _entries) {
    if (!active_)
      return;
    if (kind_ != MonitorElement::Kind::TPROFILE && kind_ != MonitorElement::Kind::TPROFILE2D)
      return;

    if (mes_.empty() || !mes_[0])
      return;

    mes_[0]->setBinEntries(_bin, _entries);
  }

  double MESetNonObject::getBinContent(EcalDQMSetupObjects const edso, int _bin, int) const {
    if (!active_)
      return 0.;
    if (kind_ == MonitorElement::Kind::REAL)
      return 0.;

    if (mes_.empty() || !mes_[0])
      return 0.;

    return mes_[0]->getBinContent(_bin);
  }

  double MESetNonObject::getFloatValue() const {
    if (kind_ == MonitorElement::Kind::REAL)
      return mes_[0]->getFloatValue();
    else
      return 0.;
  }

  double MESetNonObject::getBinError(EcalDQMSetupObjects const edso, int _bin, int) const {
    if (!active_)
      return 0.;
    if (kind_ == MonitorElement::Kind::REAL)
      return 0.;

    if (mes_.empty() || !mes_[0])
      return 0.;

    return mes_[0]->getBinError(_bin);
  }

  double MESetNonObject::getBinEntries(EcalDQMSetupObjects const edso, int _bin, int) const {
    if (!active_)
      return 0.;
    if (kind_ != MonitorElement::Kind::TPROFILE && kind_ != MonitorElement::Kind::TPROFILE2D)
      return 0.;

    if (mes_.empty() || !mes_[0])
      return 0.;

    return mes_[0]->getBinEntries(_bin);
  }

  int MESetNonObject::findBin(EcalDQMSetupObjects const edso, double _x, double _y /* = 0.*/) const {
    if (!active_)
      return 0;

    if (mes_.empty() || !mes_[0])
      return 0;

    if (kind_ == MonitorElement::Kind::TH1F || kind_ == MonitorElement::Kind::TPROFILE)
      return mes_[0]->getTH1()->FindBin(_x);
    else if (kind_ == MonitorElement::Kind::TH2F || kind_ == MonitorElement::Kind::TPROFILE2D)
      return mes_[0]->getTH1()->FindBin(_x, _y);
    else
      return 0;
  }

  bool MESetNonObject::isVariableBinning() const {
    return (xaxis_ && (!xaxis_->edges.empty())) || (yaxis_ && (!yaxis_->edges.empty())) ||
           (zaxis_ && (!zaxis_->edges.empty()));
  }
}  // namespace ecaldqm
