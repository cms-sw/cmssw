#include "DQM/EcalCommon/interface/MESetEcal.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetUtils.h"

#include <limits>
#include <sstream>

namespace ecaldqm {
  MESetEcal::MESetEcal(std::string const &_fullPath,
                       binning::ObjectType _otype,
                       binning::BinningType _btype,
                       MonitorElement::Kind _kind,
                       unsigned _logicalDimensions,
                       binning::AxisSpecs const *_xaxis /* = 0*/,
                       binning::AxisSpecs const *_yaxis /* = 0*/,
                       binning::AxisSpecs const *_zaxis /* = 0*/)
      : MESet(_fullPath, _otype, _btype, _kind),
        logicalDimensions_(_logicalDimensions),
        xaxis_(_xaxis ? new binning::AxisSpecs(*_xaxis) : nullptr),
        yaxis_(_yaxis ? new binning::AxisSpecs(*_yaxis) : nullptr),
        zaxis_(_zaxis ? new binning::AxisSpecs(*_zaxis) : nullptr) {
    if (btype_ == binning::kUser && ((logicalDimensions_ > 0 && !xaxis_) || (logicalDimensions_ > 1 && !yaxis_)))
      throw_("Need axis specifications");
  }

  MESetEcal::MESetEcal(MESetEcal const &_orig)
      : MESet(_orig),
        logicalDimensions_(_orig.logicalDimensions_),
        xaxis_(_orig.xaxis_ ? new binning::AxisSpecs(*_orig.xaxis_) : nullptr),
        yaxis_(_orig.yaxis_ ? new binning::AxisSpecs(*_orig.yaxis_) : nullptr),
        zaxis_(_orig.zaxis_ ? new binning::AxisSpecs(*_orig.zaxis_) : nullptr) {}

  MESetEcal::~MESetEcal() {
    delete xaxis_;
    delete yaxis_;
    delete zaxis_;
  }

  MESet &MESetEcal::operator=(MESet const &_rhs) {
    delete xaxis_;
    delete yaxis_;
    delete zaxis_;
    xaxis_ = nullptr;
    yaxis_ = nullptr;
    zaxis_ = nullptr;

    MESetEcal const *pRhs(dynamic_cast<MESetEcal const *>(&_rhs));
    if (pRhs) {
      logicalDimensions_ = pRhs->logicalDimensions_;
      if (pRhs->xaxis_)
        xaxis_ = new binning::AxisSpecs(*pRhs->xaxis_);
      if (pRhs->yaxis_)
        yaxis_ = new binning::AxisSpecs(*pRhs->yaxis_);
      if (pRhs->zaxis_)
        zaxis_ = new binning::AxisSpecs(*pRhs->zaxis_);
    }
    return MESet::operator=(_rhs);
  }

  MESet *MESetEcal::clone(std::string const &_path /* = ""*/) const {
    std::string path(path_);
    if (!_path.empty())
      path_ = _path;
    MESet *copy(new MESetEcal(*this));
    path_ = path;
    return copy;
  }

  void MESetEcal::book(DQMStore::IBooker &_ibooker) {
    using namespace std;

    auto oldscope = MonitorElementData::Scope::RUN;
    if (lumiFlag_)
      oldscope = _ibooker.setScope(MonitorElementData::Scope::LUMI);

    clear();

    vector<string> mePaths(generatePaths());

    for (unsigned iME(0); iME < mePaths.size(); iME++) {
      string &path(mePaths[iME]);
      if (path.find('%') != string::npos)
        throw_("book() called with incompletely formed path [" + path + "]");

      binning::ObjectType actualObject(binning::getObject(otype_, iME));

      binning::AxisSpecs xaxis, yaxis, zaxis;

      bool isHistogram(logicalDimensions_ > 0);
      bool isMap(logicalDimensions_ > 1);

      if (isHistogram) {
        if (xaxis_)
          xaxis = *xaxis_;
        if (yaxis_)
          yaxis = *yaxis_;
        if (zaxis_)
          zaxis = *zaxis_;

        if (xaxis.nbins == 0) {  // uses preset
          binning::AxisSpecs xdef(binning::getBinning(actualObject, btype_, isMap, 1, iME));
          if (xaxis.labels || !xaxis.title.empty()) {  // PSet specifies title / label only
            std::string *labels(xaxis.labels);
            std::string title(xaxis.title);
            xaxis = xdef;
            delete[] xaxis.labels;
            xaxis.labels = labels;
            xaxis.title = title;
          } else
            xaxis = xdef;
        }

        if (isMap && yaxis.nbins == 0) {
          binning::AxisSpecs ydef(binning::getBinning(actualObject, btype_, isMap, 2, iME));
          if (yaxis.labels || !yaxis.title.empty()) {  // PSet specifies title / label only
            std::string *labels(yaxis.labels);
            std::string title(yaxis.title);
            yaxis = ydef;
            delete[] yaxis.labels;
            yaxis.labels = labels;
            yaxis.title = title;
          } else
            yaxis = ydef;
        }

        if (yaxis.high - yaxis.low < 1.e-10) {
          yaxis.low = -numeric_limits<double>::max();
          yaxis.high = numeric_limits<double>::max();
        }

        if (zaxis.high - zaxis.low < 1.e-10) {
          zaxis.low = -numeric_limits<double>::max();
          zaxis.high = numeric_limits<double>::max();
        }
      }

      size_t slashPos(path.find_last_of('/'));
      string name(path.substr(slashPos + 1));
      _ibooker.cd();
      _ibooker.setCurrentFolder(path.substr(0, slashPos));

      MonitorElement *me(nullptr);

      switch (kind_) {
        case MonitorElement::Kind::REAL:
          me = _ibooker.bookFloat(name);

          break;

        case MonitorElement::Kind::TH1F:
          if (xaxis.edges)
            me = _ibooker.book1D(name, name, xaxis.nbins, xaxis.edges);
          else
            me = _ibooker.book1D(name, name, xaxis.nbins, xaxis.low, xaxis.high);

          break;

        case MonitorElement::Kind::TPROFILE:
          if (xaxis.edges) {
            // DQMStore bookProfile interface uses double* for bin edges
            double *edges(new double[xaxis.nbins + 1]);
            std::copy(xaxis.edges, xaxis.edges + xaxis.nbins + 1, edges);
            me = _ibooker.bookProfile(name, name, xaxis.nbins, edges, yaxis.low, yaxis.high, "");
            delete[] edges;
          } else
            me = _ibooker.bookProfile(name, name, xaxis.nbins, xaxis.low, xaxis.high, yaxis.low, yaxis.high, "");

          break;

        case MonitorElement::Kind::TH2F:
          if (xaxis.edges || yaxis.edges) {
            binning::AxisSpecs *specs[] = {&xaxis, &yaxis};
            for (int iSpec(0); iSpec < 2; iSpec++) {
              if (!specs[iSpec]->edges) {
                specs[iSpec]->edges = new float[specs[iSpec]->nbins + 1];
                int nbins(specs[iSpec]->nbins);
                double low(specs[iSpec]->low), high(specs[iSpec]->high);
                for (int i(0); i < nbins + 1; i++)
                  specs[iSpec]->edges[i] = low + (high - low) / nbins * i;
              }
            }
            me = _ibooker.book2D(name, name, xaxis.nbins, xaxis.edges, yaxis.nbins, yaxis.edges);
          } else
            me = _ibooker.book2D(name, name, xaxis.nbins, xaxis.low, xaxis.high, yaxis.nbins, yaxis.low, yaxis.high);

          break;

        case MonitorElement::Kind::TPROFILE2D:
          if (zaxis.edges) {
            zaxis.low = zaxis.edges[0];
            zaxis.high = zaxis.edges[zaxis.nbins];
          }
          if (xaxis.edges || yaxis.edges)
            throw_("Variable bin size for 2D profile not implemented");
          me = _ibooker.bookProfile2D(name,
                                      name,
                                      xaxis.nbins,
                                      xaxis.low,
                                      xaxis.high,
                                      yaxis.nbins,
                                      yaxis.low,
                                      yaxis.high,
                                      zaxis.low,
                                      zaxis.high,
                                      "");

          break;

        default:
          break;
      }

      if (!me)
        throw_("ME could not be booked");

      if (isHistogram) {
        me->setAxisTitle(xaxis.title, 1);
        me->setAxisTitle(yaxis.title, 2);
        if (isMap)
          me->setAxisTitle(zaxis.title, 3);

        if (xaxis.labels) {
          for (int iBin(1); iBin <= xaxis.nbins; ++iBin)
            me->setBinLabel(iBin, xaxis.labels[iBin - 1], 1);
        }
        if (yaxis.labels) {
          for (int iBin(1); iBin <= yaxis.nbins; ++iBin)
            me->setBinLabel(iBin, yaxis.labels[iBin - 1], 2);
        }
        if (zaxis.labels) {
          for (int iBin(1); iBin <= zaxis.nbins; ++iBin)
            me->setBinLabel(iBin, zaxis.labels[iBin - 1], 3);
        }

        /* FIX: In ROOT 6.0.12 bit 20 is used by ROOT (bit 19 was already in use
      in 6.0.x). Talking with the ROOT team, users should never use SetBit for
      their own purpose since those bits are reserved for ROOT internally. See
      https://github.com/cms-sw/cmssw/issues/21423 for some alternative ways of
      attaching additional information to a TH1

      // For plot tagging in RenderPlugin; default values are 1 for both
      // bits 19 - 23 are free in TH1::fBits
      // can only pack object + logical dimensions into 5 bits (4 bits for
      object, 1 bit for dim (1 -> dim >= 2))
      me->getTH1()->SetBit(uint32_t(actualObject + 1) << 20);
      if(isMap) me->getTH1()->SetBit(0x1 << 19);
      */

        // The render plugin requires some metadata in order to set the correct
        // rendering for each plot. The original solution was to use bits number
        // 19 to 23 in TH1::fBits, but these were meant for ROOT's internal usage
        // and eventually started breaking things, see:
        // https://github.com/cms-sw/cmssw/issues/21423 The current solution is to
        // use the UniqueID field in the parent TObject, which is expected to work
        // if there is no TRef pointing to the TObject.

        // To check that indeed there is no such TRef, one can use
        // TestBit(kIsReferenced), e.g. std::cout << "Need to set unique ID at
        // path = " << path << "; for this path, TestBit(kIsReferenced) is: " <<
        // (me->getTH1()->TestBit(kIsReferenced)? "true": "false") << std::endl;
        // // should always output false for the solution to work

        // Originally, bit 19 in TH1::fBits was set to isMap, while bits 20-23
        // contained (actualObject+1). The idea is to make sure that both these
        // variables are easily recoverable in the render plugin, as in this
        // solution, where isMap is the last bit.
        me->getTH1()->SetUniqueID(uint32_t(2 * (actualObject + 1) + (isMap ? 1 : 0)));
      }

      mes_.push_back(me);
    }

    if (lumiFlag_)
      _ibooker.setScope(oldscope);

    active_ = true;
  }

  bool MESetEcal::retrieve(DQMStore::IGetter &_igetter, std::string *_failedPath /* = 0*/) const {
    clear();

    std::vector<std::string> mePaths(generatePaths());
    if (mePaths.empty()) {
      if (_failedPath)
        _failedPath->clear();
      return false;
    }

    for (unsigned iME(0); iME < mePaths.size(); iME++) {
      std::string &path(mePaths[iME]);
      if (path.find('%') != std::string::npos)
        throw_("retrieve() called with incompletely formed path [" + path + "]");

      MonitorElement *me(_igetter.get(path));
      if (me)
        mes_.push_back(me);
      else {
        clear();
        if (_failedPath)
          *_failedPath = path;
        return false;
      }
    }

    active_ = true;
    return true;
  }

  void MESetEcal::fill(DetId const &_id, double _x /* = 1.*/, double _wy /* = 1.*/, double _w /* = 1.*/) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    fill_(iME, _x, _wy, _w);
  }

  void MESetEcal::fill(EcalElectronicsId const &_id, double _x /* = 1.*/, double _wy /* = 1.*/, double _w /* = 1.*/) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    fill_(iME, _x, _wy, _w);
  }

  void MESetEcal::fill(int _dcctccid, double _x /* = 1.*/, double _wy /* = 1.*/, double _w /* = 1.*/) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    fill_(iME, _x, _wy, _w);
  }

  void MESetEcal::fill(double _x, double _wy /* = 1.*/, double _w /* = 1.*/) {
    if (!active_)
      return;

    if (mes_.size() != 1)
      return;

    fill_(0, _x, _wy, _w);
  }

  void MESetEcal::setBinContent(DetId const &_id, int _bin, double _content) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinContent(_bin, _content);
  }

  void MESetEcal::setBinContent(EcalElectronicsId const &_id, int _bin, double _content) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinContent(_bin, _content);
  }

  void MESetEcal::setBinContent(int _dcctccid, int _bin, double _content) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    mes_[iME]->setBinContent(_bin, _content);
  }

  void MESetEcal::setBinError(DetId const &_id, int _bin, double _error) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinError(_bin, _error);
  }

  void MESetEcal::setBinError(EcalElectronicsId const &_id, int _bin, double _error) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinError(_bin, _error);
  }

  void MESetEcal::setBinError(int _dcctccid, int _bin, double _error) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    mes_[iME]->setBinError(_bin, _error);
  }

  void MESetEcal::setBinEntries(DetId const &_id, int _bin, double _entries) {
    if (!active_)
      return;
    if (kind_ != MonitorElement::Kind::TPROFILE && kind_ != MonitorElement::Kind::TPROFILE2D)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinEntries(_bin, _entries);
  }

  void MESetEcal::setBinEntries(EcalElectronicsId const &_id, int _bin, double _entries) {
    if (!active_)
      return;
    if (kind_ != MonitorElement::Kind::TPROFILE && kind_ != MonitorElement::Kind::TPROFILE2D)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    mes_[iME]->setBinEntries(_bin, _entries);
  }

  void MESetEcal::setBinEntries(int _dcctccid, int _bin, double _entries) {
    if (!active_)
      return;
    if (kind_ != MonitorElement::Kind::TPROFILE && kind_ != MonitorElement::Kind::TPROFILE2D)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    mes_[iME]->setBinEntries(_bin, _entries);
  }

  double MESetEcal::getBinContent(DetId const &_id, int _bin) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinContent(_bin);
  }

  double MESetEcal::getBinContent(EcalElectronicsId const &_id, int _bin) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinContent(_bin);
  }

  double MESetEcal::getBinContent(int _dcctccid, int _bin) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getBinContent(_bin);
  }

  double MESetEcal::getBinError(DetId const &_id, int _bin) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinError(_bin);
  }

  double MESetEcal::getBinError(EcalElectronicsId const &_id, int _bin) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinError(_bin);
  }

  double MESetEcal::getBinError(int _dcctccid, int _bin) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getBinError(_bin);
  }

  double MESetEcal::getBinEntries(DetId const &_id, int _bin) const {
    if (!active_)
      return 0.;
    if (kind_ != MonitorElement::Kind::TPROFILE && kind_ != MonitorElement::Kind::TPROFILE2D)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinEntries(_bin);
  }

  double MESetEcal::getBinEntries(EcalElectronicsId const &_id, int _bin) const {
    if (!active_)
      return 0.;
    if (kind_ != MonitorElement::Kind::TPROFILE && kind_ != MonitorElement::Kind::TPROFILE2D)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getBinEntries(_bin);
  }

  double MESetEcal::getBinEntries(int _dcctccid, int _bin) const {
    if (!active_)
      return 0.;
    if (kind_ != MonitorElement::Kind::TPROFILE && kind_ != MonitorElement::Kind::TPROFILE2D)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getBinEntries(_bin);
  }

  int MESetEcal::findBin(DetId const &_id, double _x, double _y /* = 0.*/) const {
    if (!active_)
      return -1;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_x, _y);
  }

  int MESetEcal::findBin(EcalElectronicsId const &_id, double _x, double _y /* = 0.*/) const {
    if (!active_)
      return -1;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_x, _y);
  }

  int MESetEcal::findBin(int _dcctccid, double _x, double _y /* = 0.*/) const {
    if (!active_)
      return -1;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getTH1()->FindBin(_x, _y);
  }

  bool MESetEcal::isVariableBinning() const {
    return (xaxis_ && xaxis_->edges) || (yaxis_ && yaxis_->edges) || (zaxis_ && zaxis_->edges);
  }

  std::vector<std::string> MESetEcal::generatePaths() const {
    using namespace std;

    vector<string> paths(0);

    unsigned nME(binning::getNObjects(otype_));

    for (unsigned iME(0); iME < nME; iME++) {
      binning::ObjectType obj(binning::getObject(otype_, iME));

      string path(path_);
      map<string, string> replacements;

      switch (obj) {
        case binning::kEB:
        case binning::kEBMEM:
          replacements["subdet"] = "EcalBarrel";
          replacements["prefix"] = "EB";
          replacements["suffix"] = "";
          replacements["subdetshort"] = "EB";
          replacements["subdetshortsig"] = "EB";
          replacements["supercrystal"] = "trigger tower";
          break;
        case binning::kEE:
        case binning::kEEMEM:
          replacements["subdet"] = "EcalEndcap";
          replacements["prefix"] = "EE";
          replacements["subdetshort"] = "EE";
          replacements["subdetshortsig"] = "EE";
          replacements["supercrystal"] = "super crystal";
          break;
        case binning::kEEm:
          replacements["subdet"] = "EcalEndcap";
          replacements["prefix"] = "EE";
          replacements["suffix"] = " EE -";
          replacements["subdetshort"] = "EE";
          replacements["subdetshortsig"] = "EEM";
          replacements["supercrystal"] = "super crystal";
          break;
        case binning::kEEp:
          replacements["subdet"] = "EcalEndcap";
          replacements["prefix"] = "EE";
          replacements["suffix"] = " EE +";
          replacements["subdetshort"] = "EE";
          replacements["subdetshortsig"] = "EEP";
          replacements["supercrystal"] = "super crystal";
          break;
        case binning::kSM:
          if (iME <= kEEmHigh || iME >= kEEpLow) {
            replacements["subdet"] = "EcalEndcap";
            replacements["prefix"] = "EE";
            replacements["supercrystal"] = "super crystal";
          } else {
            replacements["subdet"] = "EcalBarrel";
            replacements["prefix"] = "EB";
            replacements["supercrystal"] = "trigger tower";
          }
          replacements["sm"] = binning::channelName(iME + 1);
          break;
        case binning::kEBSM:
          replacements["subdet"] = "EcalBarrel";
          replacements["prefix"] = "EB";
          replacements["sm"] = binning::channelName(iME + kEBmLow + 1);
          replacements["supercrystal"] = "trigger tower";
          break;
        case binning::kEESM:
          replacements["subdet"] = "EcalEndcap";
          replacements["prefix"] = "EE";
          replacements["sm"] = binning::channelName(iME <= kEEmHigh ? iME + 1 : iME + 37);
          replacements["supercrystal"] = "super crystal";
          break;
        case binning::kSMMEM: {
          unsigned iDCC(memDCCId(iME) - 1);
          // dccId(unsigned) skips DCCs without MEM
          if (iDCC <= kEEmHigh || iDCC >= kEEpLow) {
            replacements["subdet"] = "EcalEndcap";
            replacements["prefix"] = "EE";
          } else {
            replacements["subdet"] = "EcalBarrel";
            replacements["prefix"] = "EB";
          }
          replacements["sm"] = binning::channelName(iDCC + 1);
        } break;
        case binning::kEBSMMEM: {
          unsigned iDCC(memDCCId(iME + 4) - 1);
          replacements["subdet"] = "EcalBarrel";
          replacements["prefix"] = "EB";
          replacements["sm"] = binning::channelName(iDCC + 1);
        } break;
        case binning::kEESMMEM: {
          unsigned iDCC(memDCCId(iME < 4 ? iME : iME + 36) - 1);
          replacements["subdet"] = "EcalEndcap";
          replacements["prefix"] = "EE";
          replacements["sm"] = binning::channelName(iDCC + 1);
        }
        default:
          break;
      }

      paths.push_back(formPath(replacements));
    }

    return paths;
  }
}  // namespace ecaldqm
