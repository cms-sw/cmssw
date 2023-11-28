#include "DQM/EcalCommon/interface/MESet.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DQM/EcalCommon/interface/MESetUtils.h"
#include "DQM/EcalCommon/interface/StatusManager.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TPRegexp.h"
#include "TString.h"

namespace ecaldqm {
  MESet::MESet()
      : mes_(0),
        path_(""),
        otype_(binning::nObjType),
        btype_(binning::nBinType),
        kind_(MonitorElement::Kind::INVALID),
        lumiFlag_(false),
        batchMode_(false),
        active_(false) {}

  MESet::MESet(std::string const &_path,
               binning::ObjectType _otype,
               binning::BinningType _btype,
               MonitorElement::Kind _kind)
      : mes_(0),
        path_(_path),
        otype_(_otype),
        btype_(_btype),
        kind_(_kind),
        lumiFlag_(false),
        batchMode_(false),
        active_(false) {
    if (path_.empty() || path_.find('/') == std::string::npos ||
        (otype_ != binning::kChannel && path_.find('/') == path_.size() - 1))
      throw_(_path + " cannot be used for ME path name");

    switch (kind_) {
      case MonitorElement::Kind::REAL:
      case MonitorElement::Kind::TH1F:
      case MonitorElement::Kind::TPROFILE:
      case MonitorElement::Kind::TH2F:
      case MonitorElement::Kind::TPROFILE2D:
        break;
      default:
        throw_("Unsupported MonitorElement kind");
    }
  }

  MESet::MESet(MESet const &_orig)
      : mes_(_orig.mes_),
        path_(_orig.path_),
        otype_(_orig.otype_),
        btype_(_orig.btype_),
        kind_(_orig.kind_),
        lumiFlag_(_orig.lumiFlag_),
        batchMode_(_orig.batchMode_),
        active_(_orig.active_) {}

  MESet::~MESet() {}

  MESet &MESet::operator=(MESet const &_rhs) {
    mes_ = _rhs.mes_;
    path_ = _rhs.path_;
    otype_ = _rhs.otype_;
    btype_ = _rhs.btype_;
    kind_ = _rhs.kind_;
    active_ = _rhs.active_;

    return *this;
  }

  MESet *MESet::clone(std::string const &_path /* = ""*/) const {
    std::string path(path_);
    if (!_path.empty())
      path_ = _path;
    MESet *copy(new MESet(*this));
    path_ = path;
    return copy;
  }

  void MESet::clear() const {
    active_ = false;
    mes_.clear();
  }

  void MESet::setAxisTitle(std::string const &_title, int _axis /* = 1*/) {
    if (!active_)
      return;

    unsigned nME(mes_.size());
    for (unsigned iME(0); iME < nME; iME++)
      mes_[iME]->setAxisTitle(_title, _axis);
  }

  void MESet::reset(EcalElectronicsMapping const *electronicsMap,
                    double _content /* = 0.*/,
                    double _err /* = 0.*/,
                    double _entries /* = 0.*/) {
    if (!active_)
      return;

    resetAll(_content, _err, _entries);
  }

  void MESet::resetAll(double _content /* = 0.*/, double _err /* = 0.*/, double _entries /* = 0.*/) {
    if (!active_)
      return;

    unsigned nME(mes_.size());

    if (kind_ == MonitorElement::Kind::REAL) {
      for (unsigned iME(0); iME < nME; iME++)
        mes_[iME]->Fill(_content);
      return;
    }

    bool simple(true);
    if (_content != 0. || _err != 0. || _entries != 0.)
      simple = false;

    for (unsigned iME(0); iME < nME; iME++) {
      TH1 *h(mes_[iME]->getTH1());
      h->Reset();
      if (simple)
        continue;

      int nbinsX(h->GetNbinsX());
      int nbinsY(h->GetNbinsY());
      double entries(0.);
      for (int ix(1); ix <= nbinsX; ix++) {
        for (int iy(1); iy <= nbinsY; iy++) {
          int bin(h->GetBin(ix, iy));
          h->SetBinContent(bin, _content);
          h->SetBinError(bin, _err);
          if (kind_ == MonitorElement::Kind::TPROFILE) {
            static_cast<TProfile *>(h)->SetBinEntries(bin, _entries);
            entries += _entries;
          } else if (kind_ == MonitorElement::Kind::TPROFILE2D) {
            static_cast<TProfile2D *>(h)->SetBinEntries(bin, _entries);
            entries += _entries;
          }
        }
      }
      if (entries == 0.)
        entries = _entries;
      h->SetEntries(entries);
    }
  }

  std::string MESet::formPath(std::map<std::string, std::string> const &_replacements) const {
    TString path(path_);

    for (typename MESet::PathReplacements::const_iterator repItr(_replacements.begin()); repItr != _replacements.end();
         ++repItr) {
      TString pattern("\\%\\(");
      pattern += repItr->first;
      pattern += "\\)s";

      TPRegexp re(pattern);

      re.Substitute(path, repItr->second, "g");
    }

    return path.Data();
  }

  bool MESet::maskMatches(DetId const &_id,
                          uint32_t _mask,
                          StatusManager const *_statusManager,
                          EcalTrigTowerConstituentsMap const *trigTowerMap) const {
    if (!_statusManager)
      return false;

    if ((_statusManager->getStatus(_id.rawId()) & _mask) != 0)
      return true;

    int subdet(_id.subdetId());

    bool searchNeighborsInTower(btype_ == binning::kTriggerTower || btype_ == binning::kSuperCrystal);

    // turn off masking for good channel for the time being
    // update the RenderPlugin then enable again

    switch (subdet) {
      case EcalBarrel:  // this is a DetId for single crystal in barrel
      {
        EBDetId ebId(_id);
        EcalTrigTowerDetId ttId(ebId.tower());
        if ((_statusManager->getStatus(ttId.rawId()) & _mask) != 0)
          return true;

        if (searchNeighborsInTower) {
          std::vector<DetId> ids(trigTowerMap->constituentsOf(ttId));
          for (std::vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr)
            if ((_statusManager->getStatus(idItr->rawId()) & _mask) != 0)
              return true;
        }
      } break;

      case EcalEndcap:
        if (isEcalScDetId(_id)) {
          EcalScDetId scId(_id);
          for (int ix(1); ix <= 5; ix++) {
            for (int iy(1); iy <= 5; iy++) {
              int iix((scId.ix() - 1) * 5 + ix);
              int iiy((scId.iy() - 1) * 5 + iy);
              if (!EEDetId::validDetId(iix, iiy, scId.zside()))
                continue;
              if ((_statusManager->getStatus(EEDetId(iix, iiy, scId.zside()).rawId()) & _mask) != 0)
                return true;
            }
          }
        } else {
          EEDetId eeId(_id);
          EcalScDetId scId(eeId.sc());
          if ((_statusManager->getStatus(scId.rawId()) & _mask) != 0)
            return true;

          if (searchNeighborsInTower) {
            for (int ix(1); ix <= 5; ix++) {
              for (int iy(1); iy <= 5; iy++) {
                int iix((scId.ix() - 1) * 5 + ix);
                int iiy((scId.iy() - 1) * 5 + iy);
                if (!EEDetId::validDetId(iix, iiy, scId.zside()))
                  continue;
                if ((_statusManager->getStatus(EEDetId(iix, iiy, scId.zside()).rawId()) & _mask) != 0)
                  return true;
              }
            }
          }
        }
        break;

      case EcalTriggerTower: {
        EcalTrigTowerDetId ttId(_id);
        std::vector<DetId> ids(trigTowerMap->constituentsOf(ttId));
        for (std::vector<DetId>::iterator idItr(ids.begin()); idItr != ids.end(); ++idItr)
          if ((_statusManager->getStatus(idItr->rawId()) & _mask) != 0)
            return true;
      } break;

      default:
        break;
    }

    return false;
  }

  void MESet::fill_(unsigned _iME, int _bin, double _w) {
    if (kind_ == MonitorElement::Kind::REAL)
      return;

    MonitorElement *me(mes_[_iME]);
    if (!me)
      return;

    TH1 *h(me->getTH1());

    int nbinsX(h->GetNbinsX());

    double x(h->GetXaxis()->GetBinCenter(_bin % (nbinsX + 2)));

    if (kind_ == MonitorElement::Kind::TH1F || kind_ == MonitorElement::Kind::TPROFILE)
      me->Fill(x, _w);
    else {
      double y(h->GetYaxis()->GetBinCenter(_bin / (nbinsX + 2)));
      me->Fill(x, y, _w);
    }
  }

  void MESet::fill_(unsigned _iME, int _bin, double _y, double _w) {
    if (kind_ != MonitorElement::Kind::TH2F && kind_ != MonitorElement::Kind::TPROFILE2D)
      return;

    MonitorElement *me(mes_[_iME]);
    if (!me)
      return;

    TH1 *h(me->getTH1());

    int nbinsX(h->GetNbinsX());

    double x(h->GetXaxis()->GetBinCenter(_bin % (nbinsX + 2)));
    me->Fill(x, _y, _w);
  }

  void MESet::fill_(unsigned _iME, double _x, double _wy, double _w) {
    MonitorElement *me(mes_[_iME]);
    if (!me)
      return;

    if (kind_ == MonitorElement::Kind::REAL)
      me->Fill(_x);
    else if (kind_ == MonitorElement::Kind::TH1F || kind_ == MonitorElement::Kind::TPROFILE)
      me->Fill(_x, _wy);
    else
      me->Fill(_x, _wy, _w);
  }

  MESet::ConstBin::ConstBin(MESet const &_meSet, unsigned _iME /* = 0*/, int _iBin /* = 1*/)
      : meSet_(&_meSet), iME(_iME), iBin(_iBin), otype(binning::nObjType) {
    if (iME == unsigned(-1))
      return;

    // current internal bin numbering scheme does not allow 1D histograms
    // (overflow & underflow in each y)
    MonitorElement::Kind kind(meSet_->getKind());
    //    if(kind != MonitorElement::Kind::TH1F && kind !=
    //    MonitorElement::Kind::TPROFILE &&
    if (kind != MonitorElement::Kind::TH2F && kind != MonitorElement::Kind::TPROFILE2D)
      throw cms::Exception("InvalidOperation") << "MESet::ConstBin::Ctor: const_iterator only available for MESet of "
                                                  "2D histograms";

    MonitorElement const *me(meSet_->getME(iME));

    if (!me)
      throw cms::Exception("InvalidOperation")
          << "MESet::ConstBin::Ctor: ME " << iME << " does not exist for MESet " << meSet_->getPath();

    if (iBin == 1 && (kind == MonitorElement::Kind::TH2F || kind == MonitorElement::Kind::TPROFILE2D))
      iBin = me->getNbinsX() + 3;

    otype = binning::getObject(meSet_->getObjType(), iME);
  }

  MESet::ConstBin &MESet::ConstBin::operator=(ConstBin const &_rhs) {
    if (meSet_->getObjType() != _rhs.meSet_->getObjType() || meSet_->getBinType() != _rhs.meSet_->getBinType())
      throw cms::Exception("IncompatibleAssignment")
          << "Iterator of otype " << _rhs.meSet_->getObjType() << " and btype " << _rhs.meSet_->getBinType()
          << " to otype " << meSet_->getObjType() << " and btype " << meSet_->getBinType() << " ("
          << _rhs.meSet_->getPath() << " to " << meSet_->getPath() << ")";

    iME = _rhs.iME;
    iBin = _rhs.iBin;
    otype = _rhs.otype;

    return *this;
  }

  MESet::const_iterator::const_iterator(EcalElectronicsMapping const *electronicsMap,
                                        MESet const &_meSet,
                                        DetId const &_id)
      : bin_() {
    binning::ObjectType otype(_meSet.getObjType());
    unsigned iME(binning::findPlotIndex(electronicsMap, otype, _id));
    if (iME == unsigned(-1))
      return;

    binning::BinningType btype(_meSet.getBinType());
    int bin(binning::findBin2D(electronicsMap, otype, btype, _id));
    if (bin == 0)
      return;

    bin_.setMESet(_meSet);
    bin_.iME = iME;
    bin_.iBin = bin;
    bin_.otype = otype;
  }

  MESet::const_iterator &MESet::const_iterator::operator++() {
    unsigned &iME(bin_.iME);
    MESet const *meSet(bin_.getMESet());

    if (iME == unsigned(-1))
      return *this;

    int &bin(bin_.iBin);
    binning::ObjectType &otype(bin_.otype);

    MonitorElement::Kind kind(meSet->getKind());
    MonitorElement const *me(meSet->getME(iME));

    int nbinsX(me->getNbinsX());

    ++bin;
    if (bin == 1) {
      iME = 0;
      me = meSet->getME(iME);
      nbinsX = me->getNbinsX();
      if (kind == MonitorElement::Kind::TH2F || kind == MonitorElement::Kind::TPROFILE2D)
        bin = nbinsX + 3;
      otype = binning::getObject(meSet->getObjType(), iME);
    }

    if (bin % (nbinsX + 2) == nbinsX + 1) {
      if (kind == MonitorElement::Kind::TH1F || kind == MonitorElement::Kind::TPROFILE ||
          bin / (nbinsX + 2) == me->getNbinsY()) {
        iME += 1;
        me = meSet->getME(iME);
        if (!me) {
          iME = unsigned(-1);
          bin = -1;
          otype = binning::nObjType;
        } else {
          nbinsX = me->getNbinsX();
          if (kind == MonitorElement::Kind::TH2F || kind == MonitorElement::Kind::TPROFILE2D)
            bin = nbinsX + 3;
          else
            bin = 1;

          otype = binning::getObject(meSet->getObjType(), iME);
        }
      } else
        bin += 2;
    }

    return *this;
  }

  MESet::const_iterator &MESet::const_iterator::toNextChannel(const EcalElectronicsMapping *electronicsMap) {
    if (!bin_.getMESet())
      return *this;
    do
      operator++();
    while (bin_.iME != unsigned(-1) && !bin_.isChannel(electronicsMap));

    return *this;
  }

  bool MESet::const_iterator::up() {
    MESet const *meSet(bin_.getMESet());
    if (bin_.iME == unsigned(-1) || bin_.iBin < 1)
      return false;

    MonitorElement::Kind kind(meSet->getKind());
    if (kind != MonitorElement::Kind::TH2F && kind != MonitorElement::Kind::TPROFILE2D)
      return false;

    MonitorElement const *me(meSet->getME(bin_.iME));

    if (bin_.iBin / (me->getNbinsX() + 2) >= me->getNbinsY())
      return false;

    bin_.iBin += me->getNbinsX() + 2;

    return true;
  }

  bool MESet::const_iterator::down() {
    MESet const *meSet(bin_.getMESet());
    if (bin_.iME == unsigned(-1) || bin_.iBin < 1)
      return false;

    MonitorElement::Kind kind(meSet->getKind());
    if (kind != MonitorElement::Kind::TH2F && kind != MonitorElement::Kind::TPROFILE2D)
      return false;

    MonitorElement const *me(meSet->getME(bin_.iME));

    if (bin_.iBin / (me->getNbinsX() + 2) <= 1)
      return false;

    bin_.iBin -= me->getNbinsX() + 2;

    return true;
  }

  bool MESet::const_iterator::left() {
    MESet const *meSet(bin_.getMESet());
    if (bin_.iME == unsigned(-1) || bin_.iBin < 1)
      return false;

    MonitorElement::Kind kind(meSet->getKind());
    if (kind != MonitorElement::Kind::TH2F && kind != MonitorElement::Kind::TPROFILE2D)
      return false;

    MonitorElement const *me(meSet->getME(bin_.iME));

    if (bin_.iBin % (me->getNbinsX() + 2) <= 1)
      return false;

    bin_.iBin -= 1;

    return true;
  }

  bool MESet::const_iterator::right() {
    MESet const *meSet(bin_.getMESet());
    if (bin_.iME == unsigned(-1) || bin_.iBin < 1)
      return false;

    MonitorElement::Kind kind(meSet->getKind());
    if (kind != MonitorElement::Kind::TH2F && kind != MonitorElement::Kind::TPROFILE2D)
      return false;

    MonitorElement const *me(meSet->getME(bin_.iME));

    if (bin_.iBin % (me->getNbinsX() + 2) >= me->getNbinsX())
      return false;

    bin_.iBin += 1;

    return true;
  }
}  // namespace ecaldqm
