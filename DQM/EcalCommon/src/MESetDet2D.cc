#include "DQM/EcalCommon/interface/MESetDet2D.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {
  MESetDet2D::MESetDet2D(std::string const &_fullPath,
                         binning::ObjectType _otype,
                         binning::BinningType _btype,
                         MonitorElement::Kind _kind,
                         binning::AxisSpecs const *_zaxis /* = 0*/)
      : MESetEcal(_fullPath, _otype, _btype, _kind, 2, nullptr, nullptr, _zaxis) {
    switch (kind_) {
      case MonitorElement::Kind::TH2F:
      case MonitorElement::Kind::TPROFILE2D:
        break;
      default:
        throw_("Unsupported MonitorElement kind");
    }
  }

  MESetDet2D::MESetDet2D(MESetDet2D const &_orig) : MESetEcal(_orig) {}

  MESetDet2D::~MESetDet2D() {}

  MESet *MESetDet2D::clone(std::string const &_path /* = ""*/) const {
    std::string path(path_);
    if (!_path.empty())
      path_ = _path;
    MESet *copy(new MESetDet2D(*this));
    path_ = path;
    return copy;
  }

  void MESetDet2D::book(DQMStore::IBooker &_ibooker) {
    MESetEcal::book(_ibooker);

    if (btype_ == binning::kCrystal) {
      for (unsigned iME(0); iME < mes_.size(); iME++) {
        MonitorElement *me(mes_[iME]);

        binning::ObjectType actualObject(binning::getObject(otype_, iME));
        if (actualObject == binning::kMEM) {
          for (int iBin(1); iBin <= me->getNbinsX(); ++iBin)
            me->setBinLabel(iBin, binning::channelName(memDCCId(iBin - 1)));
        }
        if (actualObject == binning::kEBMEM) {
          for (int iBin(1); iBin <= me->getNbinsX(); ++iBin)
            me->setBinLabel(iBin, binning::channelName(iBin + kEBmLow));
        }
        if (actualObject == binning::kEEMEM) {
          for (int iBin(1); iBin <= me->getNbinsX() / 2; ++iBin) {
            me->setBinLabel(iBin, binning::channelName(memDCCId(iBin - 1)));
            me->setBinLabel(iBin + me->getNbinsX() / 2, binning::channelName(memDCCId(iBin + 39)));
          }
        }
      }
    } else if (btype_ == binning::kDCC) {
      for (unsigned iME(0); iME < mes_.size(); iME++) {
        MonitorElement *me(mes_[iME]);

        binning::ObjectType actualObject(binning::getObject(otype_, iME));
        if (actualObject == binning::kEcal) {
          me->setBinLabel(1, "EE", 2);
          me->setBinLabel(6, "EE", 2);
          me->setBinLabel(3, "EB", 2);
          me->setBinLabel(5, "EB", 2);
        }
      }
    }

    // To avoid the ambiguity between "content == 0 because the mean is 0" and
    // "content == 0 because the entry is 0" RenderPlugin must be configured
    // accordingly
    if (!batchMode_ && kind_ == MonitorElement::Kind::TPROFILE2D)
      resetAll(0., 0., -1.);
  }

  void MESetDet2D::fill(DetId const &_id, double _w /* = 1.*/, double, double) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin;

    if (btype_ == binning::kRCT) {
      bin = binning::findBin2D(obj, btype_, _id);
      fill_(iME, bin, _w);
    } else {
      if (isEndcapTTId(_id)) {
        std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
        unsigned nId(ids.size());
        for (unsigned iId(0); iId < nId; iId++) {
          bin = binning::findBin2D(obj, binning::kTriggerTower, ids[iId]);
          fill_(iME, bin, _w);
        }
      } else {
        bin = binning::findBin2D(obj, btype_, _id);
        fill_(iME, bin, _w);
      }
    }
  }

  void MESetDet2D::fill(EcalElectronicsId const &_id, double _w /* = 1.*/, double, double) {
    if (!active_)
      return;

    unsigned iME(0);
    if (btype_ == binning::kPseudoStrip)
      iME = binning::findPlotIndex(otype_, _id.dccId(), binning::kPseudoStrip);
    else
      iME = binning::findPlotIndex(otype_, _id);
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin;

    if (btype_ == binning::kPseudoStrip) {
      EcalElectronicsId stid(_id);
      std::vector<DetId> ids(
          getElectronicsMap()->pseudoStripConstituents(stid.dccId(), stid.towerId(), stid.stripId()));
      unsigned nId(ids.size());
      for (unsigned iId(0); iId < nId; iId++) {
        bin = binning::findBin2D(obj, btype_, ids[iId]);
        fill_(iME, bin, _w);
      }
    } else {
      bin = binning::findBin2D(obj, btype_, _id);
      fill_(iME, bin, _w);
    }
  }

  void MESetDet2D::fill(int _dcctccid, double _w /* = 1.*/, double, double) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _dcctccid));
    fill_(iME, bin, _w);
  }

  void MESetDet2D::setBinContent(DetId const &_id, double _content) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin;

    if (isEndcapTTId(_id)) {
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      unsigned nId(ids.size());
      for (unsigned iId(0); iId < nId; iId++) {
        bin = binning::findBin2D(obj, binning::kTriggerTower, ids[iId]);
        mes_[iME]->setBinContent(bin, _content);
      }
    } else {
      bin = binning::findBin2D(obj, btype_, _id);
      mes_[iME]->setBinContent(bin, _content);
    }
  }

  void MESetDet2D::setBinContent(EcalElectronicsId const &_id, double _content) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _id));
    mes_[iME]->setBinContent(bin, _content);
  }

  void MESetDet2D::setBinContent(int _dcctccid, double _content) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _dcctccid));
    mes_[iME]->setBinContent(bin, _content);
  }

  void MESetDet2D::setBinError(DetId const &_id, double _error) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin;

    if (isEndcapTTId(_id)) {
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      unsigned nId(ids.size());
      for (unsigned iId(0); iId < nId; iId++) {
        bin = binning::findBin2D(obj, binning::kTriggerTower, ids[iId]);
        mes_[iME]->setBinError(bin, _error);
      }
    } else {
      bin = binning::findBin2D(obj, btype_, _id);
      mes_[iME]->setBinError(bin, _error);
    }
  }

  void MESetDet2D::setBinError(EcalElectronicsId const &_id, double _error) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _id));
    mes_[iME]->setBinError(bin, _error);
  }

  void MESetDet2D::setBinError(int _dcctccid, double _error) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _dcctccid));
    mes_[iME]->setBinError(bin, _error);
  }

  void MESetDet2D::setBinEntries(DetId const &_id, double _entries) {
    if (!active_)
      return;
    if (kind_ != MonitorElement::Kind::TPROFILE2D)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin;

    if (isEndcapTTId(_id)) {
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      unsigned nId(ids.size());
      for (unsigned iId(0); iId < nId; iId++) {
        bin = binning::findBin2D(obj, binning::kTriggerTower, ids[iId]);
        mes_[iME]->setBinEntries(bin, _entries);
      }
    } else {
      bin = binning::findBin2D(obj, btype_, _id);
      mes_[iME]->setBinEntries(bin, _entries);
    }
  }

  void MESetDet2D::setBinEntries(EcalElectronicsId const &_id, double _entries) {
    if (!active_)
      return;
    if (kind_ != MonitorElement::Kind::TPROFILE2D)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _id));
    mes_[iME]->setBinEntries(bin, _entries);
  }

  void MESetDet2D::setBinEntries(int _dcctccid, double _entries) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _dcctccid));
    mes_[iME]->setBinEntries(bin, _entries);
  }

  double MESetDet2D::getBinContent(DetId const &_id, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin;

    if (isEndcapTTId(_id)) {
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      bin = binning::findBin2D(obj, binning::kTriggerTower, ids[0]);
    } else {
      bin = binning::findBin2D(obj, btype_, _id);
    }

    return mes_[iME]->getBinContent(bin);
  }

  double MESetDet2D::getBinContent(EcalElectronicsId const &_id, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _id));

    return mes_[iME]->getBinContent(bin);
  }

  double MESetDet2D::getBinContent(int _dcctccid, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _dcctccid));

    return mes_[iME]->getBinContent(bin);
  }

  double MESetDet2D::getBinError(DetId const &_id, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin;

    if (isEndcapTTId(_id)) {
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      bin = binning::findBin2D(obj, binning::kTriggerTower, ids[0]);
    } else {
      bin = binning::findBin2D(obj, btype_, _id);
    }

    return mes_[iME]->getBinError(bin);
  }

  double MESetDet2D::getBinError(EcalElectronicsId const &_id, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _id));

    return mes_[iME]->getBinError(bin);
  }

  double MESetDet2D::getBinError(int _dcctccid, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _dcctccid));

    return mes_[iME]->getBinError(bin);
  }

  double MESetDet2D::getBinEntries(DetId const &_id, int) const {
    if (!active_)
      return 0.;
    if (kind_ != MonitorElement::Kind::TPROFILE2D)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin;

    if (isEndcapTTId(_id)) {
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      bin = binning::findBin2D(obj, binning::kTriggerTower, ids[0]);
    } else {
      bin = binning::findBin2D(obj, btype_, _id);
    }

    double entries(mes_[iME]->getBinEntries(bin));
    if (entries < 0.)
      return 0.;
    else
      return entries;
  }

  double MESetDet2D::getBinEntries(EcalElectronicsId const &_id, int) const {
    if (!active_)
      return 0.;
    if (kind_ != MonitorElement::Kind::TPROFILE2D)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _id));

    double entries(mes_[iME]->getBinEntries(bin));
    if (entries < 0.)
      return 0.;
    else
      return entries;
  }

  double MESetDet2D::getBinEntries(int _dcctccid, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    int bin(binning::findBin2D(obj, btype_, _dcctccid));

    double entries(mes_[iME]->getBinEntries(bin));
    if (entries < 0.)
      return 0.;
    else
      return entries;
  }

  int MESetDet2D::findBin(DetId const &_id) const {
    if (!active_)
      return 0;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    if (isEndcapTTId(_id)) {
      std::vector<DetId> ids(getTrigTowerMap()->constituentsOf(EcalTrigTowerDetId(_id)));
      return binning::findBin2D(obj, binning::kTriggerTower, ids[0]);
    } else
      return binning::findBin2D(obj, btype_, _id);
  }

  int MESetDet2D::findBin(EcalElectronicsId const &_id) const {
    if (!active_)
      return 0;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    binning::ObjectType obj(binning::getObject(otype_, iME));

    return binning::findBin2D(obj, btype_, _id);
  }

  void MESetDet2D::reset(double _content /* = 0.*/, double _err /* = 0.*/, double _entries /* = 0.*/) {
    unsigned nME(binning::getNObjects(otype_));

    bool isProfile(kind_ == MonitorElement::Kind::TPROFILE2D);

    for (unsigned iME(0); iME < nME; iME++) {
      MonitorElement *me(mes_[iME]);

      binning::ObjectType obj(binning::getObject(otype_, iME));

      int nbinsX(me->getTH1()->GetNbinsX());
      int nbinsY(me->getTH1()->GetNbinsY());
      for (int ix(1); ix <= nbinsX; ix++) {
        for (int iy(1); iy <= nbinsY; iy++) {
          int bin((nbinsX + 2) * iy + ix);
          if (!binning::isValidIdBin(obj, btype_, iME, bin))
            continue;
          me->setBinContent(bin, _content);
          me->setBinError(bin, _err);
          if (isProfile)
            me->setBinEntries(bin, _entries);
        }
      }
    }
  }

  void MESetDet2D::fill_(unsigned _iME, int _bin, double _w) {
    if (kind_ == MonitorElement::Kind::TPROFILE2D) {
      MonitorElement *me(mes_.at(_iME));
      if (me->getBinEntries(_bin) < 0.) {
        me->setBinContent(_bin, 0.);
        me->setBinEntries(_bin, 0.);
        me->getTProfile2D()->SetEntries(me->getTProfile2D()->GetEntries() + 1.);
      }
    }

    MESet::fill_(_iME, _bin, _w);
  }

  void MESetDet2D::fill_(unsigned _iME, int _bin, double _y, double _w) {
    if (kind_ == MonitorElement::Kind::TPROFILE2D) {
      MonitorElement *me(mes_.at(_iME));
      if (me->getBinEntries(_bin) < 0.) {
        me->setBinContent(_bin, 0.);
        me->setBinEntries(_bin, 0.);
        me->getTProfile2D()->SetEntries(me->getTProfile2D()->GetEntries() + 1.);
      }
    }

    MESet::fill_(_iME, _bin, _y, _w);
  }

  void MESetDet2D::fill_(unsigned _iME, double _x, double _wy, double _w) {
    if (kind_ == MonitorElement::Kind::TPROFILE2D) {
      MonitorElement *me(mes_.at(_iME));
      int bin(me->getTProfile2D()->FindBin(_x, _wy));
      if (me->getBinEntries(bin) < 0.) {
        me->setBinContent(bin, 0.);
        me->setBinEntries(bin, 0.);
        me->getTProfile2D()->SetEntries(me->getTProfile2D()->GetEntries() + 1.);
      }
    }

    MESet::fill_(_iME, _x, _wy, _w);
  }
}  // namespace ecaldqm
