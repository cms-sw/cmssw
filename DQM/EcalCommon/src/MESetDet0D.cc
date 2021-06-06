#include "DQM/EcalCommon/interface/MESetDet0D.h"

namespace ecaldqm {
  MESetDet0D::MESetDet0D(std::string const &_fullPath,
                         binning::ObjectType _otype,
                         binning::BinningType _btype,
                         MonitorElement::Kind _kind)
      : MESetEcal(_fullPath, _otype, _btype, _kind, 0, nullptr, nullptr) {
    switch (kind_) {
      case MonitorElement::Kind::REAL:
        break;
      default:
        throw_("Unsupported MonitorElement kind");
    }
  }

  MESetDet0D::MESetDet0D(MESetDet0D const &_orig) : MESetEcal(_orig) {}

  MESetDet0D::~MESetDet0D() {}

  MESet *MESetDet0D::clone(std::string const &_path /* = ""*/) const {
    std::string path(path_);
    if (!_path.empty())
      path_ = _path;
    MESet *copy(new MESetDet0D(*this));
    path_ = path;
    return copy;
  }

  void MESetDet0D::fill(EcalDQMSetupObjects const edso, DetId const &_id, double _value, double, double) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(edso.electronicsMap, otype_, _id));
    checkME_(iME);

    mes_[iME]->Fill(_value);
  }

  void MESetDet0D::fill(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, double _value, double, double) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(edso.electronicsMap, otype_, _id));
    checkME_(iME);

    mes_[iME]->Fill(_value);
  }

  void MESetDet0D::fill(EcalDQMSetupObjects const edso, int _dcctccid, double _value, double, double) {
    if (!active_)
      return;

    unsigned iME(binning::findPlotIndex(edso.electronicsMap, otype_, _dcctccid, btype_));
    checkME_(iME);

    mes_[iME]->Fill(_value);
  }

  double MESetDet0D::getBinContent(EcalDQMSetupObjects const edso, DetId const &_id, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(edso.electronicsMap, otype_, _id));
    checkME_(iME);

    return mes_[iME]->getFloatValue();
  }

  double MESetDet0D::getBinContent(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(edso.electronicsMap, otype_, _id));
    checkME_(iME);

    return mes_[iME]->getFloatValue();
  }

  double MESetDet0D::getBinContent(EcalDQMSetupObjects const edso, int _dcctccid, int) const {
    if (!active_)
      return 0.;

    unsigned iME(binning::findPlotIndex(edso.electronicsMap, otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getFloatValue();
  }

  void MESetDet0D::reset(EcalElectronicsMapping const *electronicsMap, double _value /* = 0.*/, double, double) {
    unsigned nME(mes_.size());
    for (unsigned iME(0); iME < nME; iME++)
      mes_[iME]->Fill(_value);
  }
}  // namespace ecaldqm
