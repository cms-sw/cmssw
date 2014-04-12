#include "DQM/EcalCommon/interface/MESetDet0D.h"

namespace ecaldqm
{
  MESetDet0D::MESetDet0D(std::string const& _fullPath, binning::ObjectType _otype, binning::BinningType _btype, MonitorElement::Kind _kind) :
    MESetEcal(_fullPath, _otype, _btype, _kind, 0, 0, 0)
  {
    switch(kind_){
    case MonitorElement::DQM_KIND_REAL:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }
  }

  MESetDet0D::MESetDet0D(MESetDet0D const& _orig) :
    MESetEcal(_orig)
  {
  }

  MESetDet0D::~MESetDet0D()
  {
  }

  MESet*
  MESetDet0D::clone(std::string const& _path/* = ""*/) const
  {
    std::string path(path_);
    if(_path != "") path_ = _path;
    MESet* copy(new MESetDet0D(*this));
    path_ = path;
    return copy;
  }

  void
  MESetDet0D::fill(DetId const& _id, double _value, double, double)
  {
    if(!active_) return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    mes_[iME]->Fill(_value);
  }

  void
  MESetDet0D::fill(EcalElectronicsId const& _id, double _value, double, double)
  {
    if(!active_) return;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    mes_[iME]->Fill(_value);
  }

  void
  MESetDet0D::fill(int _dcctccid, double _value, double, double)
  {
    if(!active_) return;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    mes_[iME]->Fill(_value);
  }

  double
  MESetDet0D::getBinContent(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getFloatValue();
  }

  double
  MESetDet0D::getBinContent(EcalElectronicsId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _id));
    checkME_(iME);

    return mes_[iME]->getFloatValue();
  }

  double
  MESetDet0D::getBinContent(int _dcctccid, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binning::findPlotIndex(otype_, _dcctccid, btype_));
    checkME_(iME);

    return mes_[iME]->getFloatValue();
  }

  void
  MESetDet0D::reset(double _value/* = 0.*/, double, double)
  {
    unsigned nME(mes_.size());
    for(unsigned iME(0); iME < nME; iME++)
      mes_[iME]->Fill(_value);
  }
}
