#include "DQM/EcalCommon/interface/MESetDet0D.h"

namespace ecaldqm
{

  MESetDet0D::MESetDet0D(MEData const& _data) :
    MESetEcal(_data, 0)
  {
    switch(data_->kind){
    case MonitorElement::DQM_KIND_REAL:
      break;
    default:
      throw_("Unsupported MonitorElement kind");
    }
  }

  MESetDet0D::~MESetDet0D()
  {
  }

  void
  MESetDet0D::fill(DetId const& _id, double _value, double, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    mes_[iME]->Fill(_value);
  }

  void
  MESetDet0D::fill(EcalElectronicsId const& _id, double _value, double, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    mes_[iME]->Fill(_value);
  }

  void
  MESetDet0D::fill(unsigned _dcctccid, double _value, double, double)
  {
    if(!active_) return;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
    checkME_(iME);

    mes_[iME]->Fill(_value);
  }

  double
  MESetDet0D::getBinContent(DetId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getFloatValue();
  }

  double
  MESetDet0D::getBinContent(EcalElectronicsId const& _id, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _id));
    checkME_(iME);

    return mes_[iME]->getFloatValue();
  }

  double
  MESetDet0D::getBinContent(unsigned _dcctccid, int) const
  {
    if(!active_) return 0.;

    unsigned iME(binService_->findPlot(data_->otype, _dcctccid, data_->btype));
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
