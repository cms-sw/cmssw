#ifndef MESetDet0D_H
#define MESetDet0D_H

#include "MESetEcal.h"

namespace ecaldqm
{
  /* class MESetDet0D
     subdetector-based MonitorElement wrapper
     represents single float MEs (DQM_KIND_REAL)
     fill = setBinContent
  */

  class MESetDet0D : public MESetEcal
  {
  public :
    MESetDet0D(std::string const&, binning::ObjectType, binning::BinningType, MonitorElement::Kind);
    MESetDet0D(MESetDet0D const&);
    ~MESetDet0D();

    MESet* clone(std::string const& = "") const override;

    void fill(DetId const&, double, double = 0., double = 0.) override;
    void fill(EcalElectronicsId const&, double, double = 0., double = 0.) override;
    void fill(int, double, double = 0., double = 0.) override;

    void setBinContent(DetId const& _id, int, double _value) override { fill(_id, _value); }
    void setBinContent(EcalElectronicsId const& _id, int, double _value) override { fill(_id, _value); }
    void setBinContent(int _dcctccid, int, double _value) override { fill(_dcctccid, _value); }

    double getBinContent(DetId const&, int = 0) const override;
    double getBinContent(EcalElectronicsId const&, int = 0) const override;
    double getBinContent(int, int = 0) const override;

    void reset(double = 0., double = 0., double = 0.) override;
  };
}

#endif
