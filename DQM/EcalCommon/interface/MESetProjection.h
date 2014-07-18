#ifndef MESetProjection_H
#define MESetProjection_H

#include "MESetEcal.h"

namespace ecaldqm
{

  /* class MESetProjection
     MonitorElement wrapper for projection type 1D MEs
  */

  class MESetProjection : public MESetEcal {
  public :
    MESetProjection(std::string const&, binning::ObjectType, binning::BinningType, MonitorElement::Kind, binning::AxisSpecs const* = 0);
    MESetProjection(MESetProjection const&);
    ~MESetProjection();

    MESet* clone(std::string const& = "") const override;

    void fill(DetId const&, double = 1., double = 0., double = 0.) override;
    void fill(int, double = 1., double = 1., double = 0.) override;
    void fill(double, double = 1., double = 0.) override;

    void setBinContent(DetId const&, double) override;

    void setBinError(DetId const&, double) override;

    void setBinEntries(DetId const&, double) override;

    double getBinContent(DetId const&, int = 0) const override;

    double getBinError(DetId const&, int = 0) const override;

    double getBinEntries(DetId const&, int = 0) const override;
  };
}

#endif
