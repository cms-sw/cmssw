#ifndef MESetProjection_H
#define MESetProjection_H

#include "MESetEcal.h"

namespace ecaldqm
{

  /* class MESetProjection
     MonitorElement wrapper for projection type 1D MEs
  */

  class MESetProjection : public MESetEcal
  {
  public :
    MESetProjection(std::string const&, BinService::ObjectType, BinService::BinningType, MonitorElement::Kind, BinService::AxisSpecs const* = 0);
    MESetProjection(MESetProjection const&);
    ~MESetProjection();

    MESet* clone() const;

    void fill(DetId const&, double = 1., double _unused1 = 0., double _unused2 = 0.);
    void fill(double, double = 1., double _unused = 0.);

    void setBinContent(DetId const&, double);

    void setBinError(DetId const&, double);

    void setBinEntries(DetId const&, double);

    double getBinContent(DetId const&, int _unused = 0) const;

    double getBinError(DetId const&, int _unused = 0) const;

    double getBinEntries(DetId const&, int _unused = 0) const;

  };
}

#endif
