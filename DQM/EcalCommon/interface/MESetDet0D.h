#ifndef MESetDet0D_H
#define MESetDet0D_H

#include "MESetEcal.h"

namespace ecaldqm
{
  class MESetDet0D : public MESetEcal
  {
  public :
    MESetDet0D(std::string const&, MEData const&, bool _readOnly = false);
    ~MESetDet0D();

    void fill(DetId const&, double, double _unused1 = 0., double _unused2 = 0.);
    void fill(unsigned, double, double _unused1 = 0., double _unused2 = 0.);
    void fill(double, double _unused1 = 0., double _unused = 0.);
  };
}

#endif
