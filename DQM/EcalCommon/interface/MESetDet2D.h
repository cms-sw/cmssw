#ifndef MESetDet2D_H
#define MESetDet2D_H

#include "MESetEcal.h"

namespace ecaldqm
{
  class MESetDet2D : public MESetEcal
  {
  public :
    MESetDet2D(std::string const&, MEData const&, bool _readOnly = false);
    ~MESetDet2D();

    void fill(DetId const&, double _w = 1., double _unused1 = 0., double _unused2 = 0.);
  };
}

#endif
