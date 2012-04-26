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

    void fill(DetId const&, float _w = 1., float _unused1 = 0., float _unused2 = 0.);
  };
}

#endif
