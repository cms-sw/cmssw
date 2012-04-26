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

    void fill(DetId const&, float, float _unused1 = 0., float _unused2 = 0.);
    void fill(unsigned, float, float _unused1 = 0., float _unused2 = 0.);
    void fill(float, float _unused1 = 0., float _unused = 0.);
  };
}

#endif
