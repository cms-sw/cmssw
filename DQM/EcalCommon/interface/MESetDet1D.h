#ifndef MESetDet1D_H
#define MESetDet1D_H

#include "MESetEcal.h"

namespace ecaldqm
{
  class MESetDet1D : public MESetEcal
  {
  public :
    MESetDet1D(std::string const&, MEData const&, bool _readOnly = false);
    ~MESetDet1D();

    // fill using the private fill_
    void fill(DetId const&, float _wy = 1., float _w = 1., float _unused = 0.);
    void fill(unsigned, float _wy = 1., float _w = 1., float _unused = 0.);

    float getBinContent(DetId const&, int _bin = 0) const;
    float getBinContent(unsigned, int _bin = 0) const;

    float getBinError(DetId const&, int _bin = 0) const;
    float getBinError(unsigned, int _bin = 0) const;

    float getBinEntries(DetId const&, int _bin = 0) const;
    float getBinEntries(unsigned, int _bin = 0) const;

  private:
    void find_(uint32_t) const;
    void fill_(float, float);
  };
}

#endif
