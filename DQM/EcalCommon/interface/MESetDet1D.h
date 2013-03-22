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
    void fill(DetId const&, double _wy = 1., double _w = 1., double _unused = 0.);
    void fill(unsigned, double _wy = 1., double _w = 1., double _unused = 0.);

    double getBinContent(DetId const&, int _bin = 0) const;
    double getBinContent(unsigned, int _bin = 0) const;

    double getBinError(DetId const&, int _bin = 0) const;
    double getBinError(unsigned, int _bin = 0) const;

    double getBinEntries(DetId const&, int _bin = 0) const;
    double getBinEntries(unsigned, int _bin = 0) const;

  private:
    void find_(uint32_t) const;
    void fill_(double, double);
  };
}

#endif
