#ifndef MESetTrend_H
#define MESetTrend_H

#include "MESetEcal.h"

// correct way is to have MESetDet which inherits from MESetEcal and implements fill functions
// this way MESetEcal can focus only on generateNames etc. (object properties)

namespace ecaldqm
{
  class MESetTrend : public MESetEcal
  {
  public :
    MESetTrend(std::string const&, MEData const&, bool _readOnly = false);
    ~MESetTrend();

    void book();

    void fill(DetId const&, double, double _wy = 1., double _w = 1.);
    void fill(unsigned, double, double _wy = 1., double _w = 1.);
    void fill(double, double _wy = 1., double _w = 1.);

    void setBinContent(DetId const&, double, double _err = 0.) {}
    void setBinContent(unsigned, double, double _err = 0.) {}

    void setBinEntries(DetId const&, double) {}
    void setBinEntries(unsigned, double) {}

    double getBinContent(DetId const&, int _bin = 0) const { return 0.; }
    double getBinContent(unsigned, int _bin = 0) const { return 0.; }

    double getBinError(DetId const&, int _bin = 0) const { return 0.; }
    double getBinError(unsigned, int _bin = 0) const { return 0.; }

    double getBinEntries(DetId const&, int _bin = 0) const { return 0.; }
    double getBinEntries(unsigned, int _bin = 0) const { return 0.; }

    void setTimeZero(time_t _t0) { t0_ = _t0; }
    time_t getTimeZero() const { return t0_; }
    void setMinutely(bool _minutely) { minutely_ = _minutely; }
    bool getMinutely() const { return minutely_; }

  private:
    bool shift_(time_t);

    time_t t0_;
    bool minutely_;

    time_t tLow_;
  };
}

#endif
