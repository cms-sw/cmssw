#ifndef MESetTrend_H
#define MESetTrend_H

#include "MESetEcal.h"

namespace ecaldqm
{
  /* class MESetTrend
     time on xaxis
     channel id is used to identify the plot
  */

  class MESetTrend : public MESetEcal
  {
  public :
    MESetTrend(MEData const&);
    ~MESetTrend();

    void book();

    void fill(DetId const&, double, double _wy = 1., double _w = 1.);
    void fill(EcalElectronicsId const&, double, double _wy = 1., double _w = 1.);
    void fill(unsigned, double, double _wy = 1., double _w = 1.);
    void fill(double, double _wy = 1., double _w = 1.);

    int findBin(DetId const&, double, double _y = 0.) const;
    int findBin(EcalElectronicsId const&, double, double _y = 0.) const;
    int findBin(unsigned, double, double _y = 0.) const;
    int findBin(double, double _y = 0.) const;

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
