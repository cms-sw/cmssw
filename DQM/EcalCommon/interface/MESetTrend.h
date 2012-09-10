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
    MESetTrend(std::string const&, BinService::ObjectType, BinService::BinningType, MonitorElement::Kind, bool, bool, BinService::AxisSpecs const* = 0);
    MESetTrend(MESetTrend const&);
    ~MESetTrend();

    MESet& operator=(MESet const&);

    MESet* clone() const;

    void book();

    void fill(DetId const&, double, double _wy = 1., double _w = 1.);
    void fill(EcalElectronicsId const&, double, double _wy = 1., double _w = 1.);
    void fill(unsigned, double, double _wy = 1., double _w = 1.);
    void fill(double, double _wy = 1., double _w = 1.);

    int findBin(DetId const&, double, double _y = 0.) const;
    int findBin(EcalElectronicsId const&, double, double _y = 0.) const;
    int findBin(unsigned, double, double _y = 0.) const;
    int findBin(double, double _y = 0.) const;

    bool isMinutely() const { return minutely_; }
    bool isCumulative() const { return currentBin_ > 0; }

  private:
    bool shift_(unsigned);

    bool minutely_;
    int currentBin_; // only used for cumulative case
  };
}

#endif
