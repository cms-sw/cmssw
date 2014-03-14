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
    MESetTrend(std::string const&, binning::ObjectType, binning::BinningType, MonitorElement::Kind, binning::AxisSpecs const* = 0, binning::AxisSpecs const* = 0);
    MESetTrend(MESetTrend const&);
    ~MESetTrend() {}

    MESet& operator=(MESet const&) override;

    MESet* clone(std::string const& = "") const override;

    void book(DQMStore&) override;
    void book(DQMStore::IBooker&) override;

    void fill(DetId const&, double, double = 1., double = 1.) override;
    void fill(EcalElectronicsId const&, double, double = 1., double = 1.) override;
    void fill(int, double, double = 1., double = 1.) override;
    void fill(double, double = 1., double = 1.) override;

    int findBin(DetId const&, double, double = 0.) const override;
    int findBin(EcalElectronicsId const&, double, double = 0.) const override;
    int findBin(int, double, double = 0.) const override;
    int findBin(double, double = 0.) const;

    bool isVariableBinning() const override { return true; }

    void setMinutely() { minutely_ = true; }
    void setShiftAxis() { shiftAxis_ = true; }
    void setCumulative();
    bool isMinutely() const { return minutely_; }
    bool canShiftAxis() const { return shiftAxis_; }
    bool isCumulative() const { return currentBin_ > 0; }

  private:
    template<class Bookable> void doBook_(Bookable&);
    bool shift_(unsigned);

    bool minutely_; // if true, bins in minutes instead of lumis
    bool shiftAxis_; // if true, shift x values
    int currentBin_; // only used for cumulative case
  };
}

#endif
