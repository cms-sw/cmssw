#ifndef MESetDet1D_H
#define MESetDet1D_H

#include "MESetEcal.h"

namespace ecaldqm
{
  /*
    class MESetDet1D
    channel ID-based MonitorElement wrapper
    channel id <-> x axis bin
  */

  class MESetDet1D : public MESetEcal
  {
  public :
    MESetDet1D(std::string const&, binning::ObjectType, binning::BinningType, MonitorElement::Kind, binning::AxisSpecs const* = 0);
    MESetDet1D(MESetDet1D const&);
    ~MESetDet1D();

    MESet* clone(std::string const& = "") const override;

    void book(DQMStore&) override;
    void book(DQMStore::IBooker&) override;

    void fill(DetId const&, double = 1., double = 1., double = 0.) override;
    void fill(EcalElectronicsId const&, double = 1., double = 1., double = 0.) override;
    void fill(int, double = 1., double = 1., double = 0.) override;

    void setBinContent(DetId const&, double) override;
    void setBinContent(EcalElectronicsId const&, double) override;
    void setBinContent(int, double) override;
    void setBinContent(DetId const&, int, double) override;
    void setBinContent(EcalElectronicsId const&, int, double) override;
    void setBinContent(int, int, double) override;

    void setBinError(DetId const&, double) override;
    void setBinError(EcalElectronicsId const&, double) override;
    void setBinError(int, double) override;
    void setBinError(DetId const&, int, double) override;
    void setBinError(EcalElectronicsId const&, int, double) override;
    void setBinError(int, int, double) override;

    void setBinEntries(DetId const&, double) override;
    void setBinEntries(EcalElectronicsId const&, double) override;
    void setBinEntries(int, double) override;
    void setBinEntries(DetId const&, int, double) override;
    void setBinEntries(EcalElectronicsId const&, int, double) override;
    void setBinEntries(int, int, double) override;

    double getBinContent(DetId const&, int = 0) const override;
    double getBinContent(EcalElectronicsId const&, int = 0) const override;
    double getBinContent(int, int = 0) const override;

    double getBinError(DetId const&, int = 0) const override;
    double getBinError(EcalElectronicsId const&, int = 0) const override;
    double getBinError(int, int = 0) const override;

    double getBinEntries(DetId const&, int = 0) const override;
    double getBinEntries(EcalElectronicsId const&, int = 0) const override;
    double getBinEntries(int, int = 0) const override;

    int findBin(DetId const&) const;
    int findBin(EcalElectronicsId const&) const;
    int findBin(int) const;
    int findBin(DetId const&, double, double = 0.) const override;
    int findBin(EcalElectronicsId const&, double, double = 0.) const override;
    int findBin(int, double, double = 0.) const override;

    void reset(double = 0., double = 0., double = 0.) override;

  private:
    template<class Bookable> void doBook_(Bookable&);
  };
}

#endif
