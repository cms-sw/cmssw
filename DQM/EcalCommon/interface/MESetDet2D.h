#ifndef MESetDet2D_H
#define MESetDet2D_H

#include "MESetEcal.h"

namespace ecaldqm
{
  /* class MESetDet2D
     channel ID-based MonitorElement wrapper
     channel id <-> 2D cell
  */
  class MESetDet2D : public MESetEcal
  {
  public :
    MESetDet2D(std::string const&, binning::ObjectType, binning::BinningType, MonitorElement::Kind, binning::AxisSpecs const* = 0);
    MESetDet2D(MESetDet2D const&);
    ~MESetDet2D();

    MESet* clone(std::string const& = "") const override;

    void book(DQMStore&) override;
    void book(DQMStore::IBooker&) override;

    void fill(DetId const&, double = 1., double = 0., double = 0.) override;
    void fill(EcalElectronicsId const&, double = 1., double = 0., double = 0.) override;
    void fill(int, double = 1., double = 1., double = 1.) override;

    void setBinContent(DetId const&, double) override;
    void setBinContent(EcalElectronicsId const&, double) override;
    void setBinContent(int, double) override;

    void setBinError(DetId const&, double) override;
    void setBinError(EcalElectronicsId const&, double) override;
    void setBinError(int, double) override;

    void setBinEntries(DetId const&, double) override;
    void setBinEntries(EcalElectronicsId const&, double) override;
    void setBinEntries(int, double) override;

    double getBinContent(DetId const&, int = 0) const override;
    double getBinContent(EcalElectronicsId const&, int = 0) const override;
    double getBinContent(int, int = 0) const override;

    double getBinError(DetId const&, int = 0) const override;
    double getBinError(EcalElectronicsId const&, int = 0) const override;
    double getBinError(int, int = 0) const override;

    double getBinEntries(DetId const&, int = 0) const override;
    double getBinEntries(EcalElectronicsId const&, int = 0) const override;
    double getBinEntries(int, int) const override;

    int findBin(DetId const&) const;
    int findBin(EcalElectronicsId const&) const;

    void reset(double = 0., double = 0., double = 0.) override;

    void softReset() override;

  protected:
    void fill_(unsigned, int, double) override;
    void fill_(unsigned, int, double, double) override;
    void fill_(unsigned, double, double, double) override;

  private:
    template<class Bookable> void doBook_(Bookable&);
  };
}

#endif
