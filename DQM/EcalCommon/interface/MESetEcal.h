#ifndef MESetEcal_H
#define MESetEcal_H

#include "MESet.h"

namespace ecaldqm
{

  /* class MESetEcal
     implements plot <-> detector part relationship
     base class for channel-binned histograms
     MESetEcal is only filled given an object identifier and a bin (channel id does not give a bin)
  */

  class MESetEcal : public MESet
  {
  public :
    MESetEcal(std::string const&, binning::ObjectType, binning::BinningType, MonitorElement::Kind, unsigned, binning::AxisSpecs const* = 0, binning::AxisSpecs const* = 0, binning::AxisSpecs const* = 0);
    MESetEcal(MESetEcal const&);
    ~MESetEcal();

    MESet& operator=(MESet const&) override;

    MESet* clone(std::string const& = "") const override;

    void book(DQMStore&) override;
    void book(DQMStore::IBooker&) override;
    bool retrieve(DQMStore const&, std::string* = 0) const override;

    void fill(DetId const&, double = 1., double = 1., double = 1.) override;
    void fill(EcalElectronicsId const&, double = 1., double = 1., double = 1.) override;
    void fill(int, double = 1., double = 1., double = 1.) override;
    void fill(double, double = 1., double = 1.) override;

    void setBinContent(DetId const&, int, double) override;
    void setBinContent(EcalElectronicsId const&, int, double) override;
    void setBinContent(int, int, double) override;

    void setBinError(DetId const&, int, double) override;
    void setBinError(EcalElectronicsId const&, int, double) override;
    void setBinError(int, int, double) override;

    void setBinEntries(DetId const&, int, double) override;
    void setBinEntries(EcalElectronicsId const&, int, double) override;
    void setBinEntries(int, int, double) override;

    double getBinContent(DetId const&, int) const override;
    double getBinContent(EcalElectronicsId const&, int) const override;
    double getBinContent(int, int) const override;

    double getBinError(DetId const&, int) const override;
    double getBinError(EcalElectronicsId const&, int) const override;
    double getBinError(int, int) const override;

    double getBinEntries(DetId const&, int) const override;
    double getBinEntries(EcalElectronicsId const&, int) const override;
    double getBinEntries(int, int) const override;

    virtual int findBin(DetId const&, double, double = 0.) const;
    virtual int findBin(EcalElectronicsId const&, double, double = 0.) const;
    virtual int findBin(int, double, double = 0.) const;

    bool isVariableBinning() const override;

    std::vector<std::string> generatePaths() const;

  protected :
    unsigned logicalDimensions_;
    binning::AxisSpecs const* xaxis_;
    binning::AxisSpecs const* yaxis_;
    binning::AxisSpecs const* zaxis_;

  private:
    template<class Bookable> void doBook_(Bookable&);
  };

}

#endif
