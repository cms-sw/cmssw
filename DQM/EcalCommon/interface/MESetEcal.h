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
    MESetEcal(std::string const&, BinService::ObjectType, BinService::BinningType, MonitorElement::Kind, unsigned, BinService::AxisSpecs const* = 0, BinService::AxisSpecs const* = 0, BinService::AxisSpecs const* = 0);
    MESetEcal(MESetEcal const&);
    ~MESetEcal();

    MESet& operator=(MESet const&);

    MESet* clone() const;

    void book();
    bool retrieve() const;

    void fill(DetId const&, double = 1., double = 1., double = 1.);
    void fill(EcalElectronicsId const&, double = 1., double = 1., double = 1.);
    void fill(unsigned, double = 1., double = 1., double = 1.);
    void fill(double, double = 1., double = 1.);

    void setBinContent(DetId const&, int, double);
    void setBinContent(EcalElectronicsId const&, int, double);
    void setBinContent(unsigned, int, double);

    void setBinError(DetId const&, int, double);
    void setBinError(EcalElectronicsId const&, int, double);
    void setBinError(unsigned, int, double);

    void setBinEntries(DetId const&, int, double);
    void setBinEntries(EcalElectronicsId const&, int, double);
    void setBinEntries(unsigned, int, double);

    double getBinContent(DetId const&, int) const;
    double getBinContent(EcalElectronicsId const&, int) const;
    double getBinContent(unsigned, int) const;

    double getBinError(DetId const&, int) const;
    double getBinError(EcalElectronicsId const&, int) const;
    double getBinError(unsigned, int) const;

    double getBinEntries(DetId const&, int) const;
    double getBinEntries(EcalElectronicsId const&, int) const;
    double getBinEntries(unsigned, int) const;

    int findBin(DetId const&, double, double = 0.) const;
    int findBin(EcalElectronicsId const&, double, double = 0.) const;
    int findBin(unsigned, double, double = 0.) const;

    bool isVariableBinning() const;

    std::vector<std::string> generatePaths() const;

  protected :
    unsigned logicalDimensions_;
    BinService::AxisSpecs const* xaxis_;
    BinService::AxisSpecs const* yaxis_;
    BinService::AxisSpecs const* zaxis_;
  };

}

#endif
