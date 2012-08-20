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
    MESetEcal(MEData const&, int);
    ~MESetEcal();

    void book();
    bool retrieve() const;

    void fill(DetId const&, double _wx = 1., double _wy = 1., double _w = 1.);
    void fill(EcalElectronicsId const&, double _wx = 1., double _wy = 1., double _w = 1.);
    void fill(unsigned, double _wx = 1., double _wy = 1., double _w = 1.);

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

    std::vector<std::string> generateNames() const;

  protected :
    const unsigned logicalDimensions_;
  };

}

#endif
