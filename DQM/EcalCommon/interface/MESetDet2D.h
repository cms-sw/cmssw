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
    MESetDet2D(MEData const&);
    ~MESetDet2D();

    void fill(DetId const&, double _w = 1., double _unused1 = 0., double _unused2 = 0.);
    void fill(EcalElectronicsId const&, double _w = 1., double _unused1 = 0., double _unused2 = 0.);
    void fill(unsigned, double, double, double) {}

    void setBinContent(DetId const&, double);
    void setBinContent(EcalElectronicsId const&, double);
    void setBinContent(unsigned, int, double) {}

    void setBinError(DetId const&, double);
    void setBinError(EcalElectronicsId const&, double);
    void setBinError(unsigned, int, double) {}

    void setBinEntries(DetId const&, double);
    void setBinEntries(EcalElectronicsId const&, double);
    void setBinEntries(unsigned, int, double) {}

    double getBinContent(DetId const&, int _unused = 0) const;
    double getBinContent(EcalElectronicsId const&, int _unused = 0) const;
    double getBinContent(unsigned, int) const { return 0.; }

    double getBinError(DetId const&, int _unused = 0) const;
    double getBinError(EcalElectronicsId const&, int _unused = 0) const;
    double getBinError(unsigned, int) const { return 0.; }

    double getBinEntries(DetId const&, int _unused = 0) const;
    double getBinEntries(EcalElectronicsId const&, int _unused = 0) const;
    double getBinEntries(unsigned, int) const { return 0.; }

    int findBin(DetId const&) const;
    int findBin(EcalElectronicsId const&) const;
    int findBin(unsigned, double, double = 0.) const { return 0; }

    void reset(double _content = 0., double _err = 0., double _entries = 0.);

  protected:
    void fill_(unsigned, int, double);
    void fill_(unsigned, int, double, double);
    void fill_(unsigned, double, double, double);
  };
}

#endif
