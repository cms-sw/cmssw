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
    MESetDet2D(std::string const&, BinService::ObjectType, BinService::BinningType, MonitorElement::Kind, BinService::AxisSpecs const* = 0);
    MESetDet2D(MESetDet2D const&);
    ~MESetDet2D();

    MESet* clone() const;

    void book();

    void fill(DetId const&, double = 1., double = 0., double = 0.);
    void fill(EcalElectronicsId const&, double = 1., double = 0., double = 0.);
    void fill(unsigned, double = 1., double = 1., double = 1.);

    void setBinContent(DetId const&, double);
    void setBinContent(EcalElectronicsId const&, double);
    void setBinContent(unsigned, double);

    void setBinError(DetId const&, double);
    void setBinError(EcalElectronicsId const&, double);
    void setBinError(unsigned, double);

    void setBinEntries(DetId const&, double);
    void setBinEntries(EcalElectronicsId const&, double);
    void setBinEntries(unsigned, double);

    double getBinContent(DetId const&, int = 0) const;
    double getBinContent(EcalElectronicsId const&, int = 0) const;
    double getBinContent(unsigned, int = 0) const;

    double getBinError(DetId const&, int = 0) const;
    double getBinError(EcalElectronicsId const&, int = 0) const;
    double getBinError(unsigned, int = 0) const;

    double getBinEntries(DetId const&, int = 0) const;
    double getBinEntries(EcalElectronicsId const&, int = 0) const;
    double getBinEntries(unsigned, int) const;

    int findBin(DetId const&) const;
    int findBin(EcalElectronicsId const&) const;
    int findBin(unsigned, double, double = 0.) const;

    void reset(double _content = 0., double _err = 0., double _entries = 0.);

    void softReset();

  protected:
    void fill_(unsigned, int, double);
    void fill_(unsigned, int, double, double);
    void fill_(unsigned, double, double, double);
  };
}

#endif
