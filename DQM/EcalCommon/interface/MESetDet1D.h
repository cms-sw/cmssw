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
    MESetDet1D(std::string const&, BinService::ObjectType, BinService::BinningType, MonitorElement::Kind, BinService::AxisSpecs const* = 0);
    MESetDet1D(MESetDet1D const&);
    ~MESetDet1D();

    MESet* clone() const;

    void book();

    void fill(DetId const&, double _wy = 1., double _w = 1., double _unused = 0.);
    void fill(EcalElectronicsId const&, double _wy = 1., double _w = 1., double _unused = 0.);
    void fill(unsigned, double _wy = 1., double _w = 1., double _unused = 0.);

    void setBinContent(DetId const&, double);
    void setBinContent(EcalElectronicsId const&, double);
    void setBinContent(unsigned, double);
    void setBinContent(DetId const&, int, double);
    void setBinContent(EcalElectronicsId const&, int, double);
    void setBinContent(unsigned, int, double);

    void setBinError(DetId const&, double);
    void setBinError(EcalElectronicsId const&, double);
    void setBinError(unsigned, double);
    void setBinError(DetId const&, int, double);
    void setBinError(EcalElectronicsId const&, int, double);
    void setBinError(unsigned, int, double);

    void setBinEntries(DetId const&, double);
    void setBinEntries(EcalElectronicsId const&, double);
    void setBinEntries(unsigned, double);
    void setBinEntries(DetId const&, int, double);
    void setBinEntries(EcalElectronicsId const&, int, double);
    void setBinEntries(unsigned, int, double);

    double getBinContent(DetId const&, int _bin = 0) const;
    double getBinContent(EcalElectronicsId const&, int _bin = 0) const;
    double getBinContent(unsigned, int _bin = 0) const;

    double getBinError(DetId const&, int _bin = 0) const;
    double getBinError(EcalElectronicsId const&, int _bin = 0) const;
    double getBinError(unsigned, int _bin = 0) const;

    double getBinEntries(DetId const&, int _bin = 0) const;
    double getBinEntries(EcalElectronicsId const&, int _bin = 0) const;
    double getBinEntries(unsigned, int _bin = 0) const;

    int findBin(DetId const&) const;
    int findBin(EcalElectronicsId const&) const;
    int findBin(unsigned) const;
    int findBin(DetId const&, double, double _unused = 0.) const;
    int findBin(EcalElectronicsId const&, double, double _unused = 0.) const;
    int findBin(unsigned, double, double _unused = 0.) const;

    void reset(double _content = 0., double _err = 0., double _entries = 0.);
  };
}

#endif
