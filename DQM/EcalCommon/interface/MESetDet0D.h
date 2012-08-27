#ifndef MESetDet0D_H
#define MESetDet0D_H

#include "MESetEcal.h"

namespace ecaldqm
{
  /* class MESetDet0D
     subdetector-based MonitorElement wrapper
     represents single float MEs (DQM_KIND_REAL)
     fill = setBinContent
  */

  class MESetDet0D : public MESetEcal
  {
  public :
    MESetDet0D(std::string const&, BinService::ObjectType, BinService::BinningType, MonitorElement::Kind);
    MESetDet0D(MESetDet0D const&);
    ~MESetDet0D();

    MESet* clone() const;

    void fill(DetId const&, double, double _unused1 = 0., double _unused2 = 0.);
    void fill(EcalElectronicsId const&, double, double _unused1 = 0., double _unused2 = 0.);
    void fill(unsigned, double, double _unused1 = 0., double _unused2 = 0.);

    void setBinContent(DetId const& _id, int, double _value) { fill(_id, _value); }
    void setBinContent(EcalElectronicsId const& _id, int, double _value) { fill(_id, _value); }
    void setBinContent(unsigned _dcctccid, int, double _value) { fill(_dcctccid, _value); }

    void setBinError(DetId const&, int, double) {}
    void setBinError(EcalElectronicsId const&, int, double) {}
    void setBinError(unsigned, int, double) {}

    void setBinEntries(DetId const&, int, double) {}
    void setBinEntries(EcalElectronicsId const&, int, double) {}
    void setBinEntries(unsigned, int, double) {}

    double getBinContent(DetId const&, int _unused = 0) const;
    double getBinContent(EcalElectronicsId const&, int _unused = 0) const;
    double getBinContent(unsigned, int _unused = 0) const;

    double getBinError(DetId const&, int) const { return 0.; }
    double getBinError(EcalElectronicsId const&, int) const { return 0.; }
    double getBinError(unsigned, int) const { return 0.; }

    double getBinEntries(DetId const&, int) const { return 0.; }
    double getBinEntries(EcalElectronicsId const&, int) const { return 0.; }
    double getBinEntries(unsigned, int) const { return 0.; }

    void reset(double _content = 0., double _unused1 = 0., double _unused2 = 0.);
  };
}

#endif
