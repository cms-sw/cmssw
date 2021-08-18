#ifndef MESetDet0D_H
#define MESetDet0D_H

#include "MESetEcal.h"

namespace ecaldqm {
  /* class MESetDet0D
   subdetector-based MonitorElement wrapper
   represents single float MEs (DQM_KIND_REAL)
   fill = setBinContent
*/

  class MESetDet0D : public MESetEcal {
  public:
    MESetDet0D(std::string const &, binning::ObjectType, binning::BinningType, MonitorElement::Kind);
    MESetDet0D(MESetDet0D const &);
    ~MESetDet0D() override;

    MESet *clone(std::string const & = "") const override;

    void fill(EcalDQMSetupObjects const, DetId const &, double, double = 0., double = 0.) override;
    void fill(EcalDQMSetupObjects const, EcalElectronicsId const &, double, double = 0., double = 0.) override;
    void fill(EcalDQMSetupObjects const, int, double, double = 0., double = 0.) override;

    void setBinContent(EcalDQMSetupObjects const edso, DetId const &_id, int, double _value) override {
      fill(edso, _id, _value);
    }
    void setBinContent(EcalDQMSetupObjects const edso, EcalElectronicsId const &_id, int, double _value) override {
      fill(edso, _id, _value);
    }
    void setBinContent(EcalDQMSetupObjects const edso, int _dcctccid, int, double _value) override {
      fill(edso, _dcctccid, _value);
    }

    double getBinContent(EcalDQMSetupObjects const, DetId const &, int = 0) const override;
    double getBinContent(EcalDQMSetupObjects const, EcalElectronicsId const &, int = 0) const override;
    double getBinContent(EcalDQMSetupObjects const, int, int = 0) const override;

    void reset(EcalElectronicsMapping const *, double = 0., double = 0., double = 0.) override;
  };
}  // namespace ecaldqm

#endif
