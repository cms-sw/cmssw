#ifndef MESetDet1D_H
#define MESetDet1D_H

#include "MESetEcal.h"

namespace ecaldqm {
  /*
  class MESetDet1D
  channel ID-based MonitorElement wrapper
  channel id <-> x axis bin
*/

  class MESetDet1D : public MESetEcal {
  public:
    MESetDet1D(std::string const &,
               binning::ObjectType,
               binning::BinningType,
               MonitorElement::Kind,
               binning::AxisSpecs const * = nullptr);
    MESetDet1D(MESetDet1D const &);
    ~MESetDet1D() override;

    MESet *clone(std::string const & = "") const override;

    void book(DQMStore::IBooker &, EcalElectronicsMapping const *) override;

    void fill(EcalDQMSetupObjects const, DetId const &, double = 1., double = 1., double = 0.) override;
    void fill(EcalDQMSetupObjects const, EcalElectronicsId const &, double = 1., double = 1., double = 0.) override;
    void fill(EcalDQMSetupObjects const, int, double = 1., double = 1., double = 0.) override;

    void setBinContent(EcalDQMSetupObjects const, DetId const &, double) override;
    void setBinContent(EcalDQMSetupObjects const, EcalElectronicsId const &, double) override;
    void setBinContent(EcalDQMSetupObjects const, int, double) override;
    void setBinContent(EcalDQMSetupObjects const, DetId const &, int, double) override;
    void setBinContent(EcalDQMSetupObjects const, EcalElectronicsId const &, int, double) override;
    void setBinContent(EcalDQMSetupObjects const, int, int, double) override;

    void setBinError(EcalDQMSetupObjects const, DetId const &, double) override;
    void setBinError(EcalDQMSetupObjects const, EcalElectronicsId const &, double) override;
    void setBinError(EcalDQMSetupObjects const, int, double) override;
    void setBinError(EcalDQMSetupObjects const, DetId const &, int, double) override;
    void setBinError(EcalDQMSetupObjects const, EcalElectronicsId const &, int, double) override;
    void setBinError(EcalDQMSetupObjects const, int, int, double) override;

    void setBinEntries(EcalDQMSetupObjects const, DetId const &, double) override;
    void setBinEntries(EcalDQMSetupObjects const, EcalElectronicsId const &, double) override;
    void setBinEntries(EcalDQMSetupObjects const, int, double) override;
    void setBinEntries(EcalDQMSetupObjects const, DetId const &, int, double) override;
    void setBinEntries(EcalDQMSetupObjects const, EcalElectronicsId const &, int, double) override;
    void setBinEntries(EcalDQMSetupObjects const, int, int, double) override;

    double getBinContent(EcalDQMSetupObjects const, DetId const &, int = 0) const override;
    double getBinContent(EcalDQMSetupObjects const, EcalElectronicsId const &, int = 0) const override;
    double getBinContent(EcalDQMSetupObjects const, int, int = 0) const override;

    double getBinError(EcalDQMSetupObjects const, DetId const &, int = 0) const override;
    double getBinError(EcalDQMSetupObjects const, EcalElectronicsId const &, int = 0) const override;
    double getBinError(EcalDQMSetupObjects const, int, int = 0) const override;

    double getBinEntries(EcalDQMSetupObjects const, DetId const &, int = 0) const override;
    double getBinEntries(EcalDQMSetupObjects const, EcalElectronicsId const &, int = 0) const override;
    double getBinEntries(EcalDQMSetupObjects const, int, int = 0) const override;

    int findBin(EcalDQMSetupObjects const, DetId const &) const;
    int findBin(EcalDQMSetupObjects const, EcalElectronicsId const &) const;
    int findBin(EcalDQMSetupObjects const, int) const;
    int findBin(EcalDQMSetupObjects const, DetId const &, double, double = 0.) const override;
    int findBin(EcalDQMSetupObjects const, EcalElectronicsId const &, double, double = 0.) const override;
    int findBin(EcalDQMSetupObjects const, int, double, double = 0.) const override;

    void reset(EcalElectronicsMapping const *, double = 0., double = 0., double = 0.) override;
  };
}  // namespace ecaldqm

#endif
