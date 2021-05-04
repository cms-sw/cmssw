#ifndef MESetDet2D_H
#define MESetDet2D_H

#include "MESetEcal.h"

namespace ecaldqm {
  /* class MESetDet2D
   channel ID-based MonitorElement wrapper
   channel id <-> 2D cell
*/
  class MESetDet2D : public MESetEcal {
  public:
    MESetDet2D(std::string const &,
               binning::ObjectType,
               binning::BinningType,
               MonitorElement::Kind,
               binning::AxisSpecs const * = nullptr);
    MESetDet2D(MESetDet2D const &);
    ~MESetDet2D() override;

    MESet *clone(std::string const & = "") const override;

    void book(DQMStore::IBooker &, EcalElectronicsMapping const *) override;

    void fill(EcalDQMSetupObjects const, DetId const &, double = 1., double = 0., double = 0.) override;
    void fill(EcalDQMSetupObjects const, EcalElectronicsId const &, double = 1., double = 0., double = 0.) override;
    void fill(EcalDQMSetupObjects const, int, double = 1., double = 1., double = 1.) override;

    using MESetEcal::setBinContent;
    void setBinContent(EcalDQMSetupObjects const, DetId const &, double) override;
    void setBinContent(EcalDQMSetupObjects const, EcalElectronicsId const &, double) override;
    void setBinContent(EcalDQMSetupObjects const, int, double) override;

    using MESetEcal::setBinError;
    void setBinError(EcalDQMSetupObjects const, DetId const &, double) override;
    void setBinError(EcalDQMSetupObjects const, EcalElectronicsId const &, double) override;
    void setBinError(EcalDQMSetupObjects const, int, double) override;

    using MESetEcal::setBinEntries;
    void setBinEntries(EcalDQMSetupObjects const, DetId const &, double) override;
    void setBinEntries(EcalDQMSetupObjects const, EcalElectronicsId const &, double) override;
    void setBinEntries(EcalDQMSetupObjects const, int, double) override;

    using MESetEcal::getBinContent;
    double getBinContent(EcalDQMSetupObjects const, DetId const &, int = 0) const override;
    double getBinContent(EcalDQMSetupObjects const, EcalElectronicsId const &, int = 0) const override;
    double getBinContent(EcalDQMSetupObjects const, int, int = 0) const override;

    using MESetEcal::getBinError;
    double getBinError(EcalDQMSetupObjects const, DetId const &, int = 0) const override;
    double getBinError(EcalDQMSetupObjects const, EcalElectronicsId const &, int = 0) const override;
    double getBinError(EcalDQMSetupObjects const, int, int = 0) const override;

    using MESetEcal::getBinEntries;
    double getBinEntries(EcalDQMSetupObjects const, DetId const &, int = 0) const override;
    double getBinEntries(EcalDQMSetupObjects const, EcalElectronicsId const &, int = 0) const override;
    double getBinEntries(EcalDQMSetupObjects const, int, int) const override;

    using MESetEcal::findBin;
    int findBin(EcalDQMSetupObjects const, DetId const &) const;
    int findBin(EcalDQMSetupObjects const, EcalElectronicsId const &) const;

    void reset(EcalElectronicsMapping const *, double = 0., double = 0., double = 0.) override;

  protected:
    void fill_(unsigned, int, double) override;
    void fill_(unsigned, int, double, double) override;
    void fill_(unsigned, double, double, double) override;
  };
}  // namespace ecaldqm

#endif
