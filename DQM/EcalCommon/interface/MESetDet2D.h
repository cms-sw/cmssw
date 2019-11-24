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

    void book(DQMStore::IBooker &) override;

    void fill(DetId const &, double = 1., double = 0., double = 0.) override;
    void fill(EcalElectronicsId const &, double = 1., double = 0., double = 0.) override;
    void fill(int, double = 1., double = 1., double = 1.) override;

    using MESetEcal::setBinContent;
    void setBinContent(DetId const &, double) override;
    void setBinContent(EcalElectronicsId const &, double) override;
    void setBinContent(int, double) override;

    using MESetEcal::setBinError;
    void setBinError(DetId const &, double) override;
    void setBinError(EcalElectronicsId const &, double) override;
    void setBinError(int, double) override;

    using MESetEcal::setBinEntries;
    void setBinEntries(DetId const &, double) override;
    void setBinEntries(EcalElectronicsId const &, double) override;
    void setBinEntries(int, double) override;

    using MESetEcal::getBinContent;
    double getBinContent(DetId const &, int = 0) const override;
    double getBinContent(EcalElectronicsId const &, int = 0) const override;
    double getBinContent(int, int = 0) const override;

    using MESetEcal::getBinError;
    double getBinError(DetId const &, int = 0) const override;
    double getBinError(EcalElectronicsId const &, int = 0) const override;
    double getBinError(int, int = 0) const override;

    using MESetEcal::getBinEntries;
    double getBinEntries(DetId const &, int = 0) const override;
    double getBinEntries(EcalElectronicsId const &, int = 0) const override;
    double getBinEntries(int, int) const override;

    using MESetEcal::findBin;
    int findBin(DetId const &) const;
    int findBin(EcalElectronicsId const &) const;

    void reset(double = 0., double = 0., double = 0.) override;

  protected:
    void fill_(unsigned, int, double) override;
    void fill_(unsigned, int, double, double) override;
    void fill_(unsigned, double, double, double) override;
  };
}  // namespace ecaldqm

#endif
