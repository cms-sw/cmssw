#ifndef MESetEcal_H
#define MESetEcal_H

#include "MESet.h"

namespace ecaldqm {

  /* class MESetEcal
   implements plot <-> detector part relationship
   base class for channel-binned histograms
   MESetEcal is only filled given an object identifier and a bin (channel id
   does not give a bin)
*/

  class MESetEcal : public MESet {
  public:
    MESetEcal(std::string const &,
              binning::ObjectType,
              binning::BinningType,
              MonitorElement::Kind,
              unsigned,
              binning::AxisSpecs const * = nullptr,
              binning::AxisSpecs const * = nullptr,
              binning::AxisSpecs const * = nullptr);
    MESetEcal(MESetEcal const &);
    ~MESetEcal() override;

    MESet &operator=(MESet const &) override;

    MESet *clone(std::string const & = "") const override;

    void book(DQMStore::IBooker &, EcalElectronicsMapping const *) override;
    bool retrieve(EcalElectronicsMapping const *, DQMStore::IGetter &, std::string * = nullptr) const override;

    void fill(EcalDQMSetupObjects const, DetId const &, double = 1., double = 1., double = 1.) override;
    void fill(EcalDQMSetupObjects const, EcalElectronicsId const &, double = 1., double = 1., double = 1.) override;
    void fill(EcalDQMSetupObjects const, int, double = 1., double = 1., double = 1.) override;
    void fill(EcalDQMSetupObjects const, double, double = 1., double = 1.) override;

    void setBinContent(EcalDQMSetupObjects const, DetId const &, int, double) override;
    void setBinContent(EcalDQMSetupObjects const, EcalElectronicsId const &, int, double) override;
    void setBinContent(EcalDQMSetupObjects const, int, int, double) override;

    void setBinError(EcalDQMSetupObjects const, DetId const &, int, double) override;
    void setBinError(EcalDQMSetupObjects const, EcalElectronicsId const &, int, double) override;
    void setBinError(EcalDQMSetupObjects const, int, int, double) override;

    void setBinEntries(EcalDQMSetupObjects const, DetId const &, int, double) override;
    void setBinEntries(EcalDQMSetupObjects const, EcalElectronicsId const &, int, double) override;
    void setBinEntries(EcalDQMSetupObjects const, int, int, double) override;

    double getBinContent(EcalDQMSetupObjects const, DetId const &, int) const override;
    double getBinContent(EcalDQMSetupObjects const, EcalElectronicsId const &, int) const override;
    double getBinContent(EcalDQMSetupObjects const, int, int) const override;

    double getBinError(EcalDQMSetupObjects const, DetId const &, int) const override;
    double getBinError(EcalDQMSetupObjects const, EcalElectronicsId const &, int) const override;
    double getBinError(EcalDQMSetupObjects const, int, int) const override;

    double getBinEntries(EcalDQMSetupObjects const, DetId const &, int) const override;
    double getBinEntries(EcalDQMSetupObjects const, EcalElectronicsId const &, int) const override;
    double getBinEntries(EcalDQMSetupObjects const, int, int) const override;

    virtual int findBin(EcalDQMSetupObjects const, DetId const &, double, double = 0.) const;
    virtual int findBin(EcalDQMSetupObjects const, EcalElectronicsId const &, double, double = 0.) const;
    virtual int findBin(EcalDQMSetupObjects const, int, double, double = 0.) const;

    bool isVariableBinning() const override;

    std::vector<std::string> generatePaths(EcalElectronicsMapping const *) const;

  protected:
    unsigned logicalDimensions_;
    binning::AxisSpecs const *xaxis_;
    binning::AxisSpecs const *yaxis_;
    binning::AxisSpecs const *zaxis_;

  private:
    template <class Bookable>
    void doBook_(Bookable &);
  };

}  // namespace ecaldqm

#endif
