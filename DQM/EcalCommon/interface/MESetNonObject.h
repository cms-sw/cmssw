#ifndef MESetNonObject_H
#define MESetNonObject_H

#include "MESet.h"

namespace ecaldqm {
  class MESetNonObject : public MESet {
  public:
    MESetNonObject(std::string const &,
                   binning::ObjectType,
                   binning::BinningType,
                   MonitorElement::Kind,
                   binning::AxisSpecs const * = nullptr,
                   binning::AxisSpecs const * = nullptr,
                   binning::AxisSpecs const * = nullptr);
    MESetNonObject(MESetNonObject const &);
    ~MESetNonObject() override;

    MESet &operator=(MESet const &) override;

    MESet *clone(std::string const & = "") const override;

    void book(DQMStore::IBooker &, EcalElectronicsMapping const *) override;
    bool retrieve(EcalElectronicsMapping const *, DQMStore::IGetter &, std::string * = nullptr) const override;

    void fill(EcalDQMSetupObjects const, double, double = 1., double = 1.) override;

    void setBinContent(EcalDQMSetupObjects const, int, double) override;

    void setBinError(EcalDQMSetupObjects const, int, double) override;

    void setBinEntries(EcalDQMSetupObjects const, int, double) override;

    double getBinContent(EcalDQMSetupObjects const, int, int = 0) const override;

    double getFloatValue() const;

    double getBinError(EcalDQMSetupObjects const, int, int = 0) const override;

    double getBinEntries(EcalDQMSetupObjects const, int, int = 0) const override;

    int findBin(EcalDQMSetupObjects const, double, double = 0.) const;

    bool isVariableBinning() const override;

  protected:
    binning::AxisSpecs const *xaxis_;
    binning::AxisSpecs const *yaxis_;
    binning::AxisSpecs const *zaxis_;
  };
}  // namespace ecaldqm

#endif
