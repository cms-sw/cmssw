#ifndef MESetProjection_H
#define MESetProjection_H

#include "MESetEcal.h"

namespace ecaldqm {

  /* class MESetProjection
   MonitorElement wrapper for projection type 1D MEs
*/

  class MESetProjection : public MESetEcal {
  public:
    MESetProjection(std::string const &,
                    binning::ObjectType,
                    binning::BinningType,
                    MonitorElement::Kind,
                    binning::AxisSpecs const * = nullptr);
    MESetProjection(MESetProjection const &);
    ~MESetProjection() override;

    MESet *clone(std::string const & = "") const override;

    void fill(EcalDQMSetupObjects const, DetId const &, double = 1., double = 0., double = 0.) override;
    void fill(EcalDQMSetupObjects const, int, double = 1., double = 1., double = 0.) override;
    void fill(EcalDQMSetupObjects const, double, double = 1., double = 0.) override;

    using MESetEcal::setBinContent;
    void setBinContent(EcalDQMSetupObjects const, DetId const &, double) override;

    using MESetEcal::setBinError;
    void setBinError(EcalDQMSetupObjects const, DetId const &, double) override;

    using MESetEcal::setBinEntries;
    void setBinEntries(EcalDQMSetupObjects const, DetId const &, double) override;

    using MESetEcal::getBinContent;
    double getBinContent(EcalDQMSetupObjects const, DetId const &, int = 0) const override;

    using MESetEcal::getBinError;
    double getBinError(EcalDQMSetupObjects const, DetId const &, int = 0) const override;

    using MESetEcal::getBinEntries;
    double getBinEntries(EcalDQMSetupObjects const, DetId const &, int = 0) const override;
  };
}  // namespace ecaldqm

#endif
