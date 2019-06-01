//-*-C++-*-
//-*-Dummy.h-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////

#ifndef DUMMY_HH
#define DUMMY_HH

#include "GeneratorInterface/ExhumeInterface/interface/CrossSection.h"

namespace Exhume {

  class Dummy : public CrossSection {
  public:
    Dummy(const edm::ParameterSet&);
    double SubProcess() override;
    void SetPartons() override;
    void SetSubParameters() override;
    double SubParameterWeight() override;
    void MaximiseSubParameters() override;
    double SubParameterRange() override;

  private:
    double Inv32;
  };

}  // namespace Exhume

#endif
