//-*-C++-*-
//-*-Dummy.h-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////

#ifndef DUMMY_HH
#define DUMMY_HH

#include "GeneratorInterface/ExhumeInterface/interface/CrossSection.h"

namespace Exhume{

  class Dummy : public CrossSection{

  public:

    Dummy(const edm::ParameterSet&);
    double SubProcess();
    void SetPartons();
    void SetSubParameters();
    double SubParameterWeight();
    void MaximiseSubParameters();
    double SubParameterRange();

  private:
    double Inv32;
  };

}

#endif
