//-*-c++-*-
//-*-DiPhoton.h-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////
#ifndef DI_PHOTON_HH
#define DI_PHOTON_HH

#include "GeneratorInterface/ExhumeInterface/interface/TwoSpace.h"

namespace Exhume {
  class DiPhoton : public TwoSpace {
  public:
    DiPhoton(const edm::ParameterSet&);

    //declare inherited functions
    double SubProcess() override;
    void LIPS2Amp() override;
    void Amp2LIPS() override;

  private:
    double MatrixElement();
    //internal functions

    //internal Parameters
    double t_;  //internal

    double PI2, Inv64PI2;
    int Nc;
    double MatFact;
    int Nup;
    int Ndown;
  };
}  // namespace Exhume
#endif
