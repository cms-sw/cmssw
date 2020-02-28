#include "GeneratorInterface/Pythia8Interface/interface/P8RndmEngine.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "CLHEP/Random/RandomEngine.h"

namespace gen {

  double P8RndmEngine::flat(void) {
    if (randomEngine_ == nullptr) {
      throwNullPtr();
    }
    return randomEngine_->flat();
  }

  void P8RndmEngine::throwNullPtr() const {
    throw edm::Exception(edm::errors::LogicError) << "The Pythia 8 code attempted to a generate random number while\n"
                                                  << "the engine pointer was null. This might mean that the code\n"
                                                  << "was modified to generate a random number outside the event and\n"
                                                  << "beginLuminosityBlock methods, which is not allowed.\n";
  }
}  // namespace gen
