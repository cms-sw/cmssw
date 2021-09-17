#ifndef FWCore_Utilities_Transition_h
#define FWCore_Utilities_Transition_h
/*----------------------------------------------------------------------

Transition: The allowed framework transitions

----------------------------------------------------------------------*/
#include "FWCore/Utilities/interface/BranchType.h"
#include <type_traits>

namespace edm {
  enum class Transition {
    Event,
    BeginLuminosityBlock,
    EndLuminosityBlock,
    BeginRun,
    EndRun,
    BeginProcessBlock,
    EndProcessBlock,
    AccessInputProcessBlock,
    NumberOfTransitions,
    NumberOfEventSetupTransitions = BeginProcessBlock
  };

  //Useful for converting EndBranchType to BranchType
  constexpr BranchType convertToBranchType(Transition iValue) {
    constexpr BranchType branches[] = {InEvent, InLumi, InLumi, InRun, InRun, InProcess, InProcess};
    return branches[static_cast<std::underlying_type<Transition>::type>(iValue)];
  }

  constexpr Transition convertToTransition(BranchType iValue) {
    constexpr Transition trans[] = {
        Transition::Event, Transition::BeginLuminosityBlock, Transition::BeginRun, Transition::BeginProcessBlock};
    return trans[iValue];
  }

  constexpr bool isEndTransition(Transition iValue) {
    return iValue == Transition::EndLuminosityBlock or iValue == Transition::EndRun or
           iValue == Transition::EndProcessBlock;
  }

}  // namespace edm
#endif
