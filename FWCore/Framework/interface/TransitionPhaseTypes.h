#ifndef FWCore_Framework_TransitionPhaseTypes_h
#define FWCore_Framework_TransitionPhaseTypes_h
//
// Package:     FWCore/Framework
//
/**

 Description: Specifies the specific major phase type for a transition.

*/

namespace edm {
  class GlobalContext;
  class StreamContext;

  enum class TransitionPhaseType { Global, Stream };
  struct TransitionPhaseGlobal {
    using ContextType = GlobalContext;
    static constexpr TransitionPhaseType value = TransitionPhaseType::Global;
  };
  struct TransitionPhaseStream {
    using ContextType = StreamContext;
    static constexpr TransitionPhaseType value = TransitionPhaseType::Stream;
  };
}  // namespace edm

#endif