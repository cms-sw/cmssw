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

  struct TransitionPhaseGlobal {
    using ContextType = GlobalContext;
  };
  struct TransitionPhaseStream {
    using ContextType = StreamContext;
  };
}  // namespace edm

#endif