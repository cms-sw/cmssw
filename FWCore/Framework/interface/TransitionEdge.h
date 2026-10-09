#ifndef FWCore_Framework_TransitionEdge_h
#define FWCore_Framework_TransitionEdge_h
// Package:     FWCore/Framework
//
// Description: Used to denote the edge of a transition, i.e. the beginning or end of a transition.
//              This is used to select the correct transition action based on the edge of the transition.
//

namespace edm {
  enum class TransitionEdge { kBegin, kEnd };
}  // namespace edm
#endif
