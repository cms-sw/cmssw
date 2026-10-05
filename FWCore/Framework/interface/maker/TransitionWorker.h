#ifndef FWCore_Framework_maker_TransitionWorker_h
#define FWCore_Framework_maker_TransitionWorker_h
/*----------------------------------------------------------------------    
*/

#include "FWCore/Framework/interface/maker/Worker.h"
#include "FWCore/Framework/interface/OccurrenceTraits.h"
#include "FWCore/Framework/interface/TransitionEdge.h"
#include "FWCore/Framework/interface/TransitionPhaseTypes.h"

#include "TransitionWorker_Common.h"
#include "TransitionWorker_Event.h"
#include "TransitionWorker_InputProcessBlock.h"
#include "TransitionWorker_ProcessBlock.h"

namespace edm {
  using StreamRunWorker = TransitionWorker<RunTransitionInfo, TransitionPhaseStream>;
  using StreamLumiWorker = TransitionWorker<LumiTransitionInfo, TransitionPhaseStream>;
  using GlobalRunWorker = TransitionWorker<RunTransitionInfo, TransitionPhaseGlobal>;
  using GlobalLumiWorker = TransitionWorker<LumiTransitionInfo, TransitionPhaseGlobal>;
  using GlobalEventWorker = TransitionWorker<EventTransitionInfo, TransitionPhaseGlobal>;
}  // namespace edm
#endif
