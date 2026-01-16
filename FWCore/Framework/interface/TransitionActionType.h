#ifndef Framework_TransitionActionType_h
#define Framework_TransitionActionType_h

/*----------------------------------------------------------------------

TransitionActionType: TransitionAction

----------------------------------------------------------------------*/

namespace edm {
  enum TransitionActionType {
    TransitionActionGlobalBegin = 0,
    TransitionActionStreamBegin = 1,
    TransitionActionStreamEnd = 2,
    TransitionActionGlobalEnd = 3,
    TransitionActionProcessBlockInput = 4
  };
}
#endif
