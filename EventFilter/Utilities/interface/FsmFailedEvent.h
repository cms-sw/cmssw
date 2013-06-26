#ifndef EVF_FSMFAILEDEVENT_H
#define EVF_FSMFAILEDEVENT_H 1

#include "toolbox/Event.h"

#include <string>


namespace evf {
  
  
  class FsmFailedEvent : public toolbox::Event
  {
  public:
    //
    // construction/destruction
    //
    FsmFailedEvent(const std::string& errorMessage,void *originator=0);
    
    //
    // member functions
    //
    std::string errorMessage() { return errorMessage_; }


private:
    //
    // member data
    //
    std::string errorMessage_;
    
  };
  
}


#endif
