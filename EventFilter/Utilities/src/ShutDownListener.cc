#include "EventFilter/Utilities/interface/ShutDownListener.h"
#include "EventFilter/Utilities/interface/ShutDownNotifier.h"

namespace evf{

  ShutDownListener::ShutDownListener()
  {
    ShutDownNotifier *sd =  ShutDownNotifier::instance();
    if(sd)
      sd->registerListener(this);
  }
  ShutDownListener::~ShutDownListener()
  {
  }
} // end namespace evf
