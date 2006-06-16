#include "EventFilter/Utilities/interface/ShutDownListener.h"
#include "EventFilter/Utilities/interface/ShutDownNotifier.h"
using namespace evf;

ShutDownListener::ShutDownListener()
{
  ShutDownNotifier *sd =  ShutDownNotifier::instance();
  if(sd)
    sd->registerListener(this);
}
ShutDownListener::~ShutDownListener()
{
}
