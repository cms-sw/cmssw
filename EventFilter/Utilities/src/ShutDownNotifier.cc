#include "EventFilter/Utilities/interface/ShutDownNotifier.h"
#include "EventFilter/Utilities/interface/ShutDownListener.h"

using namespace evf;

ShutDownNotifier *ShutDownNotifier::instance_ = 0;
void ShutDownNotifier::registerListener(ShutDownListener *l)
{
  listeners_.push_back(l);
}
void ShutDownNotifier::removeListener(ShutDownListener *l)
{
  //  listeners_.pop_front();
}
void ShutDownNotifier::cleanup()
{
  listeners_.erase(listeners_.begin(),listeners_.end());
}
void ShutDownNotifier::notify()
{
  for(unsigned int i = 0; i<listeners_.size(); i++)
    listeners_[i]->onShutDown();
}
