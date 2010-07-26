#include "Fireworks/FWInterface/src/FWFFNavigator.h"
#include "TSystem.h"

void
FWFFNavigator::nextEvent()
{
   gSystem->ExitLoop();
}

void
FWFFNavigator::setCurrentEvent(const edm::Event *event)
{
   m_currentEvent = event;
   newEvent_.emit();
}
