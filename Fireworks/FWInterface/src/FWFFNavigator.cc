#include "Fireworks/FWInterface/src/FWFFNavigator.h"
#include "TSystem.h"

void
FWFFNavigator::nextEvent()
{
   m_currentTransition = kNextEvent;
   gSystem->ExitLoop();
}

void
FWFFNavigator::previousEvent()
{
   m_currentTransition = kPreviousEvent;
   gSystem->ExitLoop();
}

void
FWFFNavigator::firstEvent()
{
   m_currentTransition = kFirstEvent;
   gSystem->ExitLoop();
}

void
FWFFNavigator::lastEvent()
{
   m_currentTransition = kLastEvent;
   gSystem->ExitLoop();
}

/** API to move to a given event. Notice
    that it is also responsible for keeping registering the ID of the first
    event, so that we can stop going back.
  */
void
FWFFNavigator::setCurrentEvent(const edm::Event *event)
{
   m_currentEvent = event;
   if (m_firstEventID == edm::EventID())
      m_firstEventID = m_currentEvent->id();
   newEvent_.emit();
}

const edm::EventID &
FWFFNavigator::getFirstEventID()
{
   return m_firstEventID;
}
