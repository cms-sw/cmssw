#ifndef Fireworks_FWInterface_FWFFNavigator_h
#define Fireworks_FWInterface_FWFFNavigator_h

#include "Fireworks/Core/interface/FWNavigatorBase.h"
#include "FWCore/Framework/interface/Event.h"

namespace edm
{
   class EventBase;
}

class FWFFNavigator : public FWNavigatorBase
{
public:
   enum FWFFNavigatorState {
      kNoTransition,
      kNextEvent,
      kPreviousEvent,
      kFirstEvent,
      kLastEvent
   };

   FWFFNavigator(CmsShowMainBase &main)
      : FWNavigatorBase(main),
        m_currentEvent(0),
        m_currentTransition(kNoTransition)
      {}
   // Discard configuration for the time being.
   virtual void addTo(FWConfiguration&) const {}
   virtual void setFrom(const FWConfiguration&) {}

   virtual void nextEvent();
   virtual void previousEvent();
   // nextSelectedEvent and nextEvent are the same thing
   // for full framework, since there is no filtering. 
   virtual bool nextSelectedEvent() { nextEvent(); return true; }
   virtual bool previousSelectedEvent() { previousEvent(); return true; }
   
   // FIXME: for the time being there is no way to go to 
   //        first / last event.
   virtual void firstEvent();
   virtual void lastEvent();
   // FIXME: does not do anything for the time being.
   virtual void goToRunEvent(edm::RunNumber_t, edm::LuminosityBlockNumber_t, edm::EventNumber_t) {}
   // Notice that we have no way to tell whether the current one
   // is the last event in a stream.
   virtual bool isLastEvent() {return false;}
   virtual bool isFirstEvent() {return m_currentEvent->id() == m_firstEventID; }
   // FIXME: we need to change the API of the baseclass first.
   virtual const edm::EventBase* getCurrentEvent() const { return m_currentEvent; }
   // The number of selected events is always the total number of event.
   virtual int getNSelectedEvents() { return getNTotalEvents(); }
   // FIXME: I guess there is no way to tell how many events
   //        we have.
   virtual int getNTotalEvents() { return 0; }

   // Use to set the event from the framework.
   void setCurrentEvent(const edm::Event *);
   const edm::EventID &getFirstEventID();
   enum FWFFNavigatorState currentTransition() { return m_currentTransition; }
   void resetTransition() { m_currentTransition = kNoTransition; }
private:
   const edm::Event     *m_currentEvent;
   edm::EventID         m_firstEventID;
   enum FWFFNavigatorState m_currentTransition;
};

#endif
