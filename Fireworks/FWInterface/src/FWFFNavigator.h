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
        m_currentEvent(nullptr),
        m_currentTransition(kNoTransition)
      {}
   // Discard configuration for the time being.
   void addTo(FWConfiguration&) const override {}
   void setFrom(const FWConfiguration&) override {}

   void nextEvent() override;
   void previousEvent() override;
   // nextSelectedEvent and nextEvent are the same thing
   // for full framework, since there is no filtering. 
   bool nextSelectedEvent() override { nextEvent(); return true; }
   bool previousSelectedEvent() override { previousEvent(); return true; }
   
   // FIXME: for the time being there is no way to go to 
   //        first / last event.
   void firstEvent() override;
   void lastEvent() override;
   // FIXME: does not do anything for the time being.
   void goToRunEvent(edm::RunNumber_t, edm::LuminosityBlockNumber_t, edm::EventNumber_t) override {}
   // Notice that we have no way to tell whether the current one
   // is the last event in a stream.
   bool isLastEvent() override {return false;}
   bool isFirstEvent() override {return m_currentEvent->id() == m_firstEventID; }
   // FIXME: we need to change the API of the baseclass first.
   const edm::EventBase* getCurrentEvent() const override { return m_currentEvent; }
   // The number of selected events is always the total number of event.
   int getNSelectedEvents() override { return getNTotalEvents(); }
   // FIXME: I guess there is no way to tell how many events
   //        we have.
   int getNTotalEvents() override { return 0; }

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
