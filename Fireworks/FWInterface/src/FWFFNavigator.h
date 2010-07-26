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
   FWFFNavigator(CmsShowMainBase &main)
      : FWNavigatorBase(main),
        m_currentEvent(0)
      {}
   // Discard configuration for the time being.
   virtual void addTo(FWConfiguration&) const {}
   virtual void setFrom(const FWConfiguration&) {}

   virtual void nextEvent();
   // FIXME: previous event not available for the time being.
   virtual void previousEvent() {}
   // nextSelectedEvent and nextEvent are the same thing
   // for full framework, since there is no filtering. 
   virtual bool nextSelectedEvent() { nextEvent(); return true; }
   virtual bool previousSelectedEvent() { previousEvent(); return true; }
   
   // FIXME: for the time being there is no way to go to 
   //        first / last event.
   virtual void firstEvent() {}
   virtual void lastEvent() {}
   // FIXME: does not do anything for the time being.
   virtual void goToRunEvent(Int_t, Int_t) {}
   // This hopefully means that the GUI will always show
   // previous event grayed out.
   virtual bool isLastEvent() { return false; }
   virtual bool isFirstEvent() { return true; }
   // FIXME: we need to change the API of the baseclass first.
   virtual const edm::EventBase* getCurrentEvent() const { return m_currentEvent; };
   // The number of selected events is always the total number of event.
   virtual int getNSelectedEvents() { return getNTotalEvents(); }
   // FIXME: I guess there is no way to tell how many events
   //        we have.
   virtual int getNTotalEvents() { return 0; }

   // Use to set the event from the framework.
   void setCurrentEvent(const edm::Event *);
private:
   const edm::Event     *m_currentEvent;
};

#endif
