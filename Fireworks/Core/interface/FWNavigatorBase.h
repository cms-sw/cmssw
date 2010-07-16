// -*- C++ -*-
#ifndef Fireworks_Core_FWNavigatorBase_h
#define Fireworks_Core_FWNavigatorBase_h
//
// Package:     newVersion
// Class  :     CmsShowNavigator
// $Id: CmsShowNavigator.h,v 1.50 2010/06/17 20:17:20 amraktad Exp $
//

// system include files
#include <string>
#include <sigc++/sigc++.h>

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "Fireworks/Core/interface/FWConfigurable.h"

// forward declarations
class CmsShowMain;

namespace edm {
   class EventID;
}

class FWNavigatorBase : public FWConfigurable
{
public:
   enum EFilterState { kOff, kOn, kWithdrawn };
   enum EFilterMode  { kOr = 1, kAnd = 2 };
   
public:
   FWNavigatorBase(const CmsShowMain &);
   virtual ~FWNavigatorBase();

   //configuration management interface
   virtual void addTo(FWConfiguration&) const = 0;
   virtual void setFrom(const FWConfiguration&) = 0;

   virtual void nextEvent() = 0;
   virtual void previousEvent() = 0;
   virtual bool nextSelectedEvent() = 0;
   virtual bool previousSelectedEvent() = 0;
   virtual void firstEvent() = 0;
   virtual void lastEvent() = 0;
   virtual void goToRunEvent(Int_t,Int_t) = 0;

   virtual bool isLastEvent() = 0;
   virtual bool isFirstEvent() = 0;

   virtual const fwlite::Event* getCurrentEvent() const = 0;
   virtual int  getNSelectedEvents() = 0;
   virtual int  getNTotalEvents() = 0;

   sigc::signal<void> newEvent_;
private:
   FWNavigatorBase(const FWNavigatorBase&);    // stop default
   const FWNavigatorBase& operator=(const FWNavigatorBase&);    // stop default
   // ---------- member data --------------------------------
   // entry is an event index nubmer which runs from 0 to
   // #events or #selected_events depending on if we filter
   // events or not
   const CmsShowMain &m_main;
};

#endif
