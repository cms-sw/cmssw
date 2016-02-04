// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableViewManager_h
#define Fireworks_Core_FWTriggerTableViewManager_h
//
// Package:     Core
// Class  :     FWTriggerTableViewManager
// $Id: FWTriggerTableViewManager.h,v 1.5 2011/02/16 18:38:36 amraktad Exp $
//

#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWConfigurable.h"

class FWViewBase;
class FWGUIManager;
class TEveWindowSlot;

namespace fwlite {
   class Event;
}

class FWTriggerTableViewManager : public FWViewManagerBase, public FWConfigurable {

public:
   FWTriggerTableViewManager(FWGUIManager*);
   virtual ~FWTriggerTableViewManager();

   // dummy functions of FWViewManagerBase
   virtual FWTypeToRepresentations supportedTypesAndRepresentations() const
   { return FWTypeToRepresentations();}
   virtual void newItem(const FWEventItem*) {}

   // backward compatibility
   void addTo(FWConfiguration&) const {}
   void setFrom(const FWConfiguration&) {}

   FWViewBase *buildView (TEveWindowSlot *iParent, const std::string& type);

   // virtual void setContext(const fireworks::Context*);
protected:
   FWTriggerTableViewManager();


   virtual void modelChangesComing() {}
   virtual void modelChangesDone() {}

   virtual void eventEnd();
   virtual void colorsChanged();

   void updateProcessList();

   std::vector<boost::shared_ptr<FWTriggerTableView> > m_views;

private:
   FWTriggerTableViewManager(const FWTriggerTableViewManager&);      // stop default
   const FWTriggerTableViewManager& operator=(const FWTriggerTableViewManager&);      // stop default

   void beingDestroyed(const FWViewBase*);

};

#endif
