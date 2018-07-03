// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableViewManager_h
#define Fireworks_Core_FWTriggerTableViewManager_h
//
// Package:     Core
// Class  :     FWTriggerTableViewManager
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
   ~FWTriggerTableViewManager() override;

   // dummy functions of FWViewManagerBase
   FWTypeToRepresentations supportedTypesAndRepresentations() const override
   { return FWTypeToRepresentations();}
   void newItem(const FWEventItem*) override {}

   // backward compatibility
   void addTo(FWConfiguration&) const override {}
   void setFrom(const FWConfiguration&) override {}

   FWViewBase *buildView (TEveWindowSlot *iParent, const std::string& type);

   // virtual void setContext(const fireworks::Context*);
protected:
   FWTriggerTableViewManager();


   void modelChangesComing() override {}
   void modelChangesDone() override {}

   void eventEnd() override;
   void colorsChanged() override;

   void updateProcessList();

   std::vector<std::shared_ptr<FWTriggerTableView> > m_views;

private:
   FWTriggerTableViewManager(const FWTriggerTableViewManager&);      // stop default
   const FWTriggerTableViewManager& operator=(const FWTriggerTableViewManager&);      // stop default

   void beingDestroyed(const FWViewBase*);

};

#endif
