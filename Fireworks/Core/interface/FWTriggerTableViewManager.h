// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableViewManager_h
#define Fireworks_Core_FWTriggerTableViewManager_h
//
// Package:     Core
// Class  :     FWTriggerTableViewManager
// $Id: FWTriggerTableViewManager.h,v 1.5 2009/09/24 14:55:25 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include <set>
#include <map>
#include "Reflex/Type.h"

// user include files

#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/FWConfigurable.h"

class FWViewBase;
class FWGUIManager;
class TEveWindowSlot;

namespace fwlite {
   class Event;
}

class FWTriggerTableViewManager : public FWViewManagerBase, public FWConfigurable {
   friend class FWTriggerTableView;
   friend class FWTriggerTableViewTableManager;

public:
   FWTriggerTableViewManager(FWGUIManager*);
   virtual ~FWTriggerTableViewManager();

   virtual FWTypeToRepresentations supportedTypesAndRepresentations() const;

   virtual void newItem(const FWEventItem*);
   void destroyItem (const FWEventItem *item);
   FWViewBase *buildView (TEveWindowSlot *iParent);
   const std::vector<const FWEventItem *> &items () const {
      return m_items;
   }
   void addTo(FWConfiguration&) const;
   void addToImpl (FWConfiguration&) const;
   void setFrom(const FWConfiguration&);

   static const std::string kConfigTypeNames;
   static const std::string kConfigColumns;

protected:
   FWTriggerTableViewManager();

   /** called when models have changed and so the display must be updated*/
   virtual void modelChangesComing();
   virtual void modelChangesDone();
   virtual void colorsChanged();
   void dataChanged ();

   std::vector<boost::shared_ptr<FWTriggerTableView> > m_views;
   std::vector<const FWEventItem *> m_items;

private:
   FWTriggerTableViewManager(const FWTriggerTableViewManager&);      // stop default
   const FWTriggerTableViewManager& operator=(const FWTriggerTableViewManager&);      // stop default

   void beingDestroyed(const FWViewBase*);

};

#endif
