// -*- C++ -*-
#ifndef Fireworks_Core_FWEventItemsManager_h
#define Fireworks_Core_FWEventItemsManager_h
//
// Package:     Core
// Class  :     FWEventItemsManager
//
/**\class FWEventItemsManager FWEventItemsManager.h Fireworks/Core/interface/FWEventItemsManager.h

   Description: Manages multiple FWEventItems

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Thu Jan  3 13:27:29 EST 2008
//

// system include files
#include <vector>
#include <memory>
#include "sigc++/signal.h"

// user include files
#include "Fireworks/Core/interface/FWConfigurable.h"

// forward declarations
namespace edm {
   class EventBase;
}
namespace fireworks {
   class Context;
}

class FWEventItem;
class FWPhysicsObjectDesc;
class FWModelChangeManager;
class FWSelectionManager;
class FWItemAccessorFactory;
class FWProxyBuilderConfiguration;

class FWEventItemsManager : public FWConfigurable
{
public:
   //does not take ownership of the object to which it points but does keep reference
   FWEventItemsManager(FWModelChangeManager*);
   ~FWEventItemsManager() override;

   typedef std::vector<FWEventItem*>::const_iterator const_iterator;

   //configuration management interface
   void addTo(FWConfiguration&) const override;
   void setFrom(const FWConfiguration&) override;

   // ---------- const member functions ---------------------
   ///NOTE: iterator is allowed to return a null object for items that have been removed
   const_iterator begin() const;
   const_iterator end() const;
   // const std::vector<FWEventItem*> &items () const { return m_items; }

   const FWEventItem* find(const std::string& iName) const;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   FWEventItem* add(const FWPhysicsObjectDesc& iItem, const FWConfiguration* pbConf=nullptr,
                    bool doSetEvent=true);
   void clearItems();

   void newEvent(const edm::EventBase* iEvent);

   void setContext(fireworks::Context*);

   sigc::signal<void, FWEventItem*> newItem_;
   sigc::signal<void, const FWEventItem*> removingItem_;
   sigc::signal<void>               goingToClearItems_;

private:

   void removeItem(const FWEventItem*);
   FWEventItemsManager(const FWEventItemsManager&) = delete;    // stop default

   const FWEventItemsManager& operator=(const FWEventItemsManager&) = delete;    // stop default

   // ---------- member data --------------------------------
   std::vector<FWEventItem*> m_items;
   FWModelChangeManager* m_changeManager;
   fireworks::Context* m_context;

   const edm::EventBase* m_event;
   std::shared_ptr<FWItemAccessorFactory> m_accessorFactory;
};


#endif
