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
// $Id: FWEventItemsManager.h,v 1.14 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include <vector>
#include <boost/shared_ptr.hpp>
#include "sigc++/signal.h"

// user include files
#include "Fireworks/Core/interface/FWConfigurable.h"

// forward declarations
namespace fwlite {
   class Event;
}
namespace fireworks {
   class Context;
}

class FWEventItem;
class FWPhysicsObjectDesc;
class FWModelChangeManager;
class FWSelectionManager;
class DetIdToMatrix;
class FWItemAccessorFactory;

class FWEventItemsManager : public FWConfigurable
{
public:
   //does not take ownership of the object to which it points but does keep reference
   FWEventItemsManager(FWModelChangeManager*);
   virtual ~FWEventItemsManager();

   typedef std::vector<FWEventItem*>::const_iterator const_iterator;

   //configuration management interface
   void addTo(FWConfiguration&) const;
   void setFrom(const FWConfiguration&);

   // ---------- const member functions ---------------------
   ///NOTE: iterator is allowed to return a null object for items that have been removed
   const_iterator begin() const;
   const_iterator end() const;
   // const std::vector<FWEventItem*> &items () const { return m_items; }

   const FWEventItem* find(const std::string& iName) const;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   const FWEventItem* add(const FWPhysicsObjectDesc& iItem);
   void clearItems();

   void newEvent(const fwlite::Event* iEvent);
   void setGeom(const DetIdToMatrix* geom);

   void setContext(fireworks::Context*);

   sigc::signal<void, FWEventItem*> newItem_;
   sigc::signal<void> goingToClearItems_;
private:

   void removeItem(const FWEventItem*);
   FWEventItemsManager(const FWEventItemsManager&);    // stop default

   const FWEventItemsManager& operator=(const FWEventItemsManager&);    // stop default

   // ---------- member data --------------------------------
   std::vector<FWEventItem*> m_items;
   FWModelChangeManager* m_changeManager;
   fireworks::Context* m_context;

   const fwlite::Event* m_event;
   const DetIdToMatrix *m_geom;
   boost::shared_ptr<FWItemAccessorFactory> m_accessorFactory;
};


#endif
