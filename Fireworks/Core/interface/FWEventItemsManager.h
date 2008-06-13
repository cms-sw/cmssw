#ifndef Fireworks_Core_FWEventItemsManager_h
#define Fireworks_Core_FWEventItemsManager_h
// -*- C++ -*-
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
// $Id: FWEventItemsManager.h,v 1.8 2008/06/12 15:06:05 chrjones Exp $
//

// system include files
#include <vector>
#include "sigc++/signal.h"

// user include files
#include "Fireworks/Core/interface/FWConfigurable.h"

// forward declarations
namespace fwlite {
  class Event;
}
class FWEventItem;
class FWPhysicsObjectDesc;
class FWModelChangeManager;
class FWSelectionManager;
class DetIdToMatrix;

class FWEventItemsManager : public FWConfigurable
{
   public:
      //does not take ownership of the object to which it points but does keep reference
      FWEventItemsManager(FWModelChangeManager*,FWSelectionManager*);
      virtual ~FWEventItemsManager();

      typedef std::vector<FWEventItem*>::const_iterator const_iterator;
   
      //configuration management interface
      void addTo(FWConfiguration&) const;
      void setFrom(const FWConfiguration&);

      // ---------- const member functions ---------------------
      const_iterator begin() const;
      const_iterator end() const;

      const FWEventItem* find(const std::string& iName) const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      const FWEventItem* add(const FWPhysicsObjectDesc& iItem);
      void clearItems();

      void newEvent(const fwlite::Event* iEvent);
      void setGeom(const DetIdToMatrix* geom);
   
      sigc::signal<void, FWEventItem*> newItem_;
      sigc::signal<void> goingToClearItems_;
   private:

      void removeItem(const FWEventItem*);
      FWEventItemsManager(const FWEventItemsManager&); // stop default

      const FWEventItemsManager& operator=(const FWEventItemsManager&); // stop default

      // ---------- member data --------------------------------
      std::vector<FWEventItem*> m_items;
      FWModelChangeManager* m_changeManager;
      FWSelectionManager* m_selectionManager;
   
      const fwlite::Event* m_event;

};


#endif
