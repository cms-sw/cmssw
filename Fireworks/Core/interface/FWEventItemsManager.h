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
// $Id: FWEventItemsManager.h,v 1.2 2008/01/15 22:39:42 chrjones Exp $
//

// system include files
#include <vector>
#include "RQ_OBJECT.h"

// user include files

// forward declarations
namespace fwlite {
  class Event;
}
class FWEventItem;
class FWPhysicsObjectDesc;
class DetIdToMatrix;

class FWEventItemsManager
{
   RQ_OBJECT("FWEventItemsManager")
   public:
      FWEventItemsManager();
      virtual ~FWEventItemsManager();

      typedef std::vector<FWEventItem*>::const_iterator const_iterator;
      // ---------- const member functions ---------------------
      const_iterator begin() const;
      const_iterator end() const;

      const FWEventItem* find(const std::string& iName) const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      /** copies the FWEventItem internally.  Returns 0 if
          it was not added because an FWEventItem with the same name
          already exists
       */
      const FWEventItem* add(const FWPhysicsObjectDesc& iItem);

      void newEvent(const fwlite::Event* iEvent);
      void setGeom(const DetIdToMatrix* geom);

   private:
      void newItem(const FWEventItem*); //*SIGNAL*

      FWEventItemsManager(const FWEventItemsManager&); // stop default

      const FWEventItemsManager& operator=(const FWEventItemsManager&); // stop default

      // ---------- member data --------------------------------
      std::vector<FWEventItem*> m_items;

};


#endif
