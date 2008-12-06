#ifndef Fireworks_Core_Context_h
#define Fireworks_Core_Context_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     Context
//
/**\class Context Context.h Fireworks/Core/interface/Context.h

 Description: Central collection of all framework managers

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Sep 30 14:21:45 EDT 2008
// $Id: Context.h,v 1.2 2008/11/06 22:05:22 amraktad Exp $
//

// system include files

// user include files

// forward declarations
class FWModelChangeManager;
class FWSelectionManager;
class FWEventItemsManager;

namespace fireworks {
   class Context {

   public:
      Context(FWModelChangeManager* iCM,
              FWSelectionManager* iSM,
              FWEventItemsManager* iEM);
      //virtual ~Context();

      // ---------- const member functions ---------------------
      FWModelChangeManager* modelChangeManager() const {
         return m_changeManager;
      }
      FWSelectionManager* selectionManager() const {
         return m_selectionManager;
      }

      const FWEventItemsManager* eventItemsManager() const {
         return m_eventItemsManager;
      }
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
      Context(const Context&); // stop default

      const Context& operator=(const Context&); // stop default

      // ---------- member data --------------------------------
      FWModelChangeManager* m_changeManager;
      FWSelectionManager* m_selectionManager;
      FWEventItemsManager* m_eventItemsManager;
   };
}

#endif
