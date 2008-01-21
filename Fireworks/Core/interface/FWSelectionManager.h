#ifndef Fireworks_Core_FWSelectionManager_h
#define Fireworks_Core_FWSelectionManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWSelectionManager
// 
/**\class FWSelectionManager FWSelectionManager.h Fireworks/Core/interface/FWSelectionManager.h

 Description: Manages the list of selected Model items

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jan 18 14:40:45 EST 2008
// $Id$
//

// system include files
#include "sigc++/signal.h"
#include <set>

// user include files
#include "Fireworks/Core/interface/FWModelId.h"

// forward declarations
class FWEventItem;
class FWModelChangeManager;

class FWSelectionManager
{
   //only an item can set the selection
   friend class FWEventItem;
   public:
      FWSelectionManager(FWModelChangeManager* iCM);
      //virtual ~FWSelectionManager();

      // ---------- const member functions ---------------------
      const std::set<FWModelId>& selected() const;
   
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void clearSelection();

      sigc::signal<void, const FWSelectionManager&> selectionChanged_;
   
   private:
      void finishedAllSelections();
      void select(const FWModelId& iId);
      void unselect(const FWModelId& iId);
   
      FWSelectionManager(const FWSelectionManager&); // stop default

      const FWSelectionManager& operator=(const FWSelectionManager&); // stop default

      // ---------- member data --------------------------------
      FWModelChangeManager* m_changeManager;
      std::set<FWModelId> m_selection;
      std::set<FWModelId> m_newSelection;
      bool m_wasChanged;
};


#endif
