#ifndef Fireworks_Core_FWModelChangeManager_h
#define Fireworks_Core_FWModelChangeManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelChangeManager
// 
/**\class FWModelChangeManager FWModelChangeManager.h Fireworks/Core/interface/FWModelChangeManager.h

 Description: Manages propagating announcements of changes to Models to any interested party

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jan 17 17:37:49 EST 2008
// $Id$
//

// system include files
#include "sigc++/signal.h"
#include <set>

// user include files
#include "Fireworks/Core/interface/FWModelId.h"

// forward declarations
class FWEventItem;

class FWModelChangeManager
{

   public:
      FWModelChangeManager();
      virtual ~FWModelChangeManager();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      ///signal is emitted after all changes have been made
      sigc::signal<void,const std::set<FWModelId>& > changes_;
   
      void beginChanges();
      void changed(const FWModelId&);
      void endChanges();
   
   private:
      FWModelChangeManager(const FWModelChangeManager&); // stop default

      const FWModelChangeManager& operator=(const FWModelChangeManager&); // stop default

      // ---------- member data --------------------------------
      unsigned int m_depth;
      std::set<FWModelId> m_changes;
};

class FWChangeSentry {
public:
   FWChangeSentry(FWModelChangeManager& iM):
   m_manager(&iM) 
   {m_manager->beginChanges();}
   ~FWChangeSentry()
   { m_manager->endChanges();}
private:
   FWModelChangeManager* m_manager;
};

#endif
