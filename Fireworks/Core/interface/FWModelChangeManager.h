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
// $Id: FWModelChangeManager.h,v 1.1 2008/01/21 01:17:16 chrjones Exp $
//

// system include files
#include "sigc++/signal.h"
#include <set>

// user include files
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"

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
   
      void beginChanges();
      void changed(const FWModelId&);
      void endChanges();

      sigc::signal<void> changeSignalsAreComing_;
      sigc::signal<void> changeSignalsAreDone_;
   
      void newItemSlot(FWEventItem*);
   private:
      FWModelChangeManager(const FWModelChangeManager&); // stop default

      const FWModelChangeManager& operator=(const FWModelChangeManager&); // stop default

      // ---------- member data --------------------------------
      unsigned int m_depth;
      std::vector<FWModelIds> m_changes;
      std::vector<FWModelChangeSignal> m_changeSignals;
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
