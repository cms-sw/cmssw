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
// $Id: FWModelChangeManager.h,v 1.6 2010/05/27 08:39:34 eulisse Exp $
//

// system include files
#include "sigc++/signal.h"
#include <set>

// user include files
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWItemChangeSignal.h"

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
   void changed(const FWEventItem*);
   void endChanges();

   sigc::signal<void> changeSignalsAreComing_;
   sigc::signal<void> changeSignalsAreDone_;

   void newItemSlot(FWEventItem*);
   void itemsGoingToBeClearedSlot(void);

private:
   FWModelChangeManager(const FWModelChangeManager&);    // stop default

   const FWModelChangeManager& operator=(const FWModelChangeManager&);    // stop default

   // ---------- member data --------------------------------
   unsigned int m_depth;
   std::vector<FWModelIds> m_changes;
   std::set<const FWEventItem*> m_itemChanges;
   std::vector<FWModelChangeSignal> m_changeSignals;
   std::vector<FWItemChangeSignal> m_itemChangeSignals;
};

class FWChangeSentry {
public:
   FWChangeSentry(FWModelChangeManager& iM) :
      m_manager(&iM)
   {
      m_manager->beginChanges();
   }
   ~FWChangeSentry()
   {
      m_manager->endChanges();
   }
private:
   FWModelChangeManager* m_manager;
};

#endif
