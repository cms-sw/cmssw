// -*- C++ -*-
#ifndef Fireworks_Core_FWTableViewManager_h
#define Fireworks_Core_FWTableViewManager_h
//
// Package:     Core
// Class  :     FWTableViewManager
//
/**\class FWTableViewManager FWTableViewManager.h Fireworks/Core/interface/FWTableViewManager.h

   Description: Base class for a Manger for a specific type of View

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Sat Jan  5 10:29:00 EST 2008
// $Id: FWTableViewManager.h,v 1.16 2009/04/07 14:02:33 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include <set>
#include <map>

// user include files

//Needed for gccxml
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWTableView.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"

class FWViewBase;
class FWGUIManager;
class TEveWindowSlot;

class FWTableViewManager : public FWViewManagerBase {

public:
   FWTableViewManager(FWGUIManager*);
   virtual ~FWTableViewManager();

   // ---------- const member functions ---------------------
   virtual FWTypeToRepresentations supportedTypesAndRepresentations() const;
   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   virtual void newItem(const FWEventItem*);
   FWViewBase *buildView (TEveWindowSlot *iParent);

protected:
   FWTableViewManager();

   /** called when models have changed and so the display must be updated*/
   virtual void modelChangesComing();
   virtual void modelChangesDone();
   virtual void colorsChanged();

private:
   FWTableViewManager(const FWTableViewManager&);    // stop default
   const FWTableViewManager& operator=(const FWTableViewManager&);    // stop default

   void beingDestroyed(const FWViewBase*);

   std::vector<boost::shared_ptr<FWTableView> > m_views;
};

#endif
