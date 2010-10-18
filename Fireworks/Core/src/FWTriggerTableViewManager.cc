// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTriggerTableViewManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: FWTriggerTableViewManager.cc,v 1.1 2009/10/06 18:56:06 dmytro Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include <algorithm>
#include "TView.h"
#include "TList.h"
#include "TEveManager.h"
#include "TClass.h"
#include "Reflex/Base.h"
#include "Reflex/Type.h"

// user include files
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWTriggerTableViewManager.h"
#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"

#include "TEveSelection.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"

#include "Fireworks/Core/interface/FWEDProductRepresentationChecker.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTriggerTableViewManager::FWTriggerTableViewManager(FWGUIManager* iGUIMgr) :
   FWViewManagerBase()
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FWTriggerTableViewManager::buildView,
                 this, _1);
   iGUIMgr->registerViewBuilder(FWTriggerTableView::staticTypeName(), f);
}

FWTriggerTableViewManager::~FWTriggerTableViewManager()
{
}


class FWViewBase*
FWTriggerTableViewManager::buildView(TEveWindowSlot* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);
   boost::shared_ptr<FWTriggerTableView> view( new FWTriggerTableView(iParent, this) );
   view->setBackgroundColor(colorManager().background());
   m_views.push_back(view);
   view->beingDestroyed_.connect(boost::bind(&FWTriggerTableViewManager::beingDestroyed,
                                             this,_1));
   return view.get();
}

void
FWTriggerTableViewManager::beingDestroyed(const FWViewBase* iView)
{
   for(std::vector<boost::shared_ptr<FWTriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
      if(it->get() == iView) {
         m_views.erase(it);
         return;
      }
   }
}

void
FWTriggerTableViewManager::newItem(const FWEventItem* iItem)
{
   m_items.push_back(iItem);
   iItem->goingToBeDestroyed_.connect(boost::bind(&FWTriggerTableViewManager::destroyItem,
                                                  this, _1));
   // tell the views to update their item lists
   for(std::vector<boost::shared_ptr<FWTriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd; ++it) {
      (*it)->dataChanged();
   }
}

void FWTriggerTableViewManager::destroyItem (const FWEventItem *item)
{
   // remove the item from the list
   for (std::vector<const FWEventItem *>::iterator it = m_items.begin(),
        itEnd = m_items.end();
        it != itEnd; ++it) {
      if (*it == item) {
         m_items.erase(it);
         break;
      }
   }
   // tell the views to update their item lists
   for(std::vector<boost::shared_ptr<FWTriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd; ++it) {
      (*it)->dataChanged();
   }
}

void
FWTriggerTableViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
   // printf("changes coming\n");
}

void
FWTriggerTableViewManager::modelChangesDone()
{
   gEve->EnableRedraw();
   // tell the views to update their item lists
   for(std::vector<boost::shared_ptr<FWTriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd; ++it) {
      (*it)->dataChanged();
   }
   // printf("changes done\n");
}

void
FWTriggerTableViewManager::colorsChanged()
{
   for(std::vector<boost::shared_ptr<FWTriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
      (*it)->resetColors(colorManager());
//       printf("Changed the background color for a table to 0x%x\n",
//           colorManager().background());
   }
}

void
FWTriggerTableViewManager::dataChanged()
{
   for(std::vector<boost::shared_ptr<FWTriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
      (*it)->dataChanged();
//       printf("Changed the background color for a table to 0x%x\n",
//           colorManager().background());
   }
}

FWTypeToRepresentations
FWTriggerTableViewManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;
   return returnValue;
}

const std::string FWTriggerTableViewManager::kConfigTypeNames = "typeNames";

void FWTriggerTableViewManager::addTo (FWConfiguration &iTo) const
{
   // if there are views, it's the job of the first view to store
   // the configuration (this is to avoid ordering problems in the
   // case of multiple views)
   if (m_views.size() > 0) return;
   // if there are no views, then it's up to us to store the column
   // formats.  This is done in addToImpl, which can be called by
   // FWTriggerTableView as well
   addToImpl(iTo);
}

void FWTriggerTableViewManager::addToImpl (FWConfiguration &iTo) const
{
}

void FWTriggerTableViewManager::setFrom(const FWConfiguration &iFrom)
{
}
