// -*- C++ -*-
//
// Package:     Core
// Class  :     FWL1TriggerTableViewManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: FWL1TriggerTableViewManager.cc,v 1.1.2.1 2010/03/29 09:04:05 yana Exp $
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
#include "Fireworks/Core/interface/FWL1TriggerTableViewManager.h"
#include "Fireworks/Core/interface/FWL1TriggerTableView.h"
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
FWL1TriggerTableViewManager::FWL1TriggerTableViewManager(FWGUIManager* iGUIMgr) :
   FWViewManagerBase()
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FWL1TriggerTableViewManager::buildView,
                 this, _1);
   iGUIMgr->registerViewBuilder(FWL1TriggerTableView::staticTypeName(), f);
}

FWL1TriggerTableViewManager::~FWL1TriggerTableViewManager()
{
}


class FWViewBase*
FWL1TriggerTableViewManager::buildView(TEveWindowSlot* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);
   boost::shared_ptr<FWL1TriggerTableView> view( new FWL1TriggerTableView(iParent, this) );
   view->setBackgroundColor(colorManager().background());
   m_views.push_back(view);
   view->beingDestroyed_.connect(boost::bind(&FWL1TriggerTableViewManager::beingDestroyed,
                                             this,_1));
   return view.get();
}

void
FWL1TriggerTableViewManager::beingDestroyed(const FWViewBase* iView)
{
   for(std::vector<boost::shared_ptr<FWL1TriggerTableView> >::iterator it=
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
FWL1TriggerTableViewManager::newItem(const FWEventItem* iItem)
{
   m_items.push_back(iItem);
   iItem->goingToBeDestroyed_.connect(boost::bind(&FWL1TriggerTableViewManager::destroyItem,
                                                  this, _1));
   // tell the views to update their item lists
   for(std::vector<boost::shared_ptr<FWL1TriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd; ++it) {
      (*it)->dataChanged();
   }
}

void FWL1TriggerTableViewManager::destroyItem (const FWEventItem *item)
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
   for(std::vector<boost::shared_ptr<FWL1TriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd; ++it) {
      (*it)->dataChanged();
   }
}

void
FWL1TriggerTableViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
   // printf("changes coming\n");
}

void
FWL1TriggerTableViewManager::modelChangesDone()
{
   gEve->EnableRedraw();
   // tell the views to update their item lists
   for(std::vector<boost::shared_ptr<FWL1TriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd; ++it) {
      (*it)->dataChanged();
   }
   // printf("changes done\n");
}

void
FWL1TriggerTableViewManager::colorsChanged()
{
   for(std::vector<boost::shared_ptr<FWL1TriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
      (*it)->resetColors(colorManager());
//       printf("Changed the background color for a table to 0x%x\n",
//           colorManager().background());
   }
}

void
FWL1TriggerTableViewManager::dataChanged()
{
   for(std::vector<boost::shared_ptr<FWL1TriggerTableView> >::iterator it=
          m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
      (*it)->dataChanged();
//       printf("Changed the background color for a table to 0x%x\n",
//           colorManager().background());
   }
}

FWTypeToRepresentations
FWL1TriggerTableViewManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;
   return returnValue;
}

const std::string FWL1TriggerTableViewManager::kConfigTypeNames = "typeNames";

void FWL1TriggerTableViewManager::addTo (FWConfiguration &iTo) const
{
   std::cout << "writing configuration" << std::endl;
   // if there are views, it's the job of the first view to store
   // the configuration (this is to avoid ordering problems in the
   // case of multiple views)
   if (m_views.size() > 0) return;
   // if there are no views, then it's up to us to store the column
   // formats.  This is done in addToImpl, which can be called by
   // FWL1TriggerTableView as well
   addToImpl(iTo);
}

void FWL1TriggerTableViewManager::addToImpl (FWConfiguration &iTo) const
{
}

void FWL1TriggerTableViewManager::setFrom(const FWConfiguration &iFrom)
{
}
