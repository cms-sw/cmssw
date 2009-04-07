// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTableViewManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: FWTableViewManager.cc,v 1.17 2009/04/07 14:07:28 chrjones Exp $
//

// system include files
#include <iostream>
#include <boost/bind.hpp>
#include <algorithm>
#include "TView.h"
#include "TList.h"
#include "TEveManager.h"
#include "TClass.h"

// user include files
#include "Fireworks/Core/interface/FWTableViewManager.h"
#include "Fireworks/Core/interface/FWTableView.h"
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
FWTableViewManager::FWTableViewManager(FWGUIManager* iGUIMgr) :
   FWViewManagerBase()
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FWTableViewManager::buildView,
                 this, _1);
   iGUIMgr->registerViewBuilder(FWTableView::staticTypeName(), f);
}

FWTableViewManager::~FWTableViewManager()
{
}

//
// member functions
//
class FWViewBase*
FWTableViewManager::buildView(TEveWindowSlot* iParent)
{
   TEveManager::TRedrawDisabler disableRedraw(gEve);
   boost::shared_ptr<FWTableView> view( new FWTableView(iParent) );
   view->setBackgroundColor(colorManager().background());
   m_views.push_back(view);
   view->beingDestroyed_.connect(boost::bind(&FWTableViewManager::beingDestroyed,this,_1));
   return view.get();
}

void
FWTableViewManager::beingDestroyed(const FWViewBase* iView)
{
   for(std::vector<boost::shared_ptr<FWTableView> >::iterator it=
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
FWTableViewManager::newItem(const FWEventItem* iItem)
{

}

void
FWTableViewManager::modelChangesComing()
{
   gEve->DisableRedraw();
}

void
FWTableViewManager::modelChangesDone()
{
   gEve->EnableRedraw();
}

void
FWTableViewManager::colorsChanged()
{
   for(std::vector<boost::shared_ptr<FWTableView> >::iterator it=
       m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
      (*it)->setBackgroundColor(colorManager().background());
   }
}

FWTypeToRepresentations
FWTableViewManager::supportedTypesAndRepresentations() const
{
   FWTypeToRepresentations returnValue;
//    const std::string kSimple("simple#");

//    for(TypeToBuilders::const_iterator it = m_typeToBuilders.begin(), itEnd = m_typeToBuilders.end();
//        it != itEnd;
//        ++it) {
//       for ( std::vector<std::string>::const_iterator builderName = it->second.begin();
//             builderName != it->second.end(); ++builderName )
//       {
//          if(builderName->substr(0,kSimple.size()) == kSimple) {
//             returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>( new FWSimpleRepresentationChecker(
//                                                                                builderName->substr(kSimple.size(),
//                                                                                                    builderName->find_first_of('@')-kSimple.size()),
//                                                                                it->first)));
//          } else {

//             returnValue.add(boost::shared_ptr<FWRepresentationCheckerBase>( new FWEDProductRepresentationChecker(
//                                                                                builderName->substr(0,builderName->find_first_of('@')),
//                                                                                it->first)));
//          }
//       }
//    }
   return returnValue;
}

