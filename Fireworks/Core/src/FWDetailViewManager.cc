// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Mar  5 09:13:47 EST 2008
// $Id: FWDetailViewManager.cc,v 1.58 2010/06/02 22:39:21 chrjones Exp $
//

#include <stdio.h>
#include <boost/bind.hpp>
#include <algorithm>

#include "TClass.h"
#include "TGWindow.h"
#include "TGFrame.h"
#include "TEveWindow.h"

#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDetailViewFactory.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWRepresentationInfo.h"
#include "Fireworks/Core/interface/fwLog.h"


static
std::string viewNameFrom(const std::string& iFull)
{
   std::string::size_type first = iFull.find_first_of('@');
   std::string::size_type second = iFull.find_first_of('@',first+1);
   return iFull.substr(first+1,second-first-1);
   
}
//
// constructors and destructor
//
FWDetailViewManager::FWDetailViewManager(FWColorManager* colMng):
   m_colorManager(colMng),
   m_eveFrame(0),
   m_detailView(0)
{  
   m_colorManager->colorsHaveChanged_.connect(boost::bind(&FWDetailViewManager::colorsChanged,this));
}

FWDetailViewManager::~FWDetailViewManager()
{
   if (m_detailView != 0)
      delete m_detailView;
}

void
FWDetailViewManager::openDetailViewFor(const FWModelId &id, const std::string& iViewName)
{
   if (m_detailView != 0)
   {
      m_eveFrame->GetEveWindow()->DestroyWindow();
      delete m_detailView;
   }
   assertMainFrame();
   

   // find the right viewer for this item
   std::string typeName = ROOT::Reflex::Type::ByTypeInfo(*(id.item()->modelType()->GetTypeInfo())).Name(ROOT::Reflex::SCOPED);
   std::vector<std::string> viewerNames = findViewersFor(typeName);
   if(0==viewerNames.size()) {
      fwLog(fwlog::kError) << "FWDetailViewManager: don't know what detailed view to "
         "use for object " << id.item()->name() << std::endl;
      assert(0!=viewerNames.size());
   }

   //see if one of the names matches iViewName
   std::string match;
   for(std::vector<std::string>::iterator it = viewerNames.begin(), itEnd = viewerNames.end(); it != itEnd; ++it) {
      std::string t = viewNameFrom(*it);
      //std::cout <<"'"<<iViewName<< "' '"<<t<<"'"<<std::endl;
      if(t == iViewName) {
         match = *it;
         break;
      }
   }
   assert(match.size() != 0);
   m_detailView = FWDetailViewFactory::get()->create(match);
   assert(0!=m_detailView);

   TEveWindowSlot* ws  = (TEveWindowSlot*)(m_eveFrame->GetEveWindow());
   m_detailView->init(ws);
   m_detailView->build(id);

   TGMainFrame* mf = (TGMainFrame*)(m_eveFrame->GetParent());
   mf->MapRaised();
}

std::vector<std::string>
FWDetailViewManager::detailViewsFor(const FWModelId& iId) const
{
   std::string typeName = ROOT::Reflex::Type::ByTypeInfo(*(iId.item()->modelType()->GetTypeInfo())).Name(ROOT::Reflex::SCOPED);
   std::vector<std::string> fullNames = findViewersFor(typeName);
   std::vector<std::string> justViewNames;
   justViewNames.reserve(fullNames.size());
   std::transform(fullNames.begin(),fullNames.end(),std::back_inserter(justViewNames),&viewNameFrom);
   return justViewNames;
}

std::vector<std::string>
FWDetailViewManager::findViewersFor(const std::string& iType) const
{
   std::vector<std::string> returnValue;

   std::map<std::string,std::vector<std::string> >::const_iterator itFind = m_typeToViewers.find(iType);
   if(itFind != m_typeToViewers.end()) {
      return itFind->second;
   }
   //create a list of the available ViewManager's
   std::set<std::string> detailViews;

   std::vector<edmplugin::PluginInfo> available = FWDetailViewFactory::get()->available();
   std::transform(available.begin(),
                  available.end(),
                  std::inserter(detailViews,detailViews.begin()),
                  boost::bind(&edmplugin::PluginInfo::name_,_1));
   unsigned int closestMatch= 0xFFFFFFFF;
   for(std::set<std::string>::iterator it = detailViews.begin(), itEnd=detailViews.end();
       it!=itEnd;
       ++it) {
      std::string::size_type first = it->find_first_of('@');
      std::string type = it->substr(0,first);

      if(type == iType) {
         returnValue.push_back(viewNameFrom(*it));
      }
      //see if we match via inheritance
      FWSimpleRepresentationChecker checker(type,"",0,false);
      FWRepresentationInfo info = checker.infoFor(iType);
      if(closestMatch > info.proximity()) {
         //closestMatch = info.proximity();
         returnValue.push_back(*it);
      }
   }
   m_typeToViewers[iType]=returnValue;
   return returnValue;
}

void
FWDetailViewManager::colorsChanged()
{
   if (m_detailView) { 
      m_detailView->setBackgroundColor(m_colorManager->background());
   }
}

void
FWDetailViewManager::assertMainFrame()
{
   if (m_eveFrame == 0)
   {
      TEveWindowSlot* slot = TEveWindow::CreateWindowMainFrame();
      m_eveFrame = (TEveCompositeFrameInMainFrame*)slot->GetEveFrame();
      TGMainFrame* mf = (TGMainFrame*)m_eveFrame->GetParent();
      mf->Connect( "Destroyed()", "FWDetailViewManager", this, "eveWindowDestroyed()");   
   }
}

void
FWDetailViewManager::newEventCallback()
{
   if (m_detailView)
   {
      TGMainFrame* mf = (TGMainFrame*)m_eveFrame->GetParent();
      mf->CloseWindow();
   }
}

void
FWDetailViewManager::eveWindowDestroyed()
{
   delete  m_detailView;
   m_detailView = 0;
   m_eveFrame =0 ;
}
