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
//

#include <stdio.h>
#include <boost/bind.hpp>
#include <algorithm>
#include <sstream>

#include "TClass.h"
#include "TROOT.h"
#include "TGWindow.h"
#include "TGFrame.h"
#include "TEveManager.h"
#include "TEveWindowManager.h"
#include "TEveWindow.h"
#include "TQObject.h"

#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDetailViewFactory.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWRepresentationInfo.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWJobMetadataManager.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/Common/interface/EventBase.h"
#include "DataFormats/FWLite/interface/Event.h"
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
FWDetailViewManager::FWDetailViewManager(fireworks::Context* iCtx):
   m_context(iCtx)
{  
   // force white background for all embedded canvases
   gROOT->SetStyle("Plain");

   m_context->colorManager()->colorsHaveChanged_.connect(boost::bind(&FWDetailViewManager::colorsChanged,this));
   gEve->GetWindowManager()->Connect( "WindowDeleted(TEveWindow*)", "FWDetailViewManager", this, "eveWindowDestroyed(TEveWindow*)");   
}

FWDetailViewManager::~FWDetailViewManager()
{ 
   newEventCallback();
   gEve->GetWindowManager()->Disconnect("WindowDeleted(TEveWindow*)", this, "eveWindowDestroyed(TEveWindow*)" );
}

void
FWDetailViewManager::openDetailViewFor(const FWModelId &id, const std::string& iViewName)
{

   TEveWindowSlot* slot = TEveWindow::CreateWindowMainFrame();
   TEveCompositeFrameInMainFrame* eveFrame = (TEveCompositeFrameInMainFrame*)slot->GetEveFrame();

   // find the right viewer for this item
   std::string typeName = edm::TypeWithDict(*(id.item()->modelType()->GetTypeInfo())).name();
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
   FWDetailViewBase* detailView = FWDetailViewFactory::get()->create(match);
   assert(0!=detailView);

   TEveWindowSlot* ws  = (TEveWindowSlot*)(eveFrame->GetEveWindow());
   detailView->init(ws);
   detailView->build(id);
   detailView->setBackgroundColor(m_context->colorManager()->background());

   TGMainFrame* mf = (TGMainFrame*)(eveFrame->GetParent());
   mf->SetWindowName(Form("%s Detail View [%d]", id.item()->name().c_str(), id.index())); 
  
   m_views.push_back(ViewFrame(eveFrame, detailView, eveFrame->GetEveWindow()));

   mf->MapRaised();
}

std::vector<std::string>
FWDetailViewManager::detailViewsFor(const FWModelId& iId) const
{
   std::string typeName = edm::TypeWithDict(*(iId.item()->modelType()->GetTypeInfo())).name();
   std::vector<std::string> fullNames = findViewersFor(typeName);
   std::vector<std::string> justViewNames;
   justViewNames.reserve(fullNames.size());
   std::transform(fullNames.begin(),fullNames.end(),std::back_inserter(justViewNames),&viewNameFrom);
   return justViewNames;
}
namespace {
bool pluginComapreFunc (std::string a, std::string b) {
         std::string::size_type as = a.find_first_of('&');
         a = a.substr(0, as);
         std::string::size_type bs = b.find_first_of('&');
         b = b.substr(0, bs);
         return a == b;
}
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

      if (m_context->getHidePFBuilders()) {
         std::size_t found = it->find("PF ");
         if (found != std::string::npos)
            continue;
      }
      std::string::size_type first = it->find_first_of('@');
      std::string type = it->substr(0,first);

      //see if we match via inheritance
      FWSimpleRepresentationChecker checker(type,"",0,false);
      FWRepresentationInfo info = checker.infoFor(iType);
      bool pass = false;
      if(closestMatch > info.proximity()) {
         pass = true;
         std::string::size_type firstD = it->find_first_of('&')+1;
         if(firstD != std::string::npos) {
          std::stringstream ss(it->substr(firstD));
          std::string ml;
          while(std::getline(ss, ml, '&')) {
             if (!m_context->metadataManager()->hasModuleLabel(ml)) {
                fwLog(fwlog::kDebug) << "DetailView "<< *it << " requires module label " <<  ml << std::endl;
                pass = false;
                break;
             }
          }
         }
         if (pass)  {
            returnValue.push_back(*it);
         }
         else {
            std::string::size_type first = (*it).find_first_of('@');
            std::string vn = *it;
            vn.insert(++first, "!");
            returnValue.push_back(vn);
         }
      }
   }

   std::vector<std::string>::iterator it;
   it = std::unique (returnValue.begin(), returnValue.end(), pluginComapreFunc);
   returnValue.resize( std::distance(returnValue.begin(),it) ); 

   m_typeToViewers[iType]=returnValue;
   return returnValue;
}

void
FWDetailViewManager::colorsChanged()
{
   for (vViews_i i = m_views.begin(); i !=  m_views.end(); ++i)
      (*i).m_detailView->setBackgroundColor(m_context->colorManager()->background());
}


void
FWDetailViewManager::newEventCallback()
{
   while (!m_views.empty())
   {  
      m_views.front().m_eveWindow->DestroyWindowAndSlot();
   }
}

void
FWDetailViewManager::eveWindowDestroyed(TEveWindow* ew)
{
   for (vViews_i i = m_views.begin(); i != m_views.end(); ++i)
   {
      if (ew == i->m_eveWindow)
      {
         // printf("========================== delete %s \n", ew->GetElementName());
         delete i->m_detailView;
         m_views.erase(i);
         break;
      }
   }
}
