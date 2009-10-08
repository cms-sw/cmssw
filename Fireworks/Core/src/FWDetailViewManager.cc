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
// $Id: FWDetailViewManager.cc,v 1.48 2009/10/07 19:18:01 amraktad Exp $
//

#include <stdio.h>
#include <boost/bind.hpp>

#include "TClass.h"
#include "TGWindow.h"
#include "TGFrame.h"
#include "TEveWindow.h"
#include "TEveManager.h"
#include "TEveWindowManager.h"

#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWDetailViewFactory.h"
#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWRepresentationInfo.h"

class DetailViewFrame : public TGMainFrame
{
public:
   DetailViewFrame():
      TGMainFrame(gClient->GetRoot(), 790, 450)
   {
   };

   virtual void CloseWindow()
   {
      UnmapWindow();
   }
};


//
// constructors and destructor
//
FWDetailViewManager::FWDetailViewManager(FWColorManager* colMng):
   m_colorManager(colMng),
   m_mainFrame(0),
   m_eveFrame(0),
   m_detailView(0)
{
   m_colorManager->colorsHaveChanged_.connect(boost::bind(&FWDetailViewManager::colorsChanged,this));
   m_mainFrame = new DetailViewFrame();
   m_mainFrame->SetCleanup(kLocalCleanup);

   m_eveFrame = new TEveCompositeFrameInMainFrame(m_mainFrame, 0, m_mainFrame);
   // For now we want to keep a single detailed-view main-frame.
   // As TGMainFrame very reasonably emits the CloseWindow signal even if CloseWindow() is
   // overriden and does not close the window (DetailViewFrame does that and just unmaps the window).
   // So ... as we want to keep the main-frame, eve-frame-in-main-frame must not listen
   // to this signal as it gets emitted erroneously.
   // Probably the right place to fix this is in ROOT - but API and signal logick would have to change.
   m_mainFrame->Disconnect("CloseWindow()", m_eveFrame, "MainFrameClosed()");

   TEveWindowSlot* ew_slot = TEveWindow::CreateDefaultWindowSlot();
   ew_slot->PopulateEmptyFrame(m_eveFrame);

   m_mainFrame->AddFrame(m_eveFrame, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY));
}

FWDetailViewManager::~FWDetailViewManager()
{
   if (m_detailView != 0)
      m_detailView->getEveWindow()->DestroyWindow();
}

void
FWDetailViewManager::openDetailViewFor(const FWModelId &id)
{
   if (m_detailView != 0)
   {
      m_detailView->getEveWindow()->DestroyWindow();
      delete m_detailView;
   }

   // find the right viewer for this item
   std::string typeName = ROOT::Reflex::Type::ByTypeInfo(*(id.item()->modelType()->GetTypeInfo())).Name(ROOT::Reflex::SCOPED);
   std::string viewerName = findViewerFor(typeName);
   if(0==viewerName.size()) {
      std::cout << "FWDetailViewManager: don't know what detailed view to "
         "use for object " << id.item()->name() << std::endl;
      assert(0!=viewerName.size());
   }

   m_detailView = FWDetailViewFactory::get()->create(viewerName);

   TEveWindowSlot* ws  = (TEveWindowSlot*)(m_eveFrame->GetEveWindow());
   m_detailView->build(id, ws);
   m_mainFrame->SetWindowName(Form("%s Detail View [%d]", id.item()->name().c_str(), id.index()));
   m_mainFrame->MapSubwindows();
   m_mainFrame->Layout();
   m_mainFrame->MapWindow();

   colorsChanged();
}

bool
FWDetailViewManager::haveDetailViewFor(const FWModelId& iId) const
{
   std::string typeName = ROOT::Reflex::Type::ByTypeInfo(*(iId.item()->modelType()->GetTypeInfo())).Name(ROOT::Reflex::SCOPED);
   if(m_detailViews.end() == m_detailViews.find(typeName)) {
      return findViewerFor(typeName).size()!=0;
   }
   return true;
}

std::string
FWDetailViewManager::findViewerFor(const std::string& iType) const
{
   std::string returnValue;

   std::map<std::string,std::string>::const_iterator itFind = m_typeToViewers.find(iType);
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
         m_typeToViewers[iType]=*it;
         return *it;
      }
      //see if we match via inheritance
      FWSimpleRepresentationChecker checker(type,"");
      FWRepresentationInfo info = checker.infoFor(iType);
      if(closestMatch > info.proximity()) {
         closestMatch = info.proximity();
         returnValue=*it;
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
