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
// $Id: FWDetailViewManager.cc,v 1.25 2009/03/25 22:14:08 amraktad Exp $
//

// system include files
#include <stdio.h>
// #include <GL/gl.h>
#include <boost/bind.hpp>
#include "TGButton.h"
#include "TGFrame.h"
#include "TGLEmbeddedViewer.h"
#include "TGLScenePad.h"
#include "TGTextView.h"
// #include "TGLUtil.h"
#include "TGLLightSet.h"
#include "TEveManager.h"
#include "TEveScene.h"
#include "TEveViewer.h"

#include "TClass.h"
#include "TGLOrthoCamera.h"


// user include files
#include "Fireworks/Core/interface/FWDetailViewManager.h"
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "Fireworks/Core/interface/FWDetailViewFactory.h"

#include "Fireworks/Core/interface/FWSimpleRepresentationChecker.h"
#include "Fireworks/Core/interface/FWRepresentationInfo.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWDetailViewManager::FWDetailViewManager()
   : frame(0)
{
}

// FWDetailViewManager::FWDetailViewManager(const FWDetailViewManager& rhs)
// {
//    // do actual copying here;
// }

FWDetailViewManager::~FWDetailViewManager()
{
}

//
// assignment operators
//
// const FWDetailViewManager& FWDetailViewManager::operator=(const FWDetailViewManager& rhs)
// {
//   //An exception safe implementation is
//   FWDetailViewManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void FWDetailViewManager::close_wm ()
{
//      printf("mmmm, flaming death!\n");
   frame = 0;
}

void FWDetailViewManager::close_button ()
{
//      printf("mmmm, flaming death!\n");
   frame->CloseWindow();
   frame = 0;
}

void
FWDetailViewManager::openDetailViewFor(const FWModelId &id)
{
   printf("opening detail view for event item %s (%x), index %d\n",
          id.item()->name().c_str(), (unsigned int)id.item(), id.index());

   // make a frame
   if (frame != 0)
      frame->CloseWindow();

   frame = new // TGTransientFrame(0, gEve->GetBrowser(), 400, 400);
      TGMainFrame(0, 800, 600);
   frame->SetCleanup(kDeepCleanup);
   frame->SetWindowName(Form("%s Detail View",id.item()->name().c_str()));
   frame->SetIconName("Detail View Icon");
   frame->Connect("CloseWindow()", "FWDetailViewManager", this, "close_wm()");

   TGHorizontalFrame* hf = new TGHorizontalFrame(frame);
   frame->AddFrame(hf, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // text view 
   text_view = new TGTextView(hf,20,20);
   text_view->AddLine(Form("%s detail view:",  id.item()->name().c_str()));
   text_view->AddLine(Form("item[%d] index[%d]", (unsigned int)id.item(), id.index()));
   text_view->AddLine("");
   hf->AddFrame(text_view, new TGLayoutHints(kLHintsLeft|kLHintsTop |kLHintsExpandY));

   // viewer 
   TGLEmbeddedViewer* v = new TGLEmbeddedViewer(hf, 0, 0);
   nv = new TEveViewer();
   nv->SetGLViewer(v,v->GetFrame());
   nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   if ( TGLOrthoCamera* oCamera = dynamic_cast<TGLOrthoCamera*>( &(nv->GetGLViewer()->CurrentCamera()) ) )
      oCamera->SetEnableRotate(kTRUE);

   nv->GetGLViewer()->SetStyle(TGLRnrCtx::kOutline);
   nv->GetGLViewer()->SetClearColor(kBlack);
   // gEve->AddElement(nv, gEve->GetViewers());
   ns = gEve->SpawnNewScene("Detailed view");
   nv->AddScene(ns);
   hf->AddFrame(v->GetFrame(), new TGLayoutHints(kLHintsExpandX | kLHintsExpandY|kLHintsTop));

   // exit
   TGTextButton* exit_butt = new TGTextButton(frame, "Close");
   exit_butt->Resize(20, 20);
   exit_butt->Connect("Clicked()", "FWDetailViewManager", this, "close_button()");
   frame->AddFrame(exit_butt, new TGLayoutHints(kLHintsExpandX));
  
   // find the right viewer for this item
   std::string typeName = ROOT::Reflex::Type::ByTypeInfo(*(id.item()->modelType()->GetTypeInfo())).Name(ROOT::Reflex::SCOPED);
   std::map<std::string, FWDetailViewBase *>::iterator viewer =
      m_viewers.find(typeName);
   if (viewer == m_viewers.end()) {
      //Lookup the viewer plugin since we have not used it yet
      std::string viewerName = findViewerFor(typeName);
      if(0==viewerName.size()) {
         std::cout << "FWDetailViewManager: don't know what detailed view to "
         "use for object " << id.item()->name() << std::endl;
         assert(viewer != m_viewers.end());
      }
      FWDetailViewBase* view = FWDetailViewFactory::get()->create(viewerName);
      if(0!=view) {
         m_viewers[typeName]= view;
      } else {
         std::cout << "FWDetailViewManager:could not create detailed view for "
         "use for object " << id.item()->name() << std::endl;
         assert(viewer != m_viewers.end());
      }
      viewer =m_viewers.find(typeName);
   }

   // get better lighting
   //      TGLCapabilitySwitch sw(GL_LIGHTING, false);
   TGLLightSet *light_set = nv->GetGLViewer()->GetLightSet();
   //      light_set->SetLight(TGLLightSet::kLightFront	, false);
   //      light_set->SetLight(TGLLightSet::kLightTop	, true);
   //      light_set->SetLight(TGLLightSet::kLightBottom	, false);
   //      light_set->SetLight(TGLLightSet::kLightLeft	, false);
   //      light_set->SetLight(TGLLightSet::kLightRight	, false);
   //      light_set->SetLight(TGLLightSet::kLightMask	, false);
   light_set->SetLight(TGLLightSet::kLightSpecular, false);
   // run the viewer
   viewer->second->setTextView(text_view);
   viewer->second->setViewer(nv->GetGLViewer());
   TEveElement *list = viewer->second->build(id);
   text_view->AdjustWidth();

   if(0!=list) {
      gEve->AddElement(list, ns);
   }

   double rotation_center[3] = { 0, 0, 0 };
   //      nv->GetGLViewer()->SetPerspectiveCamera(TGLViewer::kCameraOrthoXOY, 5, 0, viewer->second->rotation_center, 0.5, 0 );
   nv->GetGLViewer()->SetPerspectiveCamera(TGLViewer::kCameraOrthoXOY, 1, 0, rotation_center, 0.5, 0 );
   nv->GetGLViewer()->CurrentCamera().Reset();
   nv->GetGLViewer()->SetPerspectiveCamera(TGLViewer::kCameraPerspXOY, 1, 0, rotation_center, 0.5, 0 );
   nv->GetGLViewer()->CurrentCamera().Reset();
   nv->GetGLViewer()->UpdateScene();

   frame->MapSubwindows();
   frame->Layout();
   frame->MapWindow();
}

//
// const member functions
//
bool
FWDetailViewManager::haveDetailViewFor(const FWModelId& iId) const
{
   std::string typeName = ROOT::Reflex::Type::ByTypeInfo(*(iId.item()->modelType()->GetTypeInfo())).Name(ROOT::Reflex::SCOPED);
   if(m_viewers.end() == m_viewers.find(typeName)) {
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

//
// static member functions
//
