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
// $Id: FWDetailViewManager.cc,v 1.28 2009/03/30 18:27:48 amraktad Exp $
//

// system include files
#include <stdio.h>
#include <boost/bind.hpp>

#include "TClass.h"
#include "TCanvas.h"
#include "TRootEmbeddedCanvas.h"
#include "TGButton.h"
#include "TGFrame.h"
#include "TLatex.h"
#include "TGLEmbeddedViewer.h"
#include "TGLScenePad.h"
#include "TGLLightSet.h"
#include "TGLOrthoCamera.h"
#include "TEveManager.h"
#include "TEveScene.h"
#include "TEveViewer.h"

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
FWDetailViewManager::FWDetailViewManager():
   m_scene(0),
   m_viewer(0),
   m_frame(0),
   m_latex(0)
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
   // destroy scene elements
   m_scene->Destroy();
   m_scene = 0;
   // embedded viewer is destroyed by closing main window
   m_viewer = 0;
   m_frame = 0;
}

void FWDetailViewManager::close_button ()
{
   // this functions calls close_mw, since it emits signal
   m_frame->CloseWindow();
   // but for some reason, we still have to zero this out by hand 
   m_frame = 0;
}

void
FWDetailViewManager::openDetailViewFor(const FWModelId &id)
{
   // printf("opening detail view for event item %s (%x), index %d\n",
   //       id.item()->name().c_str(), (unsigned int)id.item(), id.index());

   // make a frame
   if (m_frame != 0)
      m_frame->CloseWindow();

   m_frame = new // TGTransientFrame(0, gEve->GetBrowser(), 400, 400);
      TGMainFrame(0, 800, 600);
   m_frame->SetCleanup(kDeepCleanup);
   m_frame->SetWindowName(Form("%s Detail View [%d]",id.item()->name().c_str(), id.index()));
   m_frame->SetIconName("Detail View Icon");
   m_frame->Connect("CloseWindow()", "FWDetailViewManager", this, "close_wm()");

   TGHorizontalFrame* hf = new TGHorizontalFrame(m_frame);
   m_frame->AddFrame(hf, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // text view 
   TRootEmbeddedCanvas* ec = new TRootEmbeddedCanvas("Embeddedcanvas", hf, 220);
   hf->AddFrame(ec, new TGLayoutHints(kLHintsExpandY));
   m_latex = new TLatex(0.02, 0.970, Form("%s detail view:",  id.item()->name().c_str()));
   double fs = 0.07;
   m_latex->SetTextSize(fs);
   m_latex->Draw();
   m_latex->DrawLatex(0.02, 0.97 -fs*0.5, Form("index[%d]", id.index()));
   m_latex->DrawLatex(0.02, 0.97 -fs, Form("item[%d]",  (unsigned int)id.item()));

   // viewer 
   TGLEmbeddedViewer* v = new TGLEmbeddedViewer(hf, 0, 0);
   m_viewer = new TEveViewer();
   m_viewer->SetGLViewer(v,v->GetFrame());
   m_viewer->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   if ( TGLOrthoCamera* oCamera = dynamic_cast<TGLOrthoCamera*>( &(m_viewer->GetGLViewer()->CurrentCamera()) ) )
      oCamera->SetEnableRotate(kTRUE);

   m_scene = gEve->SpawnNewScene("Detailed view");
   m_viewer->AddScene(m_scene);
   hf->AddFrame(v->GetFrame(), new TGLayoutHints(kLHintsExpandX | kLHintsExpandY|kLHintsTop));

   // exit
   TGTextButton* exit_butt = new TGTextButton(m_frame, "Close");
   exit_butt->Resize(20, 20);
   exit_butt->Connect("Clicked()", "FWDetailViewManager", this, "close_button()");
   m_frame->AddFrame(exit_butt, new TGLayoutHints(kLHintsExpandX));
  
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

  // run the viewer
   viewer->second->setLatex(m_latex);
   viewer->second->setViewer(m_viewer->GetGLViewer());
   TEveElement *list = viewer->second->build(id);
   if(list) {
      gEve->AddElement(list, m_scene);
   }

   // setup
   ec->GetCanvas()->SetEditable(kFALSE);
   m_viewer->GetGLViewer()->SetStyle(TGLRnrCtx::kOutline);
   m_viewer->GetGLViewer()->SetClearColor(kBlack);
   TGLLightSet *light_set = m_viewer->GetGLViewer()->GetLightSet();
   light_set->SetLight(TGLLightSet::kLightSpecular, false);

   m_viewer->GetGLViewer()->CurrentCamera().Reset();
   m_viewer->GetGLViewer()->UpdateScene();

   // map GUI
   m_frame->MapSubwindows();
   m_frame->Layout();
   m_frame->MapWindow();
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
