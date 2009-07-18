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
// $Id: FWDetailViewManager.cc,v 1.32 2009/06/05 19:59:24 amraktad Exp $
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

class DetailViewFrame : public TGMainFrame
{
public:
  DetailViewFrame(const TGWindow *p = 0,UInt_t w = 1,UInt_t h = 1): TGMainFrame(p, w, h) {};

  virtual void CloseWindow()
  {
    UnmapWindow();
  }
};

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
   m_detailView(0),

   m_scene(0),
   m_viewer(0),
   m_frame(0),
   m_latexCanvas(0)
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
// member functions
//

void FWDetailViewManager::close_button ()
{
   m_frame->UnmapWindow();
}

void
FWDetailViewManager::openDetailViewFor(const FWModelId &id)
{
   //printf("opening detail view for event item %s (%x), index %d\n",
   //      id.item()->name().c_str(), (unsigned int)id.item(), id.index());

   if (m_frame == 0)
      createDetailViewFrame();

   if (m_detailView)
   {
      m_detailView->clearOverlayElements();
      m_scene->DestroyElements();
      m_detailView = 0;
   }
   m_frame->SetWindowName(Form("%s Detail View [%d]", id.item()->name().c_str(), id.index()));

   // update latext
   m_latexCanvas->GetListOfPrimitives()->Delete();
   m_latexCanvas ->SetEditable(kTRUE);
   m_latexCanvas->cd();
   TLatex* latex = new TLatex(0.02, 0.970, Form("%s detail view:",  id.item()->name().c_str()));
   double fs = 0.07;
   latex->SetTextSize(fs);
   latex->Draw();
   latex->DrawLatex(0.02, 0.97 -fs*0.5, Form("index[%d]", id.index()));
   latex->DrawLatex(0.02, 0.97 -fs, Form("item[%d]",  (size_t)id.item()));

   // find the right viewer for this item
   std::string typeName = ROOT::Reflex::Type::ByTypeInfo(*(id.item()->modelType()->GetTypeInfo())).Name(ROOT::Reflex::SCOPED);
   std::map<std::string, FWDetailViewBase *>::iterator detailViewBaseIt = m_detailViews.find(typeName);

   if ( detailViewBaseIt  == m_detailViews.end()) 
   {
      //Lookup the viewer plugin since we have not used it yet
      std::string viewerName = findViewerFor(typeName);
      if(0==viewerName.size()) {
         std::cout << "FWDetailViewManager: don't know what detailed view to "
            "use for object " << id.item()->name() << std::endl;
         assert(detailViewBaseIt != m_detailViews.end());
      }
      m_detailView = FWDetailViewFactory::get()->create(viewerName);
      if( m_detailView) {
         m_detailViews[typeName]= m_detailView;
      } else {
         std::cout << "FWDetailViewManager:could not create detailed view for "
            "use for object " << id.item()->name() << std::endl;
         assert( detailViewBaseIt != m_detailViews.end());
      }
   }
   else
   {
      m_detailView = detailViewBaseIt->second;
   }

   // run the viewer
   m_detailView->setLatex(latex);
   m_detailView ->setViewer(m_viewer);
   TEveElement *list = m_detailView->build(id);
   m_latexCanvas->SetEditable(kFALSE);
   if(list)  gEve->AddElement(list, m_scene);
   
   m_frame->MapWindow();
   m_viewer->UpdateScene();
   m_viewer->CurrentCamera().Reset();
}

void
FWDetailViewManager::createDetailViewFrame()
{
   m_frame = new  DetailViewFrame(0, 800, 600);
   m_frame->SetCleanup(kDeepCleanup);
   m_frame->SetIconName("Detail View Icon");

   TGHorizontalFrame* hf = new TGHorizontalFrame(m_frame);
   m_frame->AddFrame(hf, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   // text view 
   TRootEmbeddedCanvas *rec   = new TRootEmbeddedCanvas("Embeddedcanvas", hf, 220);
   m_latexCanvas = rec->GetCanvas();
   hf->AddFrame(rec, new TGLayoutHints(kLHintsExpandY));
 
   // viewer 
   m_viewer = new TGLEmbeddedViewer(hf, 0, 0);
   TEveViewer* eveViewer= new TEveViewer("DetailViewViewer");
   //   eveViewer->AddElement(eveViewer);
   eveViewer->SetGLViewer(m_viewer, m_viewer->GetFrame());
   m_viewer->SetStyle(TGLRnrCtx::kOutline);
   m_viewer->SetClearColor(kBlack);
   TGLLightSet *light_set = m_viewer->GetLightSet();
   light_set->SetLight(TGLLightSet::kLightSpecular, false);

   //scene
   m_scene = gEve->SpawnNewScene("Detailed view");
   eveViewer->AddScene(m_scene);

   hf->AddFrame(m_viewer->GetFrame(), new TGLayoutHints(kLHintsExpandX | kLHintsExpandY|kLHintsTop));

   // exit
   TGTextButton* exit_butt = new TGTextButton(m_frame, "Close");
   exit_butt->Resize(20, 20);
   exit_butt->Connect("Clicked()", "FWDetailViewManager", this, "close_button()");
   m_frame->AddFrame(exit_butt, new TGLayoutHints(kLHintsExpandX));
  
   // map GUI
   m_frame->MapSubwindows();
   m_frame->Layout();
}

//
// const member functions
//
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

//
// static member functions
//
