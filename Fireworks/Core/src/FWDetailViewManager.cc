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
// $Id: FWDetailViewManager.cc,v 1.38 2009/07/08 15:51:02 amraktad Exp $
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
#include "TGPack.h"

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
   m_modeGL(kTRUE),

   m_mainFrame(0),
   m_pack(0),

   m_textCanvas(0),
   m_viewCanvas(0),
   m_sceneGL(0),
   m_viewerGL(0)
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


//______________________________________________________________________________
void
FWDetailViewManager::openDetailViewFor(const FWModelId &id)
{
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
   } else {
      m_detailView = detailViewBaseIt->second;
   }

   //setup GUI 
   if (m_mainFrame == 0)
      createDetailViewFrame();

   if (m_detailView->useGL() != m_modeGL) {
      m_modeGL= m_detailView->useGL();
      if (m_modeGL) {
         m_pack->HideFrame(m_viewCanvas);
         m_pack->ShowFrame(m_viewerGL->GetFrame());
      }
      else {
         m_pack->ShowFrame(m_viewCanvas);
         m_pack->HideFrame(m_viewerGL->GetFrame());
      }
   }

   // clean after previous detail view
   m_textCanvas->GetCanvas()->GetListOfPrimitives()->Delete();
   if (m_modeGL) {
      m_detailView->clearOverlayElements();
      m_sceneGL->DestroyElements();
   }
   else {
      m_viewCanvas->GetCanvas()->GetListOfPrimitives()->Delete();
      m_textCanvas ->GetCanvas()->SetEditable(kTRUE);
   }

   // build
   m_textCanvas->GetCanvas()->cd();
   m_textCanvas ->GetCanvas()->SetEditable(kTRUE);
   TLatex* latex = new TLatex(0.02, 0.970, Form("%s detail view:",  id.item()->name().c_str()));
   double fs = 0.06;
   latex->SetTextSize(fs);
   latex->Draw();
   latex->DrawLatex(0.02, 0.97 -fs*0.5, Form("index[%d]", id.index()));
   latex->DrawLatex(0.02, 0.97 -fs, Form("item[%d]",  (size_t)id.item()));
   m_detailView->setViewer(m_viewerGL);
   m_detailView->setTextCanvas(m_textCanvas->GetCanvas());
   m_detailView->setViewCanvas(m_viewCanvas->GetCanvas());
   TEveElement *list = m_detailView->build(id);
   if (m_modeGL)
   {
      if (list) gEve->AddElement(list, m_sceneGL);
      m_viewerGL->UpdateScene();
      m_viewerGL->CurrentCamera().Reset();
   }
   else
   {
      m_viewCanvas->GetCanvas()->Update();
      m_viewCanvas ->GetCanvas()->SetEditable(kFALSE);
   }
   m_textCanvas->GetCanvas()->SetEditable(kFALSE);

   m_textCanvas->GetCanvas()->Update();

   m_mainFrame->SetWindowName(Form("%s Detail View [%d]", id.item()->name().c_str(), id.index()));
   m_mainFrame->MapSubwindows();
   m_mainFrame->MapWindow();
   fflush(stdout);
}

//______________________________________________________________________________
void
FWDetailViewManager::createDetailViewFrame()
{
   m_mainFrame = new  DetailViewFrame(0, 800, 600);
   Float_t leftW = 2;
   Float_t rightW = 5;

   m_pack = new TGPack(m_mainFrame);
   m_mainFrame->AddFrame(m_pack, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   m_pack->SetVertical(kFALSE);
   m_pack->SetUseSplitters(kFALSE);

   // text view
   m_textCanvas = new TRootEmbeddedCanvas("Embeddedcanvas", m_pack);
   m_textCanvas->GetCanvas()->SetHighLightColor(-1);
   m_pack->AddFrameWithWeight(m_textCanvas, 0, leftW);

   // viewer
   m_viewerGL = new TGLEmbeddedViewer(m_pack, 0, 0);
   TEveViewer* eveViewer= new TEveViewer("DetailViewViewer");
   eveViewer->SetGLViewer(m_viewerGL, m_viewerGL->GetFrame());
   m_pack->AddFrameWithWeight(m_viewerGL->GetFrame(),0, rightW);

   m_viewerGL->SetStyle(TGLRnrCtx::kOutline);
   m_viewerGL->SetClearColor(kBlack);
   TGLLightSet *light_set = m_viewerGL->GetLightSet();
   light_set->SetLight(TGLLightSet::kLightSpecular, false);
   m_sceneGL = gEve->SpawnNewScene("Detailed view");
   eveViewer->AddScene(m_sceneGL);

   // 2D canvas
   m_viewCanvas   = new TRootEmbeddedCanvas("Embeddedcanvas", m_pack);
   m_viewCanvas->GetCanvas()->SetHighLightColor(-1);
   m_pack->AddFrameWithWeight(m_viewCanvas, 0, rightW);
   m_pack->HideFrame(m_viewCanvas);
   m_modeGL = true;

   // map GUI
   m_mainFrame->MapSubwindows();
   m_mainFrame->Layout();
   m_mainFrame->MapWindow();
}
//______________________________________________________________________________

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
