// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTableView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:41 EST 2008
// $Id: FWTableView.cc,v 1.2 2009/04/08 14:55:53 jmuelmen Exp $
//

// system include files
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <iostream>
#include <sstream>
#include <stdexcept>

// FIXME
// need camera parameters
#define private public
#include "TGLPerspectiveCamera.h"
#undef private


#include "TRootEmbeddedCanvas.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TH2F.h"
#include "TView.h"
#include "TColor.h"
#include "TEveScene.h"
#include "TGLViewer.h"
//EVIL, but only way I can avoid a double delete of TGLEmbeddedViewer::fFrame
#define private public
#include "TGLEmbeddedViewer.h"
#undef private
#include "TGTextView.h"
#include "TGTextEntry.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveWindow.h"
#include "TEveElement.h"
#include "TEveCalo.h"
#include "TEveElement.h"
#include "TEveRGBAPalette.h"
#include "TEveLegoEventHandler.h"
#include "TGLWidget.h"
#include "TGLScenePad.h"
#include "TGLFontManager.h"
#include "TEveTrans.h"
#include "TGeoTube.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"
#include "TEveText.h"
#include "TGeoArb8.h"

// user include files
#include "Fireworks/Core/interface/FWTableView.h"
#include "Fireworks/Core/interface/FWEveValueScaler.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/BuilderUtils.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
//double FWTableView::m_scale = 1;

//
// constructors and destructor
//
FWTableView::FWTableView(TEveWindowSlot* iParent)
     : m_frame(0)
{
//      TGLayoutHints *tFrameHints = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
     const int width = 100, height = 100;
//      TGVerticalFrame *topframe = new TGVerticalFrame(iParent->GetEveFrame(), 100, 100);
     m_frame = iParent->MakeFrame(0);
     TGCompositeFrame *frame = m_frame->GetGUICompositeFrame();
     TGTextView *text = new TGTextView(frame, width, height, "Blah blah blah blah blah");
     frame->AddFrame(text, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
     frame->MapSubwindows();
     frame->Layout();
     frame->MapWindow();
//      frame->Resize(0,0);

//      iParent->GetEveFrame()->Layout();
//      iParent->GetEveFrame()->MapSubwindows();
// #if 1
//      TGHorizontalFrame *buttons = new TGHorizontalFrame(topframe, width, 25, kFixedHeight);
//      topframe->AddFrame(buttons, new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY));
//      TGTextEntry *text = new TGTextEntry(buttons, "Collection");
//      buttons->AddFrame(text, new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY));
// #else
//      TGTextView *text = new TGTextView(topframe, width, height, "Blah blah blah blah blah");
//      topframe->AddFrame(text, new TGLayoutHints(kLHintsTop | kLHintsExpandX | kLHintsExpandY));
//      topframe->Layout();
//      topframe->MapSubwindows();
// //      topframe->MapWindow();
// #endif
//      iParent->GetEveFrame()->AddFrame(frame,tFrameHints);
// //      parent->HideFrame(frame);
//      iParent->GetEveFrame()->Layout();
//      iParent->GetEveFrame()->MapSubwindows();
//      iParent->GetEveFrame()->MapWindow();
#if 0
   TEveViewer* nv = new TEveViewer(staticTypeName().c_str());
   m_embeddedViewer =  nv->SpawnGLEmbeddedViewer();
   iParent->ReplaceWindow(nv);

   TGLEmbeddedViewer* ev = m_embeddedViewer;
   ev->SetCurrentCamera(TGLViewer::kCameraPerspXOZ);
   //? ev->SetEventHandler(new TTableEventHandler("Lego", ev->GetGLWidget(), ev));
   m_cameraMatrix = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamTrans()));
   m_cameraMatrixBase = const_cast<TGLMatrix*>(&(ev->CurrentCamera().GetCamBase()));
   if ( TGLPerspectiveCamera* camera =
        dynamic_cast<TGLPerspectiveCamera*>(&(ev->CurrentCamera())) )
      m_cameraFOV = &(camera->fFOV);

   TEveScene* ns = gEve->SpawnNewScene(staticTypeName().c_str());
   m_scene = ns;
   nv->AddScene(ns);
   m_viewer=nv;
   gEve->AddElement(nv, gEve->GetViewers());
   gEve->AddElement(list,ns);
   gEve->AddToListTree(list, kTRUE);
#endif
}

FWTableView::~FWTableView()
{
     m_frame->DestroyWindowAndSlot();
}

void
FWTableView::setFrom(const FWConfiguration& iFrom)
{
   // take care of parameters
   FWConfigurableParameterizable::setFrom(iFrom);
#if 0

   // retrieve camera parameters

   // transformation matrix
   assert(m_cameraMatrix);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + "Table" );
      if (!value ) continue;
      std::istringstream s(value->value());
      s>>((*m_cameraMatrix)[i]);
   }

   // transformation matrix base
   assert(m_cameraMatrixBase);
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + "Table" );
      if (!value ) continue;
      std::istringstream s(value->value());
      s>>((*m_cameraMatrixBase)[i]);
   }

   {
      assert ( m_cameraFOV );
      const FWConfiguration* value = iFrom.valueForKey( "Table FOV" );
      if ( value ) {
         std::istringstream s(value->value());
         s>>*m_cameraFOV;
      }
   }
   m_viewer->GetGLViewer()->RequestDraw();
#endif
}

void
FWTableView::setBackgroundColor(Color_t iColor) 
{
//    m_viewer->GetGLViewer()->SetClearColor(iColor);
}

//
// const member functions
//
TGFrame*
FWTableView::frame() const
{
     return 0;
//    return m_embeddedViewer->GetFrame();
}

const std::string&
FWTableView::typeName() const
{
   return staticTypeName();
}

void
FWTableView::addTo(FWConfiguration& iTo) const
{
   // take care of parameters
   FWConfigurableParameterizable::addTo(iTo);
#if 0
   // store camera parameters

   // transformation matrix
   assert(m_cameraMatrix);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrix)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+"Table",FWConfiguration(osValue.str()));
   }

   // transformation matrix base
   assert(m_cameraMatrixBase);
   matrixName = "cameraMatrixBase";
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrixBase)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+"Table",FWConfiguration(osValue.str()));
   }
   {
      assert ( m_cameraFOV );
      std::ostringstream osValue;
      osValue << *m_cameraFOV;
      iTo.addKeyValue("Table FOV",FWConfiguration(osValue.str()));
   }
#endif
}

void
FWTableView::saveImageTo(const std::string& iName) const
{
//    bool succeeded = m_viewer->GetGLViewer()->SavePicture(iName.c_str());
//    if(!succeeded) {
//       throw std::runtime_error("Unable to save picture");
//    }
}

//
// static member functions
//
const std::string&
FWTableView::staticTypeName()
{
   static std::string s_name("Table");
   return s_name;
}

