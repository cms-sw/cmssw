// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveLegoView
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:41 EST 2008
// $Id: FWEveLegoView.cc,v 1.2 2008/03/20 09:39:26 dmytro Exp $
//

// system include files
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <iostream>

#include "TRootEmbeddedCanvas.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TClass.h"
#include "TH2F.h"
#include "TView.h"
#include "TColor.h"
#include "TEveScene.h"
#include "TGLViewer.h"
#include "TGLEmbeddedViewer.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveCalo.h"
#include "TEveElement.h"
#include "TEveRGBAPalette.h"

// user include files
#include "Fireworks/Core/interface/FWEveLegoView.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEveLegoView::FWEveLegoView(TGFrame* iParent, TEveElementList* list):
 m_range(this,"energy threshold (%)",0.,0.,100.)
{
   m_pad = new TEvePad;
   TGLEmbeddedViewer* ev = new TGLEmbeddedViewer(iParent, m_pad);
   m_embeddedViewer=ev;
   TEveViewer* nv = new TEveViewer(staticTypeName().c_str());
   nv->SetGLViewer(ev);
   nv->IncDenyDestroy();
   // ev->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   ev->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
   TEveScene* ns = gEve->SpawnNewScene(staticTypeName().c_str());
   m_scene = ns;
   nv->AddScene(ns);
   m_viewer=nv;
   
   TEveRGBAPalette* pal = new TEveRGBAPalette(0, 100);
   // pal->SetLimits(0, data->GetMaxVal());
   pal->SetLimits(0, 100);
   pal->SetDefaultColor((Color_t)1000);
   
   m_lego = new TEveCaloLego();
   m_lego->SetPalette(pal);
   m_lego->SetMainColor(Color_t(TColor::GetColor("#1A1A1A")));
   // lego->SetEtaLimits(etaLimLow, etaLimHigh);
   // lego->SetTitle("caloTower Et distribution");
   gEve->AddElement(m_lego, ns);
   gEve->AddToListTree(m_lego, kTRUE);
   gEve->AddElement(list,ns);
   gEve->AddToListTree(list, kTRUE);
   m_range.changed_.connect(boost::bind(&FWEveLegoView::doMinThreshold,this,_1));
}

FWEveLegoView::~FWEveLegoView()
{
}

void
FWEveLegoView::draw(TEveCaloDataHist* data)
{
   bool firstTime = (m_lego->GetData() == 0);
   m_lego->SetData(data);
   m_lego->ElementChanged();
   m_lego->InvalidateCache();
   if ( firstTime ) {
      m_scene->Repaint();
      m_viewer->Redraw(kTRUE);
      m_viewer->GetGLViewer()->ResetCurrentCamera();
   }
   m_viewer->GetGLViewer()->RequestDraw();
}

void 
FWEveLegoView::doMinThreshold(double value)
{
   m_lego->GetPalette()->SetMin( int(value) );
   m_lego->ElementChanged();
   m_lego->InvalidateCache();
   m_viewer->GetGLViewer()->RequestDraw();
}


//
// const member functions
//
TGFrame* 
FWEveLegoView::frame() const
{
   return m_embeddedViewer->GetFrame();
}

const std::string& 
FWEveLegoView::typeName() const
{
   return staticTypeName();
}

//
// static member functions
//
const std::string& 
FWEveLegoView::staticTypeName()
{
   static std::string s_name("3D Lego Pro");
   return s_name;
}

