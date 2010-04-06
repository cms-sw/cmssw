
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
// $Id: FWEveLegoView.cc,v 1.72 2010/03/28 21:36:59 matevz Exp $
//

// system include files
#include <boost/bind.hpp>

#include "TAttAxis.h"

#include "TGLViewer.h"
#include "TGLPerspectiveCamera.h"
#include "TGLOrthoCamera.h"
#include "TGLWidget.h"

#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TEveCalo.h"
#include "TEveTrans.h"
#include "TEveStraightLineSet.h"
#include "TEveCaloLegoOverlay.h"

// user include files
#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/BuilderUtils.h"
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
FWEveLegoView::FWEveLegoView(TEveWindowSlot* iParent, TEveScene* eventScene) :
   FWEveView(iParent),
   m_lego(0),
   m_overlay(0),
   m_plotEt(this,"Plot Et",true),   
   m_autoRebin(this,"Auto rebin on zoom",true),
   m_pixelsPerBin(this, "Pixels per bin", 10., 1., 20.),
   m_showScales(this,"Show scales", true),
   m_legoFixedScale(this,"Lego scale GeV)",100.,1.,1000.),
   m_legoAutoScale (this,"Lego auto scale",true)
{
   setType(FWViewType::kLego);
   setEventScene(eventScene);

   FWGLEventHandler* eh = new FWGLEventHandler((TGWindow*)viewerGL()->GetGLWidget(), (TObject*)viewerGL());
   viewerGL()->SetEventHandler(eh);

   if (eventScene->HasChildren())
   {
      m_lego =  dynamic_cast<TEveCaloLego*>( eventScene->FirstChild());
      if (m_lego) {
         m_overlay = new TEveCaloLegoOverlay();
         m_overlay->SetShowPlane(kFALSE);
         m_overlay->SetShowPerspective(kFALSE);
         m_overlay->GetAttAxis()->SetLabelSize(0.02);
         viewerGL()->AddOverlayElement(m_overlay);
         m_overlay->SetCaloLego(m_lego);
         m_overlay->SetShowScales(1); //temporary
         m_overlay->SetScalePosition(0.8, 0.6);
         eh->SetLego(m_lego);
      }
   }
   
   m_autoRebin.changed_.connect(boost::bind(&FWEveLegoView::setAutoRebin,this));
   m_pixelsPerBin.changed_.connect(boost::bind(&FWEveLegoView::setPixelsPerBin,this));
   m_plotEt.changed_.connect(boost::bind(&FWEveLegoView::plotEt,this));
   m_showScales.changed_.connect(boost::bind(&FWEveLegoView::showScales,this));
   m_legoFixedScale.changed_.connect(boost::bind(&FWEveLegoView::updateLegoScale, this));
   m_legoAutoScale .changed_.connect(boost::bind(&FWEveLegoView::updateLegoScale, this));
}

FWEveLegoView::~FWEveLegoView()
{
}


void
FWEveLegoView::setAutoRebin()
{
   if(m_lego) {
      m_lego->SetAutoRebin(m_autoRebin.value());
      m_lego->ElementChanged(kTRUE,kTRUE);
   }
}

void
FWEveLegoView::setPixelsPerBin()
{
   if(m_lego) {
      m_lego->SetPixelsPerBin((Int_t) (m_pixelsPerBin.value()));
      m_lego->ElementChanged(kTRUE,kTRUE);
   }
}

void
FWEveLegoView::plotEt()
{
   if (m_lego)
   {
      m_lego->SetPlotEt(m_plotEt.value());
      viewerGL()->RequestDraw();
   }
}

void
FWEveLegoView::showScales()
{
   if (m_overlay) m_overlay->SetShowScales(m_showScales.value());
   viewerGL()->RequestDraw();
}

void
FWEveLegoView::updateLegoScale()
{
  if (m_lego)
   {
      m_lego->SetMaxValAbs( m_legoFixedScale.value() );
      m_lego->SetScaleAbs ( ! m_legoAutoScale.value() );
      m_lego->ElementChanged(kTRUE,kTRUE);
   } 
}

//_______________________________________________________________________________

void
FWEveLegoView::setFrom(const FWConfiguration& iFrom)
{
   FWEveView::setFrom(iFrom);

   bool topView = viewerGL()->CurrentCamera().IsOrthographic();
   std::string stateName("topView"); stateName += typeName();
   assert( 0!=iFrom.valueForKey(stateName) );
   std::istringstream s(iFrom.valueForKey(stateName)->value());
   s >> topView;
   
   if (topView)
   {
      viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
      TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>( &(viewerGL()->RefCamera(TGLViewer::kCameraOrthoXOY)) );
      if (iFrom.version() > 1 )  setFromOrthoCamera(camera, iFrom);
   }
   else
   {
      viewerGL()->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
      TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>(&(viewerGL()->RefCamera(TGLViewer::kCameraPerspXOY)));
      if (iFrom.version() > 1 ) setFromPerspectiveCamera(camera, typeName(), iFrom);
   }

   viewerGL()->ResetCamerasAfterNextUpdate();
   
}

void
FWEveLegoView::addTo(FWConfiguration& iTo) const
{
   FWEveView::addTo(iTo);
   
   printf("addtoo version %d \n", iTo.version());

   bool topView =  viewerGL()->CurrentCamera().IsOrthographic();
   std::ostringstream s;
   s << topView;
   std::string name = "topView";
   iTo.addKeyValue(name+typeName(),FWConfiguration(s.str()));
   
   if (topView)
   {
      TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>(&(viewerGL()->RefCamera(TGLViewer::kCameraOrthoXOY)));
      addToOrthoCamera(camera, iTo);  
   }
   else
   {
      TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>(&(viewerGL()->RefCamera(TGLViewer::kCameraPerspXOY)));
      addToPerspectiveCamera(camera, typeName(), iTo);   
   }
}

