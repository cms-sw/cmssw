
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
// $Id: FWEveLegoView.cc,v 1.75 2010/04/09 19:49:28 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>

#include "TAxis.h"

#include "TGLViewer.h"
#include "TGLPerspectiveCamera.h"
#include "TGLOrthoCamera.h"
#include "TGLWidget.h"

#include "TEveElement.h"
#include "TEveManager.h"
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
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/Context.h"


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
   viewer()->AddScene(eventScene);

   viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);

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

void FWEveLegoView::setContext(fireworks::Context& context)
{ 
   // get lego if exist
   TEveCaloData* data = context.getCaloData();
   for (TEveElement::List_i i = data->BeginChildren(); i!= data->EndChildren(); ++i)
   {
      if( dynamic_cast<TEveCaloLego*>(*i))
      {
         m_lego = dynamic_cast<TEveCaloLego*>(*i);
         break;
      }
   }
   // else create 
   if (m_lego == 0)
   {   
      m_lego = new TEveCaloLego(data);
      m_lego->InitMainTrans();
      m_lego->RefMainTrans().SetScale(TMath::TwoPi(), TMath::TwoPi(), TMath::Pi());
      m_lego->Set2DMode(TEveCaloLego::kValSize);
      m_lego->SetDrawNumberCellPixels(20);
      gEve->AddElement(m_lego);
      data->GetEtaBins()->SetTitleFont(120);
      data->GetEtaBins()->SetTitle("h");
      data->GetPhiBins()->SetTitleFont(120);
      data->GetPhiBins()->SetTitle("f");
      data->GetPhiBins()->SetLabelSize(0.02);
      data->GetEtaBins()->SetLabelSize(0.02);
      data->GetPhiBins()->SetTitleSize(0.03);
      data->GetEtaBins()->SetTitleSize(0.03);

      // add calorimeter boundaries
      TEveStraightLineSet* boundaries = new TEveStraightLineSet("boundaries");
      boundaries->SetPickable(kFALSE);
      boundaries->SetLineWidth(2);
      boundaries->SetLineStyle(7);
      boundaries->AddLine(-1.479,-3.1416,0.001,-1.479,3.1416,0.001);
      boundaries->AddLine(1.479,-3.1416,0.001,1.479,3.1416,0.001);
      boundaries->AddLine(-2.964,-3.1416,0.001,-2.964,3.1416,0.001);
      boundaries->AddLine(2.964,-3.1416,0.001,2.964,3.1416,0.001);
      boundaries->SetLineColor(context.colorManager()->geomColor(kFWLegoBoundraryColorIndex));
      m_lego->AddElement(boundaries);
   }
   eventScene()->AddElement(m_lego);

   FWGLEventHandler* eh = new FWGLEventHandler((TGWindow*)viewerGL()->GetGLWidget(), (TObject*)viewerGL());
   eh->SetLego(m_lego);     
   viewerGL()->SetEventHandler(eh);
  
   m_overlay = new TEveCaloLegoOverlay();
   m_overlay->SetCaloLego(m_lego);
   m_overlay->SetShowPlane(kFALSE);
   m_overlay->SetScalePosition(0.8, 0.6);
   m_overlay->SetShowScales(1); //temporary
   viewerGL()->AddOverlayElement(m_overlay);
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

   if (iFrom.version() > 1)
   {
      bool topView = true;
      std::string stateName("topView"); stateName += typeName();
      assert( 0 != iFrom.valueForKey(stateName));
      std::istringstream s(iFrom.valueForKey(stateName)->value());
      s >> topView;
   
      if (topView)
      {
         viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
         TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>( &(viewerGL()->RefCamera(TGLViewer::kCameraOrthoXOY)) );
         setFromOrthoCamera(camera, iFrom);
      }
      else
      {
         viewerGL()->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
         TGLPerspectiveCamera* camera = dynamic_cast<TGLPerspectiveCamera*>(&(viewerGL()->RefCamera(TGLViewer::kCameraPerspXOY)));
         setFromPerspectiveCamera(camera, typeName(), iFrom);
      }
   }
   else
   {
      // reset camera if version not supported    
      viewerGL()->ResetCamerasAfterNextUpdate();
   }
   
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

