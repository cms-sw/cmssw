// -*- C++ -*-
//
// Package:     Core
// Class  :     FWLegoViewBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:41 EST 2008
// $Id: FWLegoViewBase.cc,v 1.18 2010/10/01 09:45:20 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>

#include "TAxis.h"

#include "TGLViewer.h"
#include "TGLLightSet.h"
#include "TGLPerspectiveCamera.h"
#include "TGLOrthoCamera.h"
#include "TEveElement.h"
#include "TEveScene.h"
#include "TEveCalo.h"
#include "TEveTrans.h"
#include "TEveCaloLegoOverlay.h"

// user include files
#include "Fireworks/Core/interface/FWGLEventHandler.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/FWLegoViewBase.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWLegoViewBase::FWLegoViewBase(TEveWindowSlot* iParent, FWViewType::EType typeId) :
   FWEveView(iParent, typeId, 4),
   m_lego(0),
   m_overlay(0),
   m_autoRebin(this,"Auto rebin on zoom-out",false),
   m_pixelsPerBin(this, "Pixels per bin", 10., 1., 20.),
   m_projectionMode(this, "Projection", 0l, 0l, 2l),
   m_cell2DMode(this, "Cell2DMode", 1l, 1l, 2l),
   m_drawValuesIn2D(this,"Draw Cell2D threshold (pixels)",40l,16l,200l),
   m_showOverlay(this,"Draw scales", true)
{
   FWViewEnergyScale* caloScale = new FWViewEnergyScale(this);
   viewContext()->addScale("Calo", caloScale);

   viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   viewerGL()->GetLightSet()->SetUseSpecular(false);

   m_projectionMode.addEntry(0, "Auto");
   m_projectionMode.addEntry(1, "3D");
   m_projectionMode.addEntry(2, "2D");

   m_cell2DMode.addEntry(1, "Plain");
   m_cell2DMode.addEntry(2, "Outline");
   if (typeId == FWViewType::kLegoHF) m_cell2DMode.set(2); // different default for HF view

   m_autoRebin.changed_.connect(boost::bind(&FWLegoViewBase::setAutoRebin,this));
   m_pixelsPerBin.changed_.connect(boost::bind(&FWLegoViewBase::setPixelsPerBin,this));
   m_drawValuesIn2D.changed_.connect(boost::bind(&FWLegoViewBase::setFontSizein2D,this));
   m_showOverlay.changed_.connect(boost::bind(&FWLegoViewBase::showOverlay,this));
   m_projectionMode.changed_.connect(boost::bind(&FWLegoViewBase::setProjectionMode, this));
   m_cell2DMode.changed_.connect(boost::bind(&FWLegoViewBase::setCell2DMode, this));
}

FWLegoViewBase::~FWLegoViewBase()
{
   viewerGL()->RemoveOverlayElement(m_overlay);
   m_lego->Destroy();
}


TEveCaloViz*
FWLegoViewBase::getEveCalo() const
{
   return static_cast<TEveCaloViz*>(m_lego);
}

void 
FWLegoViewBase::setContext(const fireworks::Context& ctx)
{
   FWEveView::setContext(ctx);
  
   TEveCaloData* data;
   if (typeId() == FWViewType::kLegoHF)  {
      data = static_cast<TEveCaloData*>(ctx.getCaloDataHF());
   }
   else {
      data = static_cast<TEveCaloData*>(ctx.getCaloData());
   } 

   data->GetEtaBins()->SetTitleFont(120);
   data->GetEtaBins()->SetTitle("h");
   data->GetPhiBins()->SetTitleFont(120);
   data->GetPhiBins()->SetTitle("f");
   data->GetPhiBins()->SetLabelSize(0.02);
   data->GetEtaBins()->SetLabelSize(0.02);
   data->GetPhiBins()->SetTitleSize(0.03);
   data->GetEtaBins()->SetTitleSize(0.03);

   m_lego = new TEveCaloLego(data);
   m_lego->InitMainTrans();
   m_lego->RefMainTrans().SetScale(TMath::TwoPi(), TMath::TwoPi(), TMath::Pi());
   m_lego->Set2DMode((TEveCaloLego::E2DMode_e)m_cell2DMode.value());
   m_lego->SetDrawNumberCellPixels(m_drawValuesIn2D.value());
   m_lego->SetAutoRebin(m_autoRebin.value());
   m_lego->SetPixelsPerBin(m_pixelsPerBin.value());

   // note, do not restore max tower height, since it has not value
   FWViewEnergyScale*  caloScale = viewContext()->getEnergyScale("Calo");
   m_lego->SetScaleAbs(caloScale->getScaleMode() == FWViewEnergyScale::kFixedScale);
   m_lego->SetMaxValAbs(caloScale->getMaxFixedVal());
   
   
   // set flat in 2D
   m_lego->SetHasFixedHeightIn2DMode(true);
   m_lego->SetFixedHeightValIn2DMode(0.001);
   eventScene()->AddElement(m_lego);

   // possiblity for outline
   m_lego->SetPlotEt(caloScale->getPlotEt());

   TEveLegoEventHandler* eh = dynamic_cast<TEveLegoEventHandler*>( viewerGL()->GetEventHandler());
   eh->SetLego(m_lego);
  
   m_overlay = new TEveCaloLegoOverlay();
   m_overlay->SetCaloLego(m_lego);
   m_overlay->SetShowPlane(kFALSE);
   m_overlay->SetScalePosition(0.8, 0.6);
   m_overlay->SetShowScales(1); //temporary
   viewerGL()->AddOverlayElement(m_overlay);
}
   
void
FWLegoViewBase::setAutoRebin()
{
   m_lego->SetAutoRebin(m_autoRebin.value());
   m_lego->ElementChanged(kTRUE,kTRUE);
}

void
FWLegoViewBase::setPixelsPerBin()
{
   m_lego->SetPixelsPerBin((Int_t) (m_pixelsPerBin.value()));
   m_lego->ElementChanged(kTRUE,kTRUE);
}

void
FWLegoViewBase::showOverlay()
{
   if (m_overlay) m_overlay->SetShowScales(m_showOverlay.value());
   viewerGL()->RequestDraw();
}

//_______________________________________________________________________________

void
FWLegoViewBase::setFrom(const FWConfiguration& iFrom)
{
   FWEveView::setFrom(iFrom);

   // cell 2D style
   if (iFrom.version() < 5)
   {
      const FWConfiguration* value = iFrom.valueForKey( "Cell2DMode" );
      if ( value !=  0 )
      {
         int mode;
         std::istringstream s(value->value());
         s>> mode;
         m_cell2DMode.set(mode);
      }
  
   }

   // view controller parameters, changed name in version 4
   if (iFrom.version() < 4)
   {
      bool xb;/* double xd;
      {
         std::istringstream s(iFrom.valueForKey("Lego auto scale")->value());
         s >> xb; m_energyScaleMode.set(xb ? FWEveView::kAutoScale : FWEveView::kFixedScale);
      }
      {
         std::istringstream s(iFrom.valueForKey("Lego scale GeV)")->value());
         s >> xd; m_energyMaxAbsVal.set(xd);
         }*/
      {
         std::istringstream s(iFrom.valueForKey("Show scales")->value());
         s >> xb; m_showOverlay.set(xb);
      }
      {
         std::istringstream s(iFrom.valueForKey("Show scales")->value());
         s >> xb; m_showOverlay.set(xb);
      }
      {
         std::istringstream s(iFrom.valueForKey("Auto rebin on zoom")->value());
         s >> xb; m_autoRebin.set(xb);
      }
   }

   //
   // camera restore

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
FWLegoViewBase::addTo(FWConfiguration& iTo) const
{
   FWEveView::addTo(iTo);
   
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

void
FWLegoViewBase::setFontSizein2D()
{
   m_lego->SetDrawNumberCellPixels( m_drawValuesIn2D.value());
   m_lego->ElementChanged(kTRUE,kTRUE);
}

void
FWLegoViewBase::setProjectionMode()
{
   m_lego->SetProjection((TEveCaloLego::EProjection_e)m_projectionMode.value());
   m_lego->ElementChanged();
   viewerGL()->RequestDraw();
}

void
FWLegoViewBase::setCell2DMode()
{
   m_lego->Set2DMode((TEveCaloLego::E2DMode_e)m_cell2DMode.value());
   m_lego->ElementChanged();
   viewerGL()->RequestDraw();
}

void 
FWLegoViewBase::populateController(ViewerParameterGUI& gui) const
{
   FWEveView::populateController(gui);

   gui.requestTab("Style").
      separator().
      addParam(&m_projectionMode).
      addParam(&m_cell2DMode).
      addParam(&m_drawValuesIn2D);
  
   gui.requestTab("Scales").
      separator().
      addParam(&m_showOverlay);

   gui.requestTab("Rebin").
      addParam(&m_autoRebin).
      addParam(&m_pixelsPerBin);
}

