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
// $Id: FWLegoViewBase.cc,v 1.4 2010/06/18 10:17:15 yana Exp $
//

// system include files
#include <boost/bind.hpp>

#include "TAxis.h"

#include "TGLViewer.h"
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
   FWEveView(iParent, typeId),
   m_lego(0),
   m_overlay(0),
   m_plotEt(this,"Plot Et",true),   
   m_autoRebin(this,"Auto rebin on zoom",false),
   m_pixelsPerBin(this, "Pixels per bin", 10., 1., 20.),
   m_drawValuesIn2D(this,"pixel font size in 2D)",40l,16l,200l),
   m_showScales(this,"Show scales", true),
   m_legoFixedScale(this,"Lego scale GeV)",100.,1.,1000.),
   m_legoAutoScale (this,"Lego auto scale",true)
{
   FWViewEnergyScale* caloScale = new FWViewEnergyScale();
   viewContext()->addScale("Calo", caloScale);

   viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);

   m_autoRebin.changed_.connect(boost::bind(&FWLegoViewBase::setAutoRebin,this));
   m_pixelsPerBin.changed_.connect(boost::bind(&FWLegoViewBase::setPixelsPerBin,this));
   m_drawValuesIn2D.changed_.connect(boost::bind(&FWLegoViewBase::setFontSizein2D,this));
   m_plotEt.changed_.connect(boost::bind(&FWLegoViewBase::plotEt,this));
   m_showScales.changed_.connect(boost::bind(&FWLegoViewBase::showScales,this));
   m_legoFixedScale.changed_.connect(boost::bind(&FWLegoViewBase::updateLegoScale, this));
   m_legoAutoScale .changed_.connect(boost::bind(&FWLegoViewBase::updateLegoScale, this));
}

FWLegoViewBase::~FWLegoViewBase()
{
   m_lego->Destroy();
}

void FWLegoViewBase::setContext(fireworks::Context& context)
{ 
   TEveCaloData* data = getCaloData(context);
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
   m_lego->Set2DMode(TEveCaloLego::kValSize);
   m_lego->SetDrawNumberCellPixels(20);
   m_lego->SetAutoRebin(false);
   eventScene()->AddElement(m_lego);
 
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
FWLegoViewBase::plotEt()
{
   m_lego->SetPlotEt(m_plotEt.value());
   viewerGL()->RequestDraw();
}

void
FWLegoViewBase::showScales()
{
   if (m_overlay) m_overlay->SetShowScales(m_showScales.value());
   viewerGL()->RequestDraw();
}

void
FWLegoViewBase::updateLegoScale()
{
   m_lego->SetMaxValAbs( m_legoFixedScale.value() );
   m_lego->SetScaleAbs ( ! m_legoAutoScale.value() );
   m_lego->ElementChanged(kTRUE,kTRUE);
}

//_______________________________________________________________________________

void
FWLegoViewBase::setFrom(const FWConfiguration& iFrom)
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
FWLegoViewBase::eventBegin()
{
   viewContext()->resetScale();
}

void
FWLegoViewBase::eventEnd()
{
   FWEveView::eventEnd();
   viewContext()->scaleChanged();

}
