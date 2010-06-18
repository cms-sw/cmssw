// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRPZView
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Feb 19 10:33:25 EST 2008
// $Id: FWRPZView.cc,v 1.16 2010/06/18 10:17:16 yana Exp $
//

// system include files
#include <stdexcept>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>

#include "TGLViewer.h"
#include "TGLScenePad.h"
#include "TEveManager.h"
#include "TEveElement.h"
#include "TEveScene.h"
#include "TEveProjections.h"
#include "TEveProjectionManager.h"
#include "TEveProjectionAxes.h"
#include "TEveCalo.h"

// user include files
#include "Fireworks/Core/interface/FWRPZView.h"
#include "Fireworks/Core/interface/FWRPZViewGeometry.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//
// constructors and destructor
//
FWRPZView::FWRPZView(TEveWindowSlot* iParent, FWViewType::EType id) :
   FWEveView(iParent, id),
   m_calo(0),
   m_caloDistortion(this,"Calo compression",1.0,0.01,10.),
   m_muonDistortion(this,"Muon compression",0.2,0.01,10.),
   m_showProjectionAxes(this,"Show projection axes", false),
   m_compressMuon(this,"Compress detectors",false),
   m_caloFixedScale(this,"Calo scale (GeV/meter)",2.,0.01,100.),
   m_caloAutoScale(this,"Calo auto scale",false),
   m_showHF(0),
   m_showEndcaps(0)
{
   FWViewEnergyScale* caloScale = new FWViewEnergyScale();
   viewContext()->addScale("Calo", caloScale);

   TEveProjection::EPType_e projType = (id == FWViewType::kRhoZ) ? TEveProjection::kPT_RhoZ : TEveProjection::kPT_RPhi;

   m_projMgr.reset(new TEveProjectionManager(projType));
   m_projMgr->SetImportEmpty(kTRUE);
   if ( id == FWViewType::kRhoPhi) {
      m_projMgr->GetProjection()->AddPreScaleEntry(0, 130, 1.0);
      m_projMgr->GetProjection()->AddPreScaleEntry(0, 300, 0.2);
   } else {
      m_projMgr->GetProjection()->AddPreScaleEntry(0, 130, 1.0);
      m_projMgr->GetProjection()->AddPreScaleEntry(1, 310, 1.0);
      m_projMgr->GetProjection()->AddPreScaleEntry(0, 370, 0.2);
      m_projMgr->GetProjection()->AddPreScaleEntry(1, 580, 0.2);
   }

   // camera  
   viewerGL()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   if ( TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>( &(viewerGL()->CurrentCamera()) ) ) {
      camera->SetZoomMax(1e6);
   }
   geoScene()->GetGLScene()->SetSelectable(kFALSE);

   m_axes.reset(new TEveProjectionAxes(m_projMgr.get()));
   m_axes->SetRnrState(m_showProjectionAxes.value());
   m_showProjectionAxes.changed_.connect(boost::bind(&FWRPZView::showProjectionAxes,this));
   eventScene()->AddElement(m_axes.get());

   if ( id == FWViewType::kRhoPhi ) {
      m_showEndcaps = new FWBoolParameter(this,"Show calo endcaps", true);
      m_showEndcaps->changed_.connect(  boost::bind(&FWRPZView::updateCaloParameters, this) );
      m_showHF = new FWBoolParameter(this,"Show HF", true);
      m_showHF->changed_.connect(  boost::bind(&FWRPZView::updateCaloParameters, this) );
   }

   m_caloDistortion.changed_.connect(boost::bind(&FWRPZView::doDistortion,this));
   m_muonDistortion.changed_.connect(boost::bind(&FWRPZView::doDistortion,this));
   m_compressMuon.changed_.connect(boost::bind(&FWRPZView::doCompression,this,_1));
   m_caloFixedScale.changed_.connect( boost::bind(&FWRPZView::updateCaloParameters, this) );
   m_caloAutoScale.changed_.connect(  boost::bind(&FWRPZView::updateCaloParameters, this) );
}

FWRPZView::~FWRPZView()
{
   m_projMgr->DestroyElements();
}

//
// member functions
//

void
FWRPZView::setContext(fireworks::Context& context)
{
   // detector
   FWRPZViewGeometry* geo = new FWRPZViewGeometry(context.getGeom(), context.colorManager());
   m_projMgr->ImportElements(geo->getGeoElements(typeId()), geoScene());

   // calo
   TEveCaloData* data = context.getCaloData();
   TEveCalo3D* calo3d = new TEveCalo3D(data);
   m_calo = static_cast<TEveCalo2D*> (m_projMgr->ImportElements(calo3d, eventScene()));
   
   m_calo->SetBarrelRadius(129);
   m_calo->SetEndCapPos(310);
   m_calo->SetMaxTowerH( 150 );
   m_calo->SetScaleAbs( false );
}

void
FWRPZView::doDistortion()
{
   if ( typeId() == FWViewType::kRhoPhi ) {
      m_projMgr->GetProjection()->ChangePreScaleEntry(0,1,m_caloDistortion.value());
      m_projMgr->GetProjection()->ChangePreScaleEntry(0,2,m_muonDistortion.value());
   } else {
      m_projMgr->GetProjection()->ChangePreScaleEntry(0,1,m_caloDistortion.value());
      m_projMgr->GetProjection()->ChangePreScaleEntry(0,2,m_muonDistortion.value());
      m_projMgr->GetProjection()->ChangePreScaleEntry(1,1,m_caloDistortion.value());
      m_projMgr->GetProjection()->ChangePreScaleEntry(1,2,m_muonDistortion.value());
   }
   m_projMgr->UpdateName();
   m_projMgr->ProjectChildren();
   gEve->Redraw3D();
}

void
FWRPZView::doCompression(bool flag)
{
   m_projMgr->GetProjection()->SetUsePreScale(flag);
   m_projMgr->UpdateName();
   m_projMgr->ProjectChildren();
   gEve->Redraw3D();
}

void
FWRPZView::importElements(TEveElement* iChildren, float iLayer, TEveElement* iProjectedParent)
{
   float oldLayer = m_projMgr->GetCurrentDepth();
   m_projMgr->SetCurrentDepth(iLayer);
   //make sure current depth is reset even if an exception is thrown
   boost::shared_ptr<TEveProjectionManager> sentry(m_projMgr.get(),
                                                   boost::bind(&TEveProjectionManager::SetCurrentDepth,
                                                               _1,oldLayer));
   m_projMgr->ImportElements(iChildren,iProjectedParent);
}


void
FWRPZView::addTo(FWConfiguration& iTo) const
{
   FWEveView::addTo(iTo);
   TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>( &(viewerGL()->CurrentCamera()) );
   addToOrthoCamera(camera, iTo);
}

void
FWRPZView::setFrom(const FWConfiguration& iFrom)
{
   FWEveView::setFrom(iFrom);
   
   TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>( &(viewerGL()->CurrentCamera()) );
   setFromOrthoCamera(camera, iFrom);
}

void
FWRPZView::updateCaloParameters()
{
   double eta_range = 5.191;
   if ( m_showHF && !m_showHF->value() ) eta_range = 3.0;
   if ( m_showEndcaps && !m_showEndcaps->value() ) eta_range = 1.479;
   m_calo->SetEta(-eta_range,eta_range);

   m_calo->SetMaxValAbs( 150/m_caloFixedScale.value() );
   m_calo->SetScaleAbs( !m_caloAutoScale.value() );
   m_calo->ElementChanged();
   updateScaleParameters();
}

void
FWRPZView::updateScaleParameters()
{
   viewContext()->getEnergyScale("Calo")->setVal(m_calo->GetValToHeight());
   viewContext()->scaleChanged();
}

void FWRPZView::showProjectionAxes( )
{   
   if ( m_showProjectionAxes.value() )
      m_axes->SetRnrState(kTRUE);
   else
      m_axes->SetRnrState(kFALSE);
   gEve->Redraw3D();
}


void
FWRPZView::eventEnd()
{
   FWEveView::eventEnd();
   if (m_caloAutoScale.value())
   {
      updateScaleParameters();
   }
}

