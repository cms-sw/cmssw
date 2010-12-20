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
// $Id: FWRPZView.cc,v 1.27 2010/10/18 17:32:25 amraktad Exp $
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
#include "Fireworks/Core/interface/CmsShowViewPopup.h"

FWRPZViewGeometry* FWRPZView::s_geometryList = 0;

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
   m_showHF(0),
   m_showEndcaps(0)
{
   FWViewEnergyScale* caloScale = new FWViewEnergyScale(this);
   viewContext()->addScale("Calo", caloScale);

   TEveProjection::EPType_e projType = (id == FWViewType::kRhoZ) ? TEveProjection::kPT_RhoZ : TEveProjection::kPT_RPhi;

   m_projMgr = new TEveProjectionManager(projType);
   m_projMgr->IncDenyDestroy();
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

   m_axes = new TEveProjectionAxes(m_projMgr);
   m_axes->SetRnrState(m_showProjectionAxes.value());
   m_showProjectionAxes.changed_.connect(boost::bind(&FWRPZView::showProjectionAxes,this));
   eventScene()->AddElement(m_axes);

   if ( id == FWViewType::kRhoPhi ) {
      m_showEndcaps = new FWBoolParameter(this,"Include EndCaps", true);
      m_showEndcaps->changed_.connect(  boost::bind(&FWRPZView::setEtaRng, this) );
      m_showHF = new FWBoolParameter(this,"Include HF", true);
      m_showHF->changed_.connect(  boost::bind(&FWRPZView::setEtaRng, this) );
   }

   m_caloDistortion.changed_.connect(boost::bind(&FWRPZView::doDistortion,this));
   m_muonDistortion.changed_.connect(boost::bind(&FWRPZView::doDistortion,this));
   m_compressMuon.changed_.connect(boost::bind(&FWRPZView::doCompression,this,_1));
}

FWRPZView::~FWRPZView()
{
   m_calo->Destroy();
   m_projMgr->DecDenyDestroy();
}

//
// member functions
//

TEveCaloViz*
FWRPZView::getEveCalo() const
{
   return static_cast<TEveCaloViz*>(m_calo);
}

void
FWRPZView::setContext(const fireworks::Context& ctx)
{
   FWEveView::setContext(ctx);

   if (!s_geometryList)
   {
      s_geometryList = new  FWRPZViewGeometry(ctx);
      s_geometryList->IncDenyDestroy();
   }
   m_projMgr->ImportElements(s_geometryList->getGeoElements(typeId()), geoScene());

   TEveCaloData* data = context().getCaloData();

   TEveCalo3D* calo3d = new TEveCalo3D(data);

   m_calo = static_cast<TEveCalo2D*> (m_projMgr->ImportElements(calo3d, eventScene()));
   m_calo->SetBarrelRadius(context().caloR1(false));
   m_calo->SetEndCapPos(context().caloZ1(false));
   m_calo->SetAutoRange(false);

   FWViewEnergyScale*  caloScale = viewContext()->getEnergyScale("Calo");
   m_calo->SetMaxTowerH(caloScale->getMaxTowerHeight());
   m_calo->SetScaleAbs(caloScale->getScaleMode() == FWViewEnergyScale::kFixedScale);
   m_calo->SetMaxValAbs(caloScale->getMaxFixedVal());
   
   /*
   if (typeId() == FWViewType::kRhoZ && context().caloSplit())
   {
      float_t eps = 0.005;
      m_calo->SetAutoRange(false);
      m_calo->SetEta(-context().caloTransEta() -eps, context().caloTransEta() + eps);

      m_caloEndCap1 = static_cast<TEveCalo2D*> (m_projMgr->ImportElements(calo3d, eventScene()));
      m_caloEndCap1->SetBarrelRadius(context().caloR2(false));
      m_caloEndCap1->SetEndCapPos(context().caloZ2(false));
      m_caloEndCap1->SetEta(-context().caloMaxEta(), -context().caloTransEta() + eps);
      m_caloEndCap1->SetMaxTowerH(m_energyMaxTowerHeight.value());
      m_caloEndCap1->SetAutoRange(m_energyScaleMode.value() == FWEveView::kFixedScale);
      m_caloEndCap1->SetMaxValAbs(m_energyMaxAbsVal.value());

      m_caloEndCap2 = static_cast<TEveCalo2D*> (m_projMgr->ImportElements(calo3d, eventScene()));
      m_caloEndCap2->SetBarrelRadius(context().caloR2(false));
      m_caloEndCap2->SetEndCapPos(context().caloZ2(false));
      m_caloEndCap2->SetEta(context().caloTransEta() -eps, context().caloMaxEta());
      m_caloEndCap2->SetMaxTowerH(m_energyMaxTowerHeight.value());
      m_caloEndCap2->SetAutoRange(m_energyScaleMode.value() == FWEveView::kFixedScale);
      m_caloEndCap2->SetMaxValAbs(m_energyMaxAbsVal.value());
   }
   */

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
   boost::shared_ptr<TEveProjectionManager> sentry(m_projMgr,
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
FWRPZView::setEtaRng()
{
   if (typeId() == FWViewType::kRhoPhi)
   {
      // rng controllers only in RhoPhi
      double eta_range = context().caloMaxEta();
      if (m_showHF->value() ) eta_range = 3.0;
      if (!m_showEndcaps->value() ) eta_range = context().caloTransEta();
      m_calo->SetEta(-eta_range,eta_range);
   }

   FWEveView::updateEnergyScales();
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
FWRPZView::populateController(ViewerParameterGUI& gui) const
{
   FWEveView::populateController(gui);

   gui.requestTab("Projection").
      addParam(&m_compressMuon).
      addParam(&m_muonDistortion).
      addParam(&m_caloDistortion).
      addParam(&m_showProjectionAxes);

   if (typeId() == FWViewType::kRhoPhi) 
   {
      gui.requestTab("Calo").
         addParam(m_showHF).
         addParam(m_showEndcaps);
   }
}

