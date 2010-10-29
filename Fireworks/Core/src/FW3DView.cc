// -*- C++ -*-
//
// Package:     cmsShow36
// Class  :     FW3DView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Wed Apr  7 14:40:47 CEST 2010
// $Id: FW3DView.cc,v 1.50 2010/10/01 09:45:20 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "TGLViewer.h"
#include "TGLScenePad.h"
#include "TEveCalo.h"
#include "TEveScene.h"

#include "Fireworks/Core/interface/FW3DView.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWViewContext.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DView::FW3DView(TEveWindowSlot* slot, FWViewType::EType typeId):
   FW3DViewBase(slot, typeId),
   m_calo(0),
   m_caloEndCap1(0),
   m_caloEndCap2(0)
{
   viewerGL()->CurrentCamera().SetFixDefCenter(kTRUE);  

   FWViewEnergyScale* caloScale = new FWViewEnergyScale(this);
   viewContext()->addScale("Calo", caloScale);
}

FW3DView::~FW3DView()
{
   m_calo->Destroy();
}


TEveCaloViz*
FW3DView::getEveCalo() const
{
   return static_cast<TEveCaloViz*>(m_calo);
}

void FW3DView::setContext(const fireworks::Context& ctx)
{ 
   FW3DViewBase::setContext(ctx);
   
   TEveCaloData* data = context().getCaloData();
   m_calo = new TEveCalo3D(data);
   m_calo->SetElementName("calo barrel");

   FWViewEnergyScale*  caloScale = viewContext()->getEnergyScale("Calo");
   m_calo->SetMaxTowerH(caloScale->getMaxTowerHeight());
   m_calo->SetScaleAbs(caloScale->getScaleMode() == FWViewEnergyScale::kFixedScale);
   m_calo->SetMaxValAbs(caloScale->getMaxFixedVal());

   m_calo->SetBarrelRadius(context().caloR1(false));
   m_calo->SetEndCapPos(context().caloZ1(false));
   m_calo->SetFrameTransparency(80);
   eventScene()->AddElement(m_calo);

   if (context().caloSplit())
   {
      float_t eps = 0.005;
      m_calo->SetEta(-context().caloTransEta() -eps, context().caloTransEta() + eps);
      m_calo->SetAutoRange(false);

      // back
      /*
      m_caloEndCap1 = new TEveCalo3D(data);
      m_caloEndCap1->SetElementName("endcap backwad");

      m_caloEndCap1->SetMaxTowerH(m_energyMaxTowerHeight.value());
      m_caloEndCap1->SetScaleAbs(m_energyScaleMode.value() == FWEveView::kFixedScale);
      m_caloEndCap1->SetMaxValAbs(m_energyMaxAbsVal.value());

      m_caloEndCap1->SetBarrelRadius(context().caloR2(false));
      m_caloEndCap1->SetEndCapPos(context().caloZ2(false));
      m_caloEndCap1->SetFrameTransparency(80);
      m_caloEndCap1->SetEta(-context().caloMaxEta(), -context().caloTransEta() + eps);
      m_caloEndCap1->SetAutoRange(false);
      eventScene()->AddElement(m_caloEndCap1);
   
      // front

      m_caloEndCap2 = new TEveCalo3D(data);
      m_caloEndCap2->SetElementName("endcap forward");

      m_caloEndCap2->SetMaxTowerH(m_energyMaxTowerHeight.value());
      m_caloEndCap2->SetScaleAbs(m_energyScaleMode.value() == FWEveView::kFixedScale);
      m_caloEndCap2->SetMaxValAbs(m_energyMaxAbsVal.value());

      m_caloEndCap2->SetBarrelRadius(context().caloR2(false));
      m_caloEndCap2->SetEndCapPos(context().caloZ2());
      m_caloEndCap2->SetFrameTransparency(80);
      m_caloEndCap2->SetEta(context().caloTransEta() -eps, context().caloMaxEta());
      m_caloEndCap2->SetAutoRange(false);
      eventScene()->AddElement(m_caloEndCap2);*/
   }

}
