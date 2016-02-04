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
// $Id: FW3DView.cc,v 1.54 2011/01/17 14:11:43 amraktad Exp $
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
   m_calo(0)
{
   viewerGL()->CurrentCamera().SetFixDefCenter(kTRUE);  
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

   m_calo->SetBarrelRadius(context().caloR1(false));
   m_calo->SetEndCapPos(context().caloZ1(false));
   m_calo->SetFrameTransparency(80);
   m_calo->SetAutoRange(false);
   m_calo->SetScaleAbs(true);
   eventScene()->AddElement(m_calo);
}
