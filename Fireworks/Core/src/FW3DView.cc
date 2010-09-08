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
// $Id: FW3DView.cc,v 1.45 2010/08/30 15:42:33 amraktad Exp $
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
   m_caloFixedScale(this,"Calo scale (GeV/meter)",2.,0.001,100.),
   m_caloAutoScale (this,"Calo auto scale",true),
   m_calo(0),
   m_caloEndCap1(0),
   m_caloEndCap2(0)
{
   viewerGL()->CurrentCamera().SetFixDefCenter(kTRUE);
   m_caloFixedScale.changed_.connect(boost::bind(&FW3DView::updateCaloParameters, this));
   m_caloAutoScale.changed_.connect(boost::bind(&FW3DView::updateCaloParameters, this));
}

// FW3DView::FW3DView(const FW3DView& rhs)
// {
//    // do actual copying here;
// }

FW3DView::~FW3DView()
{
}

//
// assignment operators
//
// const FW3DView& FW3DView::operator=(const FW3DView& rhs)
// {
//   //An exception safe implementation is
//   FW3DView temp(rhs);
//   swap(rhs);
//
//   return *this;
// }
float off= 1.02;
//
// member functions
//
void FW3DView::setContext(const fireworks::Context& ctx)
{ 
   FW3DViewBase::setContext(ctx);
  
   TEveCaloData* data = context().getCaloData();

   float_t eps = 0.005;
   m_calo = new TEveCalo3D(data);
   m_calo->SetElementName("calo barrel");
   m_calo->SetMaxTowerH( 150 );
   m_calo->SetScaleAbs( false );

   m_calo->SetBarrelRadius(context().caloR1(false));
   m_calo->SetEndCapPos(context().caloZ1(false));
   m_calo->SetFrameTransparency(80);
   eventScene()->AddElement(m_calo);

   if (context().caloSplit())
   {
      m_calo->SetEta(-context().caloTransEta() -eps, context().caloTransEta() + eps);
      m_calo->SetAutoRange(false);

      m_caloEndCap1 = new TEveCalo3D(data);
      m_caloEndCap1->SetElementName("endcap backwad");
      m_caloEndCap1->SetMaxTowerH( 150 );
      m_caloEndCap1->SetScaleAbs( false );
      m_caloEndCap1->SetBarrelRadius(context().caloR2(false));
      m_caloEndCap1->SetEndCapPos(context().caloZ2(false));
      m_caloEndCap1->SetFrameTransparency(80);
      m_caloEndCap1->SetEta(-context().caloMaxEta(), -context().caloTransEta() + eps);
      m_caloEndCap1->SetAutoRange(false);
      eventScene()->AddElement(m_caloEndCap1);
   
      m_caloEndCap2 = new TEveCalo3D(data);
      m_caloEndCap2->SetElementName("endcap forward");
      m_caloEndCap2->SetMaxTowerH( 150 );
      m_caloEndCap2->SetScaleAbs( false );
      m_caloEndCap2->SetBarrelRadius(context().caloR2(false));
      m_caloEndCap2->SetEndCapPos(context().caloZ2());
      m_caloEndCap2->SetFrameTransparency(80);
      m_caloEndCap2->SetEta(context().caloTransEta() -eps, context().caloMaxEta());
      m_caloEndCap2->SetAutoRange(false);
      eventScene()->AddElement(m_caloEndCap2);
   }
   viewContext()->getEnergyScale("Calo")->setVal(m_calo->GetValToHeight());
}

void FW3DView::updateCaloParameters()
{
   m_calo->SetMaxValAbs(150/m_caloFixedScale.value());
   m_calo->SetScaleAbs(!m_caloAutoScale.value());
   m_calo->ElementChanged();
   updateScaleParameters();
}

void FW3DView::updateScaleParameters()
{
   viewContext()->getEnergyScale("Calo")->setVal(m_calo->GetValToHeight());
   viewContext()->scaleChanged();
}


void
FW3DView::eventEnd()
{
   FWEveView::eventEnd();
   if (m_caloAutoScale.value())
   {
      updateScaleParameters();
   }
}

//
// const member functions
//

//
// static member functions
//
void 
FW3DView::populateController(ViewerParameterGUI& gui) const
{

   FW3DViewBase::populateController(gui);

   gui.requestTab("Scale").
      addParam(&m_caloFixedScale).
      addParam(&m_caloAutoScale);
}
