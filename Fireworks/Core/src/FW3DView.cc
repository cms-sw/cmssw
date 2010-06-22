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
// $Id: FW3DView.cc,v 1.41 2010/06/18 19:51:24 amraktad Exp $
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
   m_calo(0)
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

//
// member functions
//
void FW3DView::setContext(fireworks::Context& context)
{ 
   FW3DViewBase::setContext(context);
  
   TEveCaloData* data = context.getCaloData();

   m_calo = new TEveCalo3D(data);
   m_calo->SetMaxTowerH( 150 );
   m_calo->SetScaleAbs( false );
   m_calo->SetBarrelRadius(129);
   m_calo->SetEndCapPos(310);
   m_calo->SetFrameTransparency(80);

   eventScene()->AddElement(m_calo);
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
