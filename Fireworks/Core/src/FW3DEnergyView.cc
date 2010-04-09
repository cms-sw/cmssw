// -*- C++ -*-
//
// Package:     cmsShow36
// Class  :     FW3DEnergyView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Wed Apr  7 14:40:47 CEST 2010
// $Id: FW3DEnergyView.cc,v 1.1 2010/04/07 16:56:20 amraktad Exp $
//

// system include files

// user include files
#include "TEveCalo.h"
#include "TEveScene.h"
// #include "TEveManager.h"

#include "Fireworks/Core/interface/FW3DEnergyView.h"
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
FW3DEnergyView::FW3DEnergyView(TEveWindowSlot* w, TEveScene* s):
   FW3DViewBase(w, s),
   m_calo(0)
{
   setType(FWViewType::k3DE);
}

// FW3DEnergyView::FW3DEnergyView(const FW3DEnergyView& rhs)
// {
//    // do actual copying here;
// }

FW3DEnergyView::~FW3DEnergyView()
{
}

//
// assignment operators
//
// const FW3DEnergyView& FW3DEnergyView::operator=(const FW3DEnergyView& rhs)
// {
//   //An exception safe implementation is
//   FW3DEnergyView temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void FW3DEnergyView::setGeometry(fireworks::Context& context)
{ 
   TEveCaloData* data = context.getCaloData();
   for (TEveElement::List_i i = data->BeginChildren(); i!= data->EndChildren(); ++i)
   {
      if( dynamic_cast<TEveCalo3D*>(*i))
      {
         m_calo = dynamic_cast<TEveCalo3D*>(*i);
         break;
      }
   }
   // create if not exist
   if (m_calo == 0)
   {
      TEveCaloData* data = context.getCaloData();
      m_calo = new TEveCalo3D(data);
      m_calo->SetMaxTowerH( 150 );
      m_calo->SetScaleAbs( false );
      m_calo->SetBarrelRadius(129);
      m_calo->SetEndCapPos(310);
      m_calo->SetFrameTransparency(80);
   }

   geoScene()->AddElement(m_calo);
}
//
// const member functions
//

//
// static member functions
//
