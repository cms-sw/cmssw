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
// $Id: FW3DView.cc,v 1.35 2010/04/12 12:43:10 amraktad Exp $
//

// system include files

// user include files
#include "TGLScenePad.h"
#include "TEveCalo.h"
#include "TEveScene.h"

#include "Fireworks/Core/interface/FW3DView.h"
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
FW3DView::FW3DView(TEveWindowSlot* w, TEveScene* s):
   FW3DViewBase(w, s),
   m_calo(0)
{
   setType(FWViewType::k3D);
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

   eventScene()->AddElement(m_calo);
}
//
// const member functions
//

//
// static member functions
//
