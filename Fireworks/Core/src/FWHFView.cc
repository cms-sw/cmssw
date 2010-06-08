// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHFView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Mon May 31 13:42:13 CEST 2010
// $Id: FWHFView.cc,v 1.4 2010/06/08 11:35:00 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "TAxis.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TEveScene.h"
#include "TEveCaloLegoOverlay.h"

#include "Fireworks/Core/interface/FWHFView.h"
#include "Fireworks/Core/interface/FWEveLegoView.h"
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
FWHFView::FWHFView(TEveWindowSlot* slot, FWViewType::EType typeId):
   FWLegoViewBase(slot, typeId)
{  
}

// FWHFView::FWHFView(const FWHFView& rhs)
// {
//    // do actual copying here;
// }

FWHFView::~FWHFView()
{
}

//
// assignment operators
//
// const FWHFView& FWHFView::operator=(const FWHFView& rhs)
// {
//   //An exception safe implementation is
//   FWHFView temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

TEveCaloData*
FWHFView::getCaloData(fireworks::Context& c) const
{
   return c.getCaloDataHF();
}
//
// const member functions
//

//
// static member functions
//
   

void
FWHFView::setContext(fireworks::Context& context)
{  
   FWLegoViewBase::setContext(context);
   m_lego->Set2DMode(TEveCaloLego::kValSizeOutline);

   // temporary disable  camera overlay
   // because of problems with auto -resize in slice filters

   // m_overlay->SetShowOrthographic(false);

}
