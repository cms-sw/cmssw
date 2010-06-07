// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveLegoView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Mon May 31 13:09:53 CEST 2010
// $Id: FWEveLegoView.cc,v 1.81 2010/05/31 13:01:25 amraktad Exp $
//

// system include files

// user include files
#include "TEveCalo.h"
#include "TEveStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveLegoView.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWColorManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEveLegoView::FWEveLegoView(TEveWindowSlot* slot, FWViewType::EType typeId):
FWLegoViewBase(slot, typeId)
{
}

// FWEveLegoView::FWEveLegoView(const FWEveLegoView& rhs)
// {
//    // do actual copying here;
// }

FWEveLegoView::~FWEveLegoView()
{
}

//
// assignment operators
//
// const FWEveLegoView& FWEveLegoView::operator=(const FWEveLegoView& rhs)
// {
//   //An exception safe implementation is
//   FWEveLegoView temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

//
// static member functions
//

TEveCaloData*
FWEveLegoView::getCaloData(fireworks::Context& c) const
{
   return (TEveCaloData*)c.getCaloData();
}

void
FWEveLegoView::setContext(fireworks::Context& context)
{  
   FWLegoViewBase::setContext(context); 

   // add calorimeter boundaries
   TEveStraightLineSet* boundaries = new TEveStraightLineSet("boundaries");
   boundaries->SetPickable(kFALSE);
   boundaries->SetLineWidth(2);
   boundaries->SetLineStyle(7);
   boundaries->AddLine(-1.479,-3.1416,0.001,-1.479,3.1416,0.001);
   boundaries->AddLine(1.479,-3.1416,0.001,1.479,3.1416,0.001);
   boundaries->AddLine(-2.964,-3.1416,0.001,-2.964,3.1416,0.001);
   boundaries->AddLine(2.964,-3.1416,0.001,2.964,3.1416,0.001);
   boundaries->SetLineColor(context.colorManager()->geomColor(kFWLegoBoundraryColorIndex));
   m_lego->AddElement(boundaries);
}
