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
// $Id: FWHFView.cc,v 1.3 2010/06/07 17:54:01 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "TAxis.h"
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TEveScene.h"

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
   FWLegoViewBase(slot, typeId),
   m_drawValuesIn2D(this,"pixel font size in 2D)",40l,16l,200l)
{  
   m_drawValuesIn2D.changed_.connect(boost::bind(&FWHFView::setFontSizein2D,this));
 
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
}
   
void
FWHFView::setFontSizein2D()
{
   m_lego->SetDrawNumberCellPixels( m_drawValuesIn2D.value());
   m_lego->ElementChanged(kTRUE,kTRUE);
}
