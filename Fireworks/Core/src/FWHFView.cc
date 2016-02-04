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
// $Id: FWHFView.cc,v 1.12 2011/02/22 18:37:31 amraktad Exp $
//

// system include files

// user include files
#include "TEveCalo.h"
#include "TEveCaloData.h"
#include "TEveTrans.h"

#include "Fireworks/Core/interface/FWHFView.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWGenericParameter.h"

#include "Fireworks/Core/interface/FWEnumParameter.h"

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

//
// const member functions
//

//
// static member functions
//
   

void
FWHFView::setContext(const fireworks::Context& context)
{  
   FWLegoViewBase::setContext(context);

   // phi bins center shifted for 10 degres
   m_lego->RefMainTrans().SetPos(0, -TMath::DegToRad()*10, 0);
}



