// -*- C++ -*-
//
// Package:     cmsShow36
// Class  :     FW3DRecHitView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel 
//         Created:  Wed Apr  7 14:40:31 CEST 2010
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DRecHitView.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FW3DRecHitView::FW3DRecHitView(TEveWindowSlot* w, TEveScene* s):
   FW3DViewBase(w, s)
{
   setType(FWViewType::k3DRecHit);
}

// FW3DRecHitView::FW3DRecHitView(const FW3DRecHitView& rhs)
// {
//    // do actual copying here;
// }

FW3DRecHitView::~FW3DRecHitView()
{
}

//
// assignment operators
//
// const FW3DRecHitView& FW3DRecHitView::operator=(const FW3DRecHitView& rhs)
// {
//   //An exception safe implementation is
//   FW3DRecHitView temp(rhs);
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
