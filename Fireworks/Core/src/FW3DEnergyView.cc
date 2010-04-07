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
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FW3DEnergyView.h"

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
   FW3DViewBase(w, s)
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

//
// const member functions
//

//
// static member functions
//
