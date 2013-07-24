// -*- C++ -*-
//
// Package:     Core
// Class  :     FWBoxIconButton
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 19:04:10 CST 2009
// $Id: FWBoxIconButton.cc,v 1.3 2010/06/18 10:17:14 yana Exp $
//

// system include files

// user include files
#include "Fireworks/Core/src/FWBoxIconButton.h"
#include "Fireworks/Core/src/FWBoxIconBase.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWBoxIconButton::FWBoxIconButton(const TGWindow* iParent,
                                 FWBoxIconBase* iBase,
                                 Int_t iID,
                                 GContext_t norm ,
                                 UInt_t option):
TGButton(iParent,iID,norm,option),
m_iconBase(iBase)
{
   Resize(m_iconBase->edgeLength(),m_iconBase->edgeLength());
}

// FWBoxIconButton::FWBoxIconButton(const FWBoxIconButton& rhs)
// {
//    // do actual copying here;
// }

FWBoxIconButton::~FWBoxIconButton()
{
   delete m_iconBase;
}

//
// assignment operators
//
// const FWBoxIconButton& FWBoxIconButton::operator=(const FWBoxIconButton& rhs)
// {
//   //An exception safe implementation is
//   FWBoxIconButton temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWBoxIconButton::DoRedraw()
{
   m_iconBase->draw(fId,fNormGC,0,0);
}

void FWBoxIconButton::setNormCG(GContext_t iContext)
{
   fNormGC = iContext;
}

//
// const member functions
//

//
// static member functions
//
