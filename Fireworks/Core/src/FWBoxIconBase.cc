// -*- C++ -*-
//
// Package:     Core
// Class  :     FWBoxIconBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 15:09:45 CST 2009
// $Id: FWBoxIconBase.cc,v 1.1 2009/03/04 16:40:50 chrjones Exp $
//

// system include files
#include "TVirtualX.h"

// user include files
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
FWBoxIconBase::FWBoxIconBase(unsigned int iEdgeLength):
m_edgeLength(iEdgeLength)
{
}

// FWBoxIconBase::FWBoxIconBase(const FWBoxIconBase& rhs)
// {
//    // do actual copying here;
// }

FWBoxIconBase::~FWBoxIconBase()
{
}

//
// assignment operators
//
// const FWBoxIconBase& FWBoxIconBase::operator=(const FWBoxIconBase& rhs)
// {
//   //An exception safe implementation is
//   FWBoxIconBase temp(rhs);
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
void 
FWBoxIconBase::draw(Drawable_t iID, GContext_t iContext, int iX, int iY) const
{
   //draw in background color
   gVirtualX->ClearArea(iID,iX,iY,m_edgeLength-1,m_edgeLength-1);
   //now draw foreground
   gVirtualX->DrawLine(iID, iContext, iX, iY, iX+m_edgeLength-1,iY);
   gVirtualX->DrawLine(iID, iContext, iX+m_edgeLength-1, iY, iX+m_edgeLength-1,iY+m_edgeLength-1);
   gVirtualX->DrawLine(iID, iContext, iX, iY+m_edgeLength-1, iX+m_edgeLength-1,iY+m_edgeLength-1);
   gVirtualX->DrawLine(iID, iContext, iX, iY, iX,iY+m_edgeLength-1);
   
   drawInsideBox(iID,iContext, iX+1, iY+1, m_edgeLength-2);
}

//
// static member functions
//
