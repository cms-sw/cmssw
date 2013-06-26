// -*- C++ -*-
//
// Package:     Core
// Class  :     FWColorBoxIcon
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 15:42:05 CST 2009
// $Id: FWColorBoxIcon.cc,v 1.1 2009/03/04 16:40:51 chrjones Exp $
//

// system include files
#include "TVirtualX.h"

// user include files
#include "Fireworks/Core/src/FWColorBoxIcon.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWColorBoxIcon::FWColorBoxIcon(unsigned int iEdgeLength) :
FWBoxIconBase(iEdgeLength),
m_colorContext(0)
{
}

// FWColorBoxIcon::FWColorBoxIcon(const FWColorBoxIcon& rhs)
// {
//    // do actual copying here;
// }

//FWColorBoxIcon::~FWColorBoxIcon()
//{
//}

//
// assignment operators
//
// const FWColorBoxIcon& FWColorBoxIcon::operator=(const FWColorBoxIcon& rhs)
// {
//   //An exception safe implementation is
//   FWColorBoxIcon temp(rhs);
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
FWColorBoxIcon::drawInsideBox(Drawable_t iID, GContext_t iContext, int iX, int iY, unsigned int iSize) const
{
   gVirtualX->FillRectangle(iID, m_colorContext, iX+1, iY+1, iSize-2, iSize-2);
}

//
// static member functions
//
