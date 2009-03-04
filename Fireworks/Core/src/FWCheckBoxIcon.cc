// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCheckBoxIcon
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 19 16:25:17 CST 2009
// $Id$
//

// system include files
#include "TVirtualX.h"
#include "TGPicture.h"
#include "TGClient.h"
#include "TSystem.h"

// user include files
#include "Fireworks/Core/src/FWCheckBoxIcon.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
static
const TGPicture* checkImage()
{
   static TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));

   static const TGPicture* s_picture=gClient->GetPicture(coreIcondir+"check-mark-blackbg.png");
   return s_picture;
}

//
// constructors and destructor
//
FWCheckBoxIcon::FWCheckBoxIcon(unsigned int iEdgeLength):
FWBoxIconBase(iEdgeLength),
m_checked(false)
{
}

// FWCheckBoxIcon::FWCheckBoxIcon(const FWCheckBoxIcon& rhs)
// {
//    // do actual copying here;
// }

FWCheckBoxIcon::~FWCheckBoxIcon()
{
}

//
// assignment operators
//
// const FWCheckBoxIcon& FWCheckBoxIcon::operator=(const FWCheckBoxIcon& rhs)
// {
//   //An exception safe implementation is
//   FWCheckBoxIcon temp(rhs);
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
FWCheckBoxIcon::drawInsideBox(Drawable_t iID, GContext_t iContext, int iX, int iY, unsigned int iSize) const
{
   if(m_checked) {
      int xOffset = (iSize - checkImage()->GetWidth()) /2;
      int yOffset = (iSize - checkImage()->GetHeight())/2;
      checkImage()->Draw(iID,iContext,iX+xOffset,iY+yOffset);
   }
}

//
// static member functions
//
