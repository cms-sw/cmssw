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
// $Id: FWCheckBoxIcon.cc,v 1.5 2010/06/18 10:17:14 yana Exp $
//

// system include files
#include "TGPicture.h"
#include "TGClient.h"
#include "TSystem.h"
#include <cassert>

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
   static const TGPicture* s_picture=gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"check-mark.png");
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
const TString& FWCheckBoxIcon::coreIcondir() {
   static TString path = Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE"));
   if ( gSystem->AccessPathName(path.Data()) ){ // cannot find directory
      assert(gSystem->Getenv("CMSSW_RELEASE_BASE"));
      path = Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_RELEASE_BASE"));
   }

   return path;
}
