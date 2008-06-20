// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGUISubviewArea
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Feb 15 14:13:33 EST 2008
// $Id: FWGUISubviewArea.cc,v 1.3 2008/05/18 09:42:48 jmuelmen Exp $
//

// system include files
#include "TSystem.h"
#include "TGButton.h"
#include "TGSplitFrame.h"
#include <assert.h>
#include <stdexcept>

// user include files
#include "Fireworks/Core/interface/FWGUISubviewArea.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWGUISubviewArea::FWGUISubviewArea(unsigned int iIndex, const TGSplitFrame *iParent, TGSplitFrame* iMainSplit)
: TGHorizontalFrame(iParent),
  m_mainSplit(iMainSplit),
  m_index(iIndex)
{
   m_swapButton= new TGPictureButton(this, swapIcon());
   m_swapButton->SetToolTipText("Swap to big view");
   this->AddFrame(m_swapButton);
   
   m_swapButton->Connect("Clicked()","FWGUISubviewArea",this,"swapToBigView()");

   //behavior of buttons depends on index
   if(0==iIndex) {
      m_swapButton->SetEnabled(kFALSE);
   }
}

// FWGUISubviewArea::FWGUISubviewArea(const FWGUISubviewArea& rhs)
// {
//    // do actual copying here;
// }

FWGUISubviewArea::~FWGUISubviewArea()
{
}

//
// assignment operators
//
// const FWGUISubviewArea& FWGUISubviewArea::operator=(const FWGUISubviewArea& rhs)
// {
//   //An exception safe implementation is
//   FWGUISubviewArea temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWGUISubviewArea::swapToBigView()
{
   //We know the parent is a TGSplitFrame because the constructor requires it to be so
   TGSplitFrame* p = const_cast<TGSplitFrame*>(static_cast<const TGSplitFrame*>(GetParent()));
   p->SwitchToMain();
   
   swappedToBigView_(index());
}

//
// const member functions
//

//
// static member functions
//
 const TGPicture * 
FWGUISubviewArea::swapIcon()
{
   static const TGPicture* s_icon = 0;
   if(0== s_icon) {
      const char* cmspath = gSystem->Getenv("CMSSW_BASE");
      if(0 == cmspath) {
         throw std::runtime_error("CMSSW_BASE environment variable not set");
      }
      TString coreIcondir(Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE")));
      s_icon = gClient->GetPicture(coreIcondir+"swap.png");
   }
   return s_icon;
}

void 
FWGUISubviewArea::setIndex(unsigned int iIndex) {
   if(0==iIndex) {
      m_swapButton->SetEnabled(kFALSE);
   }
   if(m_index==0) {
      m_swapButton->SetEnabled(kTRUE);
   }
   m_index = iIndex;
}
