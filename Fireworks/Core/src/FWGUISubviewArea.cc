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
// $Id: FWGUISubviewArea.cc,v 1.1 2008/02/15 20:33:03 chrjones Exp $
//

// system include files
#include "TSystem.h"
#include "TGButton.h"
#include "TGSplitFrame.h"
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
FWGUISubviewArea::FWGUISubviewArea(unsigned int iIndex, const TGWindow *iParent, TGSplitFrame* iMainSplit)
: TGHorizontalFrame(iParent),
  m_mainSplit(iMainSplit),
  m_index(iIndex)
{
   TGPictureButton* button= new TGPictureButton(this, swapIcon());
   button->SetToolTipText("Swap to big view");
   this->AddFrame(button);
   
   button->Connect("Clicked()","FWGUISubviewArea",this,"swapToBigView()");

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
   TGSplitFrame *dest = m_mainSplit->GetFirst();
   // get the pointer to the frame that has to be exchanged with the 
   // source one (the one actually in the destination)
   
   //NOTE: casting to TGCompositeFrame is WRONG but the problem is the interface requires
   // it although it really only needs a TGFrame!
   TGCompositeFrame *prev = (TGCompositeFrame *)(dest->GetFrame());
   assert(0!=prev);
   
   assert(0!=this->GetList()->Last());
   assert( dynamic_cast<TGFrameElement*>( this->GetList()->Last()));
   
   //NOTE: casting to TGCompositeFrame is WRONG but the problem is the interface requires
   // it although it really only needs a TGFrame!
   TGCompositeFrame* source = (TGCompositeFrame *)(dynamic_cast<TGFrameElement*>(this->GetList()->Last())->fFrame);
   assert(0!=source);
   TGSplitFrame::SwitchFrames( source, dest, prev);
   
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
