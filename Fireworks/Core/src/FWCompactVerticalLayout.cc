// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCompactVerticalLayout
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Mar 17 12:10:42 CDT 2009
// $Id: FWCompactVerticalLayout.cc,v 1.3 2012/07/30 22:34:52 amraktad Exp $
//

// system include files
#include <algorithm>
#include <iostream>
#include "TGFrame.h"

// user include files
#include "Fireworks/Core/src/FWCompactVerticalLayout.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWCompactVerticalLayout::FWCompactVerticalLayout(TGCompositeFrame* iMain):
TGVerticalLayout(iMain)
{
}

// FWCompactVerticalLayout::FWCompactVerticalLayout(const FWCompactVerticalLayout& rhs)
// {
//    // do actual copying here;
// }

FWCompactVerticalLayout::~FWCompactVerticalLayout()
{
}

//
// assignment operators
//
// const FWCompactVerticalLayout& FWCompactVerticalLayout::operator=(const FWCompactVerticalLayout& rhs)
// {
//   //An exception safe implementation is
//   FWCompactVerticalLayout temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
//______________________________________________________________________________
void FWCompactVerticalLayout::Layout()
{
   // Make a vertical layout of all frames in the list.
   
   TGFrameElement *ptr;
   TGLayoutHints  *layout;
   Int_t    nb_expand = 0;
   Int_t    top, bottom;
   ULong_t  hints;
   UInt_t   extra_space = 0;
   Int_t    exp = 0;
   Int_t    exp_max = 0;
   Int_t    remain;
   Int_t    x = 0, y = 0;
   Int_t    bw = fMain->GetBorderWidth();
   TGDimension size(0,0), csize(0,0);
   TGDimension msize = fMain->GetSize();
   UInt_t pad_left, pad_top, pad_right, pad_bottom;
   Int_t size_expand=0, esize_expand=0, rem_expand=0, tmp_expand = 0;
   
   if (!fList) return;
   
   fModified = kFALSE;
   
   bottom = msize.fHeight - (top = bw);
   remain = msize.fHeight - (bw << 1);
   
   std::vector<int> expandSizes;
   expandSizes.reserve(fList->GetSize());
   TIter next(fList);
   while ((ptr = (TGFrameElement *) next())) {
      if (ptr->fState & kIsVisible) {
         layout = ptr->fLayout;
         size = ptr->fFrame->GetDefaultSize();
         size.fHeight += layout->GetPadTop() + layout->GetPadBottom();
         hints = layout->GetLayoutHints();
         if ((hints & kLHintsExpandY) || (hints & kLHintsCenterY)) {
            nb_expand++;
            exp += size.fHeight;
            if (hints & kLHintsExpandY) { 
               exp_max = 0;
               expandSizes.push_back(size.fHeight);
            }
            else exp_max = TMath::Max(exp_max, (Int_t)size.fHeight);
         } else {
            remain -= size.fHeight;
            if (remain < 0)
               remain = 0;
         }
      }
   }
   
   if (nb_expand) {
      size_expand = remain/nb_expand;
      
      if (size_expand < exp_max)
         esize_expand = (remain - exp)/nb_expand;
      rem_expand = remain % nb_expand;
   }
   
   std::sort(expandSizes.begin(), expandSizes.end(),std::less<int>());
   //Now see if expanded widgets exceed their max sizes
   for(std::vector<int>::iterator it = expandSizes.begin(), itEnd = expandSizes.end();
       it != itEnd;
       ++it) {
      if(*it > size_expand) {
         break;
      }
      remain -= *it;
      --nb_expand;
      if(remain<0) {remain=0;}
      if(nb_expand>0) {
         size_expand = remain/nb_expand;
      } else {
         size_expand=msize.fHeight - (bw << 1);
      }
   }
   
   next.Reset();
   while ((ptr = (TGFrameElement *) next())) {
      if (ptr->fState & kIsVisible) {
         hints = (layout = ptr->fLayout)->GetLayoutHints();
         csize      = ptr->fFrame->GetDefaultSize();
         pad_left   = layout->GetPadLeft();
         pad_top    = layout->GetPadTop();
         pad_right  = layout->GetPadRight();
         pad_bottom = layout->GetPadBottom();
         
         if (hints & kLHintsRight) {
            x = msize.fWidth - bw - csize.fWidth - pad_right;
         } else if (hints & kLHintsCenterX) {
            x = (msize.fWidth - (bw << 1) - csize.fWidth) >> 1;
         } else { // defaults to kLHintsLeft
            x = pad_left + bw;
         }
         
         if (hints & kLHintsExpandX) {
            size.fWidth = msize.fWidth - (bw << 1) - pad_left - pad_right;
            x = pad_left + bw;
         } else {
            size.fWidth = csize.fWidth;
         }
         
         if (hints & kLHintsExpandY) {
            if (size_expand >= exp_max)
               if(static_cast<int>(csize.fHeight) > size_expand) {
                  size.fHeight = size_expand - pad_top - pad_bottom;
               } else {
                  size.fHeight = csize.fHeight;
               }
            else
               size.fHeight = csize.fHeight + esize_expand;
            
            tmp_expand += rem_expand;
            if (tmp_expand >= nb_expand) {
               size.fHeight++;
               tmp_expand -= nb_expand;
            }
         } else {
            size.fHeight = csize.fHeight;
            if (hints & kLHintsCenterY) {
               if (size_expand >= exp_max) {
                  extra_space = (size_expand - pad_top - pad_bottom - size.fHeight) >> 1;
               } else {
                  extra_space = esize_expand >> 1;
               }
               top += extra_space;
            }
         }
         
         if (hints & kLHintsBottom) {
            y = bottom - size.fHeight - pad_bottom;
            bottom -= size.fHeight + pad_top + pad_bottom;
         } else { // kLHintsTop by default
            y = top + pad_top;
            top += size.fHeight + pad_top + pad_bottom;
         }
         
         if (hints & kLHintsCenterY)
            top += extra_space;
         
         if (size.fWidth > 32768)
            size.fWidth = 1;
         if (size.fHeight > 32768)
            size.fHeight = 1;
         ptr->fFrame->MoveResize(x, y, size.fWidth, size.fHeight);
         
         fModified = fModified || (ptr->fFrame->GetX() != x) || 
         (ptr->fFrame->GetY() != y) ||
         (ptr->fFrame->GetWidth() != size.fWidth) ||
         (ptr->fFrame->GetHeight() != size.fHeight);
      }
   }

}

//______________________________________________________________________________
TGDimension FWCompactVerticalLayout::GetDefaultSize() const
{
   // Return default dimension of the vertical layout.
   
   TGFrameElement *ptr;
   TGDimension     size(0,0), msize = fMain->GetSize(), csize;
   UInt_t options = fMain->GetOptions();
   
   if ((options & kFixedWidth) && (options & kFixedHeight))
      return msize;
   
   TIter next(fList);
   while ((ptr = (TGFrameElement *) next())) {
      if (ptr->fState & kIsVisible) {
         csize = ptr->fFrame->GetDefaultSize();
         size.fWidth = TMath::Max(size.fWidth, csize.fWidth + ptr->fLayout->GetPadLeft() +
                                  ptr->fLayout->GetPadRight());
         size.fHeight += csize.fHeight + ptr->fLayout->GetPadTop() +
         ptr->fLayout->GetPadBottom();
      }
   }
   
   size.fWidth  += fMain->GetBorderWidth() << 1;
   size.fHeight += fMain->GetBorderWidth() << 1;
   
   if (options & kFixedWidth)  size.fWidth  = msize.fWidth;
   if (options & kFixedHeight) size.fHeight = msize.fHeight;
   
   return size;
}

//
// const member functions
//

//
// static member functions
//
