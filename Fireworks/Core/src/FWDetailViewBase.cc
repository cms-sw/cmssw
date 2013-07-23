// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jan  9 13:35:56 EST 2009
// $Id: FWDetailViewBase.cc,v 1.23 2010/06/18 10:17:15 yana Exp $
//

// system include files
#include "TBox.h"
#include "TEllipse.h"
#include "TEveViewer.h"

// user include files
#include "Fireworks/Core/interface/FWDetailViewBase.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWEventItem.h"

FWDetailViewBase::FWDetailViewBase(const std::type_info& iInfo) :
   m_item(0),
   m_helper(iInfo)
{
}

FWDetailViewBase::~FWDetailViewBase()
{
}


void
FWDetailViewBase::build (const FWModelId &iID)
{
   m_helper.itemChanged(iID.item());
   build(iID, m_helper.offsetObject(iID.item()->modelData(iID.index())));
}

const fireworks::Context&
FWDetailViewBase::context () const {
   return m_item->context();
}

//______________________________________________________________________________
// UTILITIES for Canvas info
void
FWDetailViewBase::drawCanvasDot(Float_t x, Float_t y, Float_t r, Color_t fillColor)
{ 
   // utility function to draw outline cricle

   Float_t ratio = 0.5;
   // fill
   TEllipse *b2 = new TEllipse(x, y, r, r*ratio);
   b2->SetFillStyle(1001);
   b2->SetFillColor(fillColor);
   b2->Draw();

   // outline
   TEllipse *b1 = new TEllipse(x, y, r, r*ratio);
   b1->SetFillStyle(0);
   b1->SetLineWidth(2);
   b1->Draw();
}

void
FWDetailViewBase::drawCanvasBox( Double_t *pos, Color_t fillCol, Int_t fillType, bool bg)
{ 
   // utility function to draw outline box

   // background
   if (bg)
   {
      TBox *b1 = new TBox(pos[0], pos[1], pos[2], pos[3]);
      b1->SetFillColor(fillCol);
      b1->Draw();
   }

   // fill  (top layer)
   TBox *b2 = new TBox(pos[0], pos[1], pos[2], pos[3]);
   b2->SetFillStyle(fillType);
   b2->SetFillColor(kBlack);
   b2->Draw();

   //outline
   TBox *b3 = new TBox(pos[0], pos[1], pos[2], pos[3]);
   b3->SetFillStyle(0);
   b3->SetLineWidth(2);
   b3->Draw();
}


