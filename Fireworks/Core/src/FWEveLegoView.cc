// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveLegoView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Mon May 31 13:09:53 CEST 2010
// $Id: FWEveLegoView.cc,v 1.85 2010/09/26 19:57:21 amraktad Exp $
//

// system include files
#include "TEveManager.h"

// user include files
#include "TEveCalo.h"
#include "TEveStraightLineSet.h"
#include "Fireworks/Core/interface/FWEveLegoView.h"
#include "Fireworks/Core/interface/Context.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/FWViewContext.h"

#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/fwLog.h"




//
// constructors and destructor
//
FWEveLegoView::FWEveLegoView(TEveWindowSlot* slot, FWViewType::EType typeId):
FWLegoViewBase(slot, typeId)
{
}


FWEveLegoView::~FWEveLegoView()
{
}

void
FWEveLegoView::setContext(const fireworks::Context& ctx)
{  
   FWLegoViewBase::setContext(ctx); 

   // add calorimeter boundaries
   m_boundaries = new TEveStraightLineSet("m_boundaries");
   m_boundaries->SetPickable(kFALSE);
   m_boundaries->SetLineWidth(2);
   m_boundaries->SetLineStyle(7);
   m_boundaries->AddLine(-1.479,-3.1416,0.001,-1.479,3.1416,0.001);
   m_boundaries->AddLine(1.479,-3.1416,0.001,1.479,3.1416,0.001);
   m_boundaries->AddLine(-2.964,-3.1416,0.001,-2.964,3.1416,0.001);
   m_boundaries->AddLine(2.964,-3.1416,0.001,2.964,3.1416,0.001);
   m_boundaries->SetLineColor(ctx.colorManager()->isColorSetDark() ? kGray+2 : kGray+1);
   m_lego->AddElement(m_boundaries);
}

void 
FWEveLegoView::setBackgroundColor(Color_t c)
{
   m_boundaries->SetLineColor(context().colorManager()->isColorSetDark() ? kGray+2 : kGray+1);
   FWEveView::setBackgroundColor(c);
}


void
FWEveLegoView::energyScalesChanged()
{
   // Virtual method of FWEveView::energyScalesChanged. Overriden here to 
   // apply context's calo data scale to  kLegoPFECAL view type.

   FWViewEnergyScale* caloScale = viewContext()->getEnergyScale("Calo");
   TEveCaloData* data = context().getCaloData();
   if (data->Empty() && caloScale->getScaleMode() == FWViewEnergyScale::kAutoScale)
   {
      fwLog(fwlog::kError) <<"Error in automatic scale mode:Tower collection empty. Please switch to fixed-scale mode. \n";
      return;
   }
   float maxVal = data->GetMaxVal(caloScale->getPlotEt());
   caloScale->setMaxVal(maxVal);
   float valToHeight = caloScale->getMaxTowerHeight()/maxVal;
   valToHeight *= 0.01*TMath::Pi(); // this call will be obsolete when scene apply transformation matrix to child nodes
   caloScale->setValToHeight(valToHeight);
   viewContext()->scaleChanged();
   getEveCalo()->ElementChanged();
   gEve->Redraw3D();
}
