// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCollectionSummaryModelCellRenderer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Wed Feb 25 10:03:23 CST 2009
// $Id: FWCollectionSummaryModelCellRenderer.cc,v 1.1 2009/03/04 16:40:50 chrjones Exp $
//

// system include files
#include "TVirtualX.h"
#include "TGClient.h"

// user include files
#include "Fireworks/Core/src/FWCollectionSummaryModelCellRenderer.h"

#include "Fireworks/Core/src/FWColorBoxIcon.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"

#include "Fireworks/Core/interface/FWEventItem.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
static const unsigned int kIconSize = 10;
static const unsigned int kSeparation = 2;
//
// constructors and destructor
//
FWCollectionSummaryModelCellRenderer::FWCollectionSummaryModelCellRenderer(const TGGC* iGC, const TGGC* iSelectContext):
FWTextTableCellRenderer(iGC,iSelectContext),
m_colorBox( new FWColorBoxIcon(kIconSize)),
m_checkBox( new FWCheckBoxIcon(kIconSize))
{
   GCValues_t t = *(iGC->GetAttributes());
   m_colorContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&t,kTRUE);
   m_colorBox->setColor(m_colorContext->GetGC());
}

// FWCollectionSummaryModelCellRenderer::FWCollectionSummaryModelCellRenderer(const FWCollectionSummaryModelCellRenderer& rhs)
// {
//    // do actual copying here;
// }

FWCollectionSummaryModelCellRenderer::~FWCollectionSummaryModelCellRenderer()
{
   delete m_colorBox;
   delete m_checkBox;
   gClient->GetResourcePool()->GetGCPool()->FreeGC(m_colorContext->GetGC());
}

//
// assignment operators
//
// const FWCollectionSummaryModelCellRenderer& FWCollectionSummaryModelCellRenderer::operator=(const FWCollectionSummaryModelCellRenderer& rhs)
// {
//   //An exception safe implementation is
//   FWCollectionSummaryModelCellRenderer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
UInt_t 
FWCollectionSummaryModelCellRenderer::width() const
{
   UInt_t w = FWTextTableCellRenderer::width();
   return w+kIconSize+kIconSize+kSeparation+kSeparation;
}

void 
FWCollectionSummaryModelCellRenderer::draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
{
   int dY = (iHeight-kIconSize)/2;
   m_checkBox->draw(iID,graphicsContext()->GetGC(),iX,iY+dY);
   m_colorBox->draw(iID,graphicsContext()->GetGC(),iX+kIconSize+kSeparation,iY+dY);
   FWTextTableCellRenderer::draw(iID, iX+kIconSize+kIconSize+kSeparation+kSeparation, iY, 
                                 iWidth-kIconSize-kIconSize-kSeparation-kSeparation, iHeight);
   return ;
}

void 
FWCollectionSummaryModelCellRenderer::setData(const FWEventItem* iItem, int iIndex)
{
   FWEventItem::ModelInfo mi = iItem->modelInfo(iIndex);
   FWTextTableCellRenderer::setData(iItem->modelName(iIndex),mi.isSelected());
   m_checkBox->setChecked(mi.displayProperties().isVisible());
   m_colorContext->SetForeground(gVirtualX->GetPixel(mi.displayProperties().color()));
}

//
// const member functions
//
FWCollectionSummaryModelCellRenderer::ClickHit 
FWCollectionSummaryModelCellRenderer::clickHit(int iX, int iY) const
{
   if(iY < 0 || iY > static_cast<int>(kIconSize)) { return kMiss;}
   if(iX>=0 && iX<=static_cast<int>(kIconSize)) {return kHitCheck;}
   if(iX>=static_cast<int>(kIconSize+kSeparation) && iX <=static_cast<int>(kIconSize+kSeparation+kIconSize)) { return kHitColor;}
   return kMiss;
}

//
// static member functions
//
