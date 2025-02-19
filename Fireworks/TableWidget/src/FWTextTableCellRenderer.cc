// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWTextTableCellRenderer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:43:54 EST 2009
// $Id: FWTextTableCellRenderer.cc,v 1.5 2011/03/02 18:34:17 amraktad Exp $
//

// system include files
#include <iostream>
#include "TGClient.h"
#include "TGFont.h"
#include "TVirtualX.h"

// user include files
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/TableWidget/src/FWTabularWidget.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTextTableCellRenderer::FWTextTableCellRenderer(const TGGC* iContext, const TGGC* iHighlightContext, Justify iJustify):
   m_context(iContext),
   m_highlightContext(iHighlightContext),
   m_font(0),
   m_isSelected(false),
   m_justify(iJustify)
{
   //TGGC* tggc= gClient->GetGCPool()->GetGC(iContext);
   m_font = gClient->GetFontPool()->FindFontByHandle(m_context->GetFont());
}

// FWTextTableCellRenderer::FWTextTableCellRenderer(const FWTextTableCellRenderer& rhs)
// {
//    // do actual copying here;
// }

FWTextTableCellRenderer::~FWTextTableCellRenderer()
{
}

//
// assignment operators
//
// const FWTextTableCellRenderer& FWTextTableCellRenderer::operator=(const FWTextTableCellRenderer& rhs)
// {
//   //An exception safe implementation is
//   FWTextTableCellRenderer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWTextTableCellRenderer::draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
{
 
   if(m_isSelected) {
      GContext_t c = m_highlightContext->GetGC();
      gVirtualX->FillRectangle(iID, c, iX - FWTabularWidget::kTextBuffer, iY - FWTabularWidget::kTextBuffer,
                               iWidth + 2*FWTabularWidget::kTextBuffer, iHeight + 2*FWTabularWidget::kTextBuffer);
      /*
        gVirtualX->DrawLine(iID,m_context->GetGC(),iX-1,iY-1,iX-1,iY+iHeight);
        gVirtualX->DrawLine(iID,m_context->GetGC(),iX+iWidth,iY-1,iX+iWidth,iY+iHeight);
        gVirtualX->DrawLine(iID,m_context->GetGC(),iX-1,iY-1,iX+iWidth,iY-1);
        gVirtualX->DrawLine(iID,m_context->GetGC(),iX-1,iY+iHeight,iX+iWidth,iY+iHeight);*
      */
   }



   FontMetrics_t metrics;
   m_font->GetFontMetrics(&metrics);
   int dX=0;
   if(m_justify==kJustifyRight) {
      int w = width();
      dX = iWidth-w;
   }
   if(m_justify==kJustifyCenter) {
      int w = width();
      dX = (iWidth-w)/2;

   }

   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX,iY+metrics.fAscent +1, m_data.c_str(),m_data.size());
}

void
FWTextTableCellRenderer::setData(const std::string& iData, bool iIsSelected) {
   m_data = iData;
   m_isSelected=iIsSelected;
}

void
FWTextTableCellRenderer::setData(const char* iData, bool iIsSelected) {
   m_data = iData;
   m_isSelected=iIsSelected;
}

void 
FWTextTableCellRenderer::setJustify(Justify iJustify)
{
   m_justify=iJustify;
}

//
// const member functions
//
UInt_t 
FWTextTableCellRenderer::width() const
{
   if(m_data.size()) {
      return m_font->TextWidth(m_data.c_str(),-1);// + 2*kTextBuffer;
   }
   return 0;
}
UInt_t 
FWTextTableCellRenderer::height() const
{
   return m_font->TextHeight(); //+  2*kTextBuffer;
}

const TGFont* 
FWTextTableCellRenderer::font() const
{
   return m_font;
}

//
// static member functions
//
const TGGC&  
FWTextTableCellRenderer::getDefaultGC()
{
   static const TGGC* s_default = gClient->GetResourcePool()->GetFrameGC();
   return *s_default;
}

const TGGC &
FWTextTableCellRenderer::getDefaultHighlightGC()
{
   // Return graphics context for highlighted frame background.
   static const TGGC* s_default = 0;
   if (!s_default) {
      GCValues_t gval;
      gval.fMask = kGCForeground | kGCBackground | kGCStipple | kGCFillStyle  | kGCGraphicsExposures;
      gval.fForeground = gVirtualX->GetPixel(kGray);//gClient->GetResourcePool()->GetFrameHiliteColor();
      gval.fBackground = gVirtualX->GetPixel(kWhite);//gClient->GetResourcePool()->GetFrameBgndColor();
      gval.fFillStyle  = kFillOpaqueStippled; // kFillTiled;
      gval.fStipple    = gClient->GetResourcePool()->GetCheckeredBitmap();
      gval.fGraphicsExposures = kFALSE;
      s_default = gClient->GetGC(&gval, kTRUE);
   }
   return *s_default;
}
