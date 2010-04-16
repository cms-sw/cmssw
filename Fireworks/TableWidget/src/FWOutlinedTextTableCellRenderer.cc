// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWOutlinedTextTableCellRenderer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:43:54 EST 2009
// $Id: FWOutlinedTextTableCellRenderer.cc,v 1.2 2009/03/04 15:30:02 chrjones Exp $
//

// system include files
#include <iostream>
#include "TGClient.h"
#include "TGResourcePool.h"
#include "TGFont.h"
#include "TVirtualX.h"

// user include files
#include "Fireworks/TableWidget/interface/FWOutlinedTextTableCellRenderer.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWOutlinedTextTableCellRenderer::FWOutlinedTextTableCellRenderer(const TGGC* iTextContext, const TGGC* iFillContext, Justify iJustify):
   m_context(iTextContext),
   m_fillContext(iFillContext),
   m_font(0),
   m_justify(iJustify)
{
   //TGGC* tggc= gClient->GetGCPool()->GetGC(iContext);
   m_font = gClient->GetFontPool()->FindFontByHandle(m_context->GetFont());
}

// FWOutlinedTextTableCellRenderer::FWOutlinedTextTableCellRenderer(const FWOutlinedTextTableCellRenderer& rhs)
// {
//    // do actual copying here;
// }

FWOutlinedTextTableCellRenderer::~FWOutlinedTextTableCellRenderer()
{
}

//
// assignment operators
//
// const FWOutlinedTextTableCellRenderer& FWOutlinedTextTableCellRenderer::operator=(const FWOutlinedTextTableCellRenderer& rhs)
// {
//   //An exception safe implementation is
//   FWOutlinedTextTableCellRenderer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWOutlinedTextTableCellRenderer::draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
{
   GContext_t c = m_fillContext->GetGC();
   gVirtualX->FillRectangle(iID, c, iX, iY, iWidth, iHeight);
      
   FontMetrics_t metrics;
   m_font->GetFontMetrics(&metrics);
   int dX=2;
   if(m_justify==kJustifyRight) {
      int w = width();
      dX = iWidth-w;
   }
   if(m_justify==kJustifyCenter) {
      int w = width();
      dX = (iWidth-w)/2;

   }
   Pixel_t oldForeground = m_context->GetForeground();
   //NOTE: this is only done temporarily and then reset
   const_cast<TGGC*>(m_context)->SetForeground(m_context->GetBackground());
   //draw outline by redrawing text 8 times
   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX-1,iY+metrics.fAscent-1, m_data.c_str(),m_data.size());
   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX-1,iY+metrics.fAscent, m_data.c_str(),m_data.size());
   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX-1,iY+metrics.fAscent+1, m_data.c_str(),m_data.size());
   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX,iY+metrics.fAscent-1, m_data.c_str(),m_data.size());
   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX,iY+metrics.fAscent+1, m_data.c_str(),m_data.size());
   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX+1,iY+metrics.fAscent-1, m_data.c_str(),m_data.size());
   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX+1,iY+metrics.fAscent, m_data.c_str(),m_data.size());
   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX+1,iY+metrics.fAscent+1, m_data.c_str(),m_data.size());

   //now draw final text
   const_cast<TGGC*>(m_context)->SetForeground(oldForeground);
   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX,iY+metrics.fAscent, m_data.c_str(),m_data.size());

}

void
FWOutlinedTextTableCellRenderer::setData(const std::string& iData) {
   m_data = iData;
}

void 
FWOutlinedTextTableCellRenderer::setJustify(Justify iJustify)
{
   m_justify=iJustify;
}

//
// const member functions
//
UInt_t 
FWOutlinedTextTableCellRenderer::width() const
{
   if(m_data.size()) {
      return m_font->TextWidth(m_data.c_str(),-1)+3;
   }
   return 0;
}
UInt_t 
FWOutlinedTextTableCellRenderer::height() const
{
   return m_font->TextHeight();
}

const TGFont* 
FWOutlinedTextTableCellRenderer::font() const
{
   return m_font;
}

//
// static member functions
//
const TGGC&  
FWOutlinedTextTableCellRenderer::getDefaultGC()
{
   static const TGGC* s_default = gClient->GetResourcePool()->GetFrameGC();
   return *s_default;
}

const TGGC &
FWOutlinedTextTableCellRenderer::getFillGC()
{
   // Return graphics context for highlighted frame background.
   static const TGGC* s_default = 0;
   if (!s_default) {
      GCValues_t gval;
      gval.fMask = kGCForeground | kGCBackground | kGCTile |
                   kGCFillStyle  | kGCGraphicsExposures;
      gval.fForeground = gClient->GetResourcePool()->GetFrameHiliteColor();
      gval.fBackground = gClient->GetResourcePool()->GetFrameBgndColor();
      gval.fFillStyle  = kFillTiled;
      gval.fTile       = gClient->GetResourcePool()->GetCheckeredPixmap();
      gval.fGraphicsExposures = kFALSE;
      s_default = gClient->GetGC(&gval, kTRUE);
   }
   return *s_default;
}
