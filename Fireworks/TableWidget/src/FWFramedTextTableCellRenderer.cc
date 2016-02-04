// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWFramedTextTableCellRenderer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:43:54 EST 2009
// $Id: FWFramedTextTableCellRenderer.cc,v 1.3 2010/06/18 12:44:24 yana Exp $
//

// system include files
#include <iostream>
#include "TGClient.h"
#include "TGFont.h"
#include "TVirtualX.h"

// user include files
#include "Fireworks/TableWidget/interface/FWFramedTextTableCellRenderer.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWFramedTextTableCellRenderer::FWFramedTextTableCellRenderer(const TGGC* iTextContext, const TGGC* iFillContext, Justify iJustify):
   m_context(iTextContext),
   m_frameContext(iFillContext),
   m_font(0),
   m_justify(iJustify)
{
   //TGGC* tggc= gClient->GetGCPool()->GetGC(iContext);
   m_font = gClient->GetFontPool()->FindFontByHandle(m_context->GetFont());
}

// FWFramedTextTableCellRenderer::FWFramedTextTableCellRenderer(const FWFramedTextTableCellRenderer& rhs)
// {
//    // do actual copying here;
// }

FWFramedTextTableCellRenderer::~FWFramedTextTableCellRenderer()
{
}

//
// assignment operators
//
// const FWFramedTextTableCellRenderer& FWFramedTextTableCellRenderer::operator=(const FWFramedTextTableCellRenderer& rhs)
// {
//   //An exception safe implementation is
//   FWFramedTextTableCellRenderer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWFramedTextTableCellRenderer::draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
{
   // GContext_t c = m_frameContext->GetGC();

   gVirtualX->DrawLine(iID,m_frameContext->GetGC(),iX-1,iY-1,iX-1,iY+iHeight);
   gVirtualX->DrawLine(iID,m_frameContext->GetGC(),iX+iWidth,iY-1,iX+iWidth,iY+iHeight);
   gVirtualX->DrawLine(iID,m_frameContext->GetGC(),iX-1,iY-1,iX+iWidth,iY-1);
   gVirtualX->DrawLine(iID,m_frameContext->GetGC(),iX-1,iY+iHeight,iX+iWidth,iY+iHeight);

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

   gVirtualX->DrawString(iID,m_context->GetGC(),iX+dX,iY+metrics.fAscent, m_data.c_str(),m_data.size());
}

void
FWFramedTextTableCellRenderer::setData(const std::string& iData) {
   m_data = iData;
}

void 
FWFramedTextTableCellRenderer::setJustify(Justify iJustify)
{
   m_justify=iJustify;
}

//
// const member functions
//
UInt_t 
FWFramedTextTableCellRenderer::width() const
{
   if(m_data.size()) {
      return m_font->TextWidth(m_data.c_str(),-1)+3;
   }
   return 0;
}
UInt_t 
FWFramedTextTableCellRenderer::height() const
{
   return m_font->TextHeight();
}

const TGFont* 
FWFramedTextTableCellRenderer::font() const
{
   return m_font;
}

//
// static member functions
//
const TGGC&  
FWFramedTextTableCellRenderer::getDefaultGC()
{
   static const TGGC* s_default = gClient->GetResourcePool()->GetFrameGC();
   return *s_default;
}

const TGGC &
FWFramedTextTableCellRenderer::getFillGC()
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
