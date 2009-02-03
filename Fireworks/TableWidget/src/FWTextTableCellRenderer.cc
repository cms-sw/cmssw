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
// $Id$
//

// system include files
#include "TGClient.h"
#include "TGResourcePool.h"
#include "TGFont.h"
#include "TVirtualX.h"

// user include files
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTextTableCellRenderer::FWTextTableCellRenderer(GContext_t iContext,FontStruct_t iFontStruct):
   m_context(iContext),
   m_fontStruct(iFontStruct),
   m_font(0),
   m_isSelected(false)
{
   m_font = gClient->GetFontPool()->FindFont(m_fontStruct);
   if(0==m_font) {
      m_font = gClient->GetFontPool()->GetFont(getDefaultFontStruct());
      m_fontStruct = m_font->GetFontStruct();
   }
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
   if(m_isSelected){
      GContext_t c = getHighlightGC()();
      gVirtualX->FillRectangle(iID, c, iX, iY, iWidth, iHeight);
   }
   FontMetrics_t metrics;
   m_font->GetFontMetrics(&metrics);
   m_font->DrawChars(iID, m_context, m_data.c_str(),m_data.size(),iX,iY+metrics.fAscent);
}

void
FWTextTableCellRenderer::setData(const std::string& iData, bool iIsSelected) {
   m_data = iData;
   m_isSelected=iIsSelected;
}

//
// const member functions
//
UInt_t 
FWTextTableCellRenderer::width() const
{
   return m_font->TextWidth(m_data.c_str(),-1);
}
UInt_t 
FWTextTableCellRenderer::height() const
{
   return m_font->TextHeight();
}

//
// static member functions
//
FontStruct_t 
FWTextTableCellRenderer::getDefaultFontStruct()
{
   static const TGFont* s_font = gClient->GetResourcePool()->GetDefaultFont();
   return s_font->GetFontStruct();
}

const TGGC&  
FWTextTableCellRenderer::getDefaultGC()
{
   static const TGGC* s_default = gClient->GetResourcePool()->GetFrameGC();
   return *s_default;
}

const TGGC &
FWTextTableCellRenderer::getHighlightGC()
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
