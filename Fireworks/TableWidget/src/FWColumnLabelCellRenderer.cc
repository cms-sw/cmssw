// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWColumnLabelCellRenderer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:44:04 EST 2009
// $Id: FWColumnLabelCellRenderer.cc,v 1.3 2012/02/22 00:15:44 amraktad Exp $
//

#include <cassert>

// system include files
#include "TVirtualX.h"
#include "TGFont.h"

// user include files
#include "Fireworks/TableWidget/interface/FWColumnLabelCellRenderer.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWColumnLabelCellRenderer::FWColumnLabelCellRenderer(const TGGC* iContext, bool isSortable):
FWTextTableCellRenderer(iContext), 
m_sortOrder(fireworks::table::kNotSorted),
m_sizeOfOrderIcon(height()),
m_isSortable(isSortable)
{
  if (m_isSortable)
  {
   FontMetrics_t metrics;
   font()->GetFontMetrics(&metrics);

   m_sizeOfOrderIcon -= 2*metrics.fDescent;
   m_sizeOfOrderIconStartX = metrics.fDescent;
   //we want the number to be even so that the point of the triangle 
   // will be right in the middle
   if(1 == m_sizeOfOrderIcon % 2) {
      ++m_sizeOfOrderIcon;
   }
  }
}


// FWColumnLabelCellRenderer::FWColumnLabelCellRenderer(const FWColumnLabelCellRenderer& rhs)
// {
//    // do actual copying here;
// }

FWColumnLabelCellRenderer::~FWColumnLabelCellRenderer()
{
}

//
// assignment operators
//
// const FWColumnLabelCellRenderer& FWColumnLabelCellRenderer::operator=(const FWColumnLabelCellRenderer& rhs)
// {
//   //An exception safe implementation is
//   FWColumnLabelCellRenderer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void FWColumnLabelCellRenderer::setSortOrder(fireworks::table::SortOrder iOrder) {
  
  if (!m_isSortable) assert (iOrder ==  fireworks::table::kNotSorted);
  
  m_sortOrder = iOrder;
}

void 
FWColumnLabelCellRenderer::draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
{
  if (m_isSortable) 
  {
    using namespace fireworks::table;
    UInt_t h = m_sizeOfOrderIcon;
    const GContext_t c = graphicsContext()->GetGC();
    int dY = m_sizeOfOrderIconStartX;
    
    if(kAscendingSort == m_sortOrder) {
      gVirtualX->DrawLine(iID, c, iX+h/2, iY+2+dY,   iX,   iY+h-2+dY);
      gVirtualX->DrawLine(iID, c, iX,     iY+h-2+dY, iX+h, iY+h-2+dY);
      gVirtualX->DrawLine(iID, c, iX+h/2, iY+2+dY,   iX+h, iY+h-2+dY);
    }
    if(kDescendingSort == m_sortOrder){
      gVirtualX->DrawLine(iID, c, iX,     iY+2+dY,   iX+h, iY+2+dY);
      gVirtualX->DrawLine(iID, c, iX+h/2, iY+h-2+dY, iX+h, iY+2+dY);
      gVirtualX->DrawLine(iID, c, iX+h/2, iY+h-2+dY, iX,   iY+2+dY);      
    }
    FWTextTableCellRenderer::draw(iID,iX+kGap+h,iY,iWidth-kGap-h,iHeight);
  }
  else
  {
    return FWTextTableCellRenderer::draw(iID, iX+kGap, iY-kGap, iWidth, iHeight);
  }

}

//
// const member functions
//
fireworks::table::SortOrder FWColumnLabelCellRenderer::sortOrder() const
{
   return m_sortOrder;
}

UInt_t FWColumnLabelCellRenderer::width() const
{
  if (m_isSortable)
   return FWTextTableCellRenderer::width()+kGap+m_sizeOfOrderIcon;
  else {
    return FWTextTableCellRenderer::width() + kGap*2;
  }

}

//
// static member functions
//
