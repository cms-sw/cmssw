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
// $Id$
//

// system include files
#include "TVirtualX.h"

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
FWColumnLabelCellRenderer::FWColumnLabelCellRenderer(GContext_t iContext, FontStruct_t iFontStruct):
FWTextTableCellRenderer(iContext,iFontStruct), 
m_sortOrder(fireworks::table::kNotSorted) {}

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
   m_sortOrder = iOrder;
}

void 
FWColumnLabelCellRenderer::draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
{
   using namespace fireworks::table;
   UInt_t h = height();
   
   if(kAscendingSort == m_sortOrder) {
      gVirtualX->DrawLine(iID, graphicsContext(), iX+h/2, iY+2, iX, iY+h-2);
      gVirtualX->DrawLine(iID, graphicsContext(), iX, iY+h-2, iX+h, iY+h-2);
      gVirtualX->DrawLine(iID, graphicsContext(), iX+h/2, iY+2, iX+h, iY+h-2);
   }
   if(kDescendingSort == m_sortOrder){
      gVirtualX->DrawLine(iID, graphicsContext(), iX, iY+2, iX+h, iY+2);
      gVirtualX->DrawLine(iID, graphicsContext(), iX+h/2, iY+h-2, iX+h, iY+2);
      gVirtualX->DrawLine(iID, graphicsContext(), iX+h/2, iY+h-2, iX, iY+2);      
   }
   FWTextTableCellRenderer::draw(iID,iX+kGap+h,iY,iWidth-kGap-h,iHeight);
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
   UInt_t h = height();
   return FWTextTableCellRenderer::width()+kGap+h;
}

//
// static member functions
//
