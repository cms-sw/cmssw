// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWTableManagerBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:40:44 EST 2009
// $Id: FWTableManagerBase.cc,v 1.6 2011/03/09 14:20:45 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTableCellRendererBase.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTableManagerBase::FWTableManagerBase():
m_sortColumn(-1),
m_sortOrder(false)
{
}

// FWTableManagerBase::FWTableManagerBase(const FWTableManagerBase& rhs)
// {
//    // do actual copying here;
// }

FWTableManagerBase::~FWTableManagerBase()
{
}

//
// assignment operators
//
// const FWTableManagerBase& FWTableManagerBase::operator=(const FWTableManagerBase& rhs)
// {
//   //An exception safe implementation is
//   FWTableManagerBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWTableManagerBase::sort(int col, bool sortOrder)
{
   if(col <= numberOfColumns()) {
      m_sortColumn = col;
      m_sortOrder = sortOrder;
      implSort(col,sortOrder);
      visualPropertiesChanged();
   }
}

void FWTableManagerBase::dataChanged()
{
   if(-1 != m_sortColumn) {
      implSort(m_sortColumn,m_sortOrder);
   }
   Emit("dataChanged()");
}
      
void FWTableManagerBase::visualPropertiesChanged()
{
   Emit("visualPropertiesChanged()");
}

//
// const member functions
//
unsigned int FWTableManagerBase::cellHeight() const
{
   FWTableCellRendererBase* cr = cellRenderer(0,0);
   if(cr) {
      return cr->height();
   }
   if(hasRowHeaders()) {
      cr = rowHeader(0);
      if(cr) {
         return cr->height();
      }
   }
   return 0;
}
      
std::vector<unsigned int> FWTableManagerBase::maxWidthForColumns() const
{
   std::vector<unsigned int> returnValue;
   returnValue.reserve(numberOfColumns());
   const int numCols= numberOfColumns();
   const int numRows = numberOfRows();
   for(int col = 0; col < numCols; ++col) {
      unsigned int max = 0;
      for(int row=0; row < numRows; ++row) {
         unsigned int width = cellRenderer(row,col)->width();
         if(width > max) {
            max = width;
         }
      }
      returnValue.push_back(max);
   }
   return returnValue;
}

bool FWTableManagerBase::hasLabelHeaders() const
{
   return true;
}

bool FWTableManagerBase::hasRowHeaders() const
{
   return false;
}
FWTableCellRendererBase* FWTableManagerBase::rowHeader(int iRow) const
{
   return 0;
}

void 
FWTableManagerBase::buttonPressedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY)
{
}
void 
FWTableManagerBase::buttonReleasedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY)
{
}

//
// static member functions
//
ClassImp(FWTableManagerBase)
