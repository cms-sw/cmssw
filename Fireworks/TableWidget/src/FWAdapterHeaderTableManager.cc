// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWAdapterHeaderTableManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:44:45 EST 2009
// $Id: FWAdapterHeaderTableManager.cc,v 1.3 2012/02/22 00:15:44 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/TableWidget/src/FWAdapterHeaderTableManager.h"
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
FWAdapterHeaderTableManager::FWAdapterHeaderTableManager(const FWTableManagerBase* iTable) :
m_table(iTable), 
m_renderer(new FWColumnLabelCellRenderer(&(FWTextTableCellRenderer::getDefaultGC()), iTable->cellDataIsSortable())), 
m_sortedColumn(-1),
m_descendingSort(true) 
{}

// FWAdapterHeaderTableManager::FWAdapterHeaderTableManager(const FWAdapterHeaderTableManager& rhs)
// {
//    // do actual copying here;
// }

FWAdapterHeaderTableManager::~FWAdapterHeaderTableManager()
{
}

//
// assignment operators
//
// const FWAdapterHeaderTableManager& FWAdapterHeaderTableManager::operator=(const FWAdapterHeaderTableManager& rhs)
// {
//   //An exception safe implementation is
//   FWAdapterHeaderTableManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWAdapterHeaderTableManager::implSort(int col, bool sortOrder) 
{ 
   m_sortedColumn=col;
   m_descendingSort=sortOrder;
}

//
// const member functions
//
int 
FWAdapterHeaderTableManager::numberOfRows() const { return 1;}

int 
FWAdapterHeaderTableManager::numberOfColumns() const { return m_table->numberOfColumns();}

int 
FWAdapterHeaderTableManager::unsortedRowNumber(int iRow) const
{
   return iRow;
}


std::vector<std::string> 
FWAdapterHeaderTableManager::getTitles() const {
   return m_table->getTitles();
}

FWTableCellRendererBase* 
FWAdapterHeaderTableManager::cellRenderer(int /*iRow*/, int iCol) const
{
   if(iCol==m_sortedColumn) {
      if(m_descendingSort) {
         m_renderer->setSortOrder(fireworks::table::kDescendingSort);         
      } else {
         m_renderer->setSortOrder(fireworks::table::kAscendingSort);         
      }
   } else {
      m_renderer->setSortOrder(fireworks::table::kNotSorted);
   }
   if(iCol < m_table->numberOfColumns()) {
      m_renderer->setData( *(getTitles().begin()+iCol),false );
   } else {
      m_renderer->setData("",false);
   }
   return m_renderer;
}

//
// static member functions
//
