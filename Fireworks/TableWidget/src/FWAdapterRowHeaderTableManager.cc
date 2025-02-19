// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWAdapterRowHeaderTableManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:45:01 EST 2009
// $Id: FWAdapterRowHeaderTableManager.cc,v 1.1 2009/02/03 20:33:03 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/TableWidget/src/FWAdapterRowHeaderTableManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWAdapterRowHeaderTableManager::FWAdapterRowHeaderTableManager(FWTableManagerBase* iTable) :
m_table(iTable)
 {
   iTable->Connect("dataChanged()","FWTableManagerBase",static_cast<FWTableManagerBase*>(this),"dataChanged()");
   iTable->Connect("visualPropertiesChanged()","FWTableManagerBase",static_cast<FWTableManagerBase*>(this),"visualPropertiesChanged()");
}

// FWAdapterRowHeaderTableManager::FWAdapterRowHeaderTableManager(const FWAdapterRowHeaderTableManager& rhs)
// {
//    // do actual copying here;
// }

FWAdapterRowHeaderTableManager::~FWAdapterRowHeaderTableManager()
{
}

//
// assignment operators
//
// const FWAdapterRowHeaderTableManager& FWAdapterRowHeaderTableManager::operator=(const FWAdapterRowHeaderTableManager& rhs)
// {
//   //An exception safe implementation is
//   FWAdapterRowHeaderTableManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWAdapterRowHeaderTableManager::implSort(int col, bool sortOrder) 
{ 
}

//
// const member functions
//
int 
FWAdapterRowHeaderTableManager::numberOfRows() const { return m_table->numberOfRows();}

int 
FWAdapterRowHeaderTableManager::numberOfColumns() const { return 1;}

int 
FWAdapterRowHeaderTableManager::unsortedRowNumber(int iRow) const
{
   return m_table->unsortedRowNumber(iRow);
}


std::vector<std::string> 
FWAdapterRowHeaderTableManager::getTitles() const {
   std::vector<std::string> names(1,std::string("labels"));
   return names;
}

FWTableCellRendererBase* 
FWAdapterRowHeaderTableManager::cellRenderer(int iRow, int /*iCol*/) const
{
   return m_table->rowHeader(iRow);
}

//
// static member functions
//
