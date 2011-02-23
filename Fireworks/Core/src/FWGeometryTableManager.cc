// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableManager
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Thomas McCauley, Alja Mrak-Tadel
//         Created:  Thu Jan 27 14:50:57 CET 2011
// $Id$
//

// system include files

// user include files
#include <iostream>

#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "Fireworks/TableWidget/interface/GlobalContexts.h"
#include "TGeoManager.h"

FWGeometryTableManager::FWGeometryTableManager()
   : m_selectedRow(-1)
{ 
   m_renderer.setGraphicsContext(&fireworks::boldGC());
   m_daughterRenderer.setGraphicsContext(&fireworks::italicGC());
   reset();
}

void FWGeometryTableManager::implSort(int, bool)
{
   recalculateVisibility();
}

int FWGeometryTableManager::unsortedRowNumber(int unsorted) const
{
   return unsorted;
}

int FWGeometryTableManager::numberOfRows() const 
{
   return m_row_to_index.size();
}

 int FWGeometryTableManager::numberOfColumns() const 
{
   return 2;
}
   

std::vector<std::string> FWGeometryTableManager::getTitles() const 
{
   std::vector<std::string> returnValue;
   returnValue.reserve(numberOfColumns());
   returnValue.push_back("Name");
   returnValue.push_back("Title");
   return returnValue;
}
  
FWTableCellRendererBase* FWGeometryTableManager::cellRenderer(int iSortedRowNumber, int iCol) const
{
   bool printNodes = false;
   if (printNodes)
   {
      std::cout<<"m_row_to_index.size(), iSortedRowNumber: "
               << m_row_to_index.size() <<" "<< iSortedRowNumber <<std::endl;
   }
   if (static_cast<int>(m_row_to_index.size()) <= iSortedRowNumber)
   {
      m_renderer.setData(std::string("Ack!"), false);
      return &m_renderer;
   }       

   FWTextTreeCellRenderer* renderer = &m_renderer;
   int unsortedRow =  m_row_to_index[iSortedRowNumber];
   const NodeInfo& data = m_nodeInfo[unsortedRow];

   std::string name;
   std::string title;

   name = data.name;
   title = data.title;
   if(printNodes)
      std::cout<<"name, title: "<< name <<"  "<< title <<std::endl;

   if (iCol == 0)
      renderer->setData(name, false);
   else if (iCol == 1)
      renderer->setData(title, false);
   else
      renderer->setData(std::string(), false);


   renderer->setIndentation(0);
   return renderer;
}

void FWGeometryTableManager::setExpanded(int row)
{
   if ( row == -1 )
      return;
      
   recalculateVisibility();
   dataChanged();
   visualPropertiesChanged();
}
  
void FWGeometryTableManager::setSelection (int row, int column, int mask) 
{
   if(mask == 4) 
   {
      if( row == m_selectedRow) 
      {
         row = -1;
      }
   }
   changeSelection(row, column);
}

const std::string FWGeometryTableManager::title() const 
{
   return "Geometry";
}

int FWGeometryTableManager::selectedRow() const 
{
   return m_selectedRow;
}

int FWGeometryTableManager::selectedColumn() const 
{
   return m_selectedColumn;
}
 
bool FWGeometryTableManager::rowIsSelected(int row) const 
{
   return m_selectedRow == row;
}

void FWGeometryTableManager::reset() 
{
   changeSelection(-1, -1);
   recalculateVisibility();
   dataChanged();
   visualPropertiesChanged();
}

void FWGeometryTableManager::recalculateVisibility()
{
   m_row_to_index.clear();
        
   for ( size_t i = 0, e = m_nodeInfo.size(); i != e; ++i )
      m_row_to_index.push_back(i);
} 

void FWGeometryTableManager::fillNodeInfo(TGeoManager* geoManager)
{
   std::cout<<"fillNodeInfo in"<<std::endl;
     
   TGeoVolume* topVolume = geoManager->GetTopVolume();

   NodeInfo topNodeInfo;
   topNodeInfo.name = geoManager->GetTopNode()->GetName();
   topNodeInfo.title = geoManager->GetTopNode()->GetTitle();
   m_nodeInfo.push_back(topNodeInfo);

   for ( size_t n = 0, 
            ne = topVolume->GetNode(0)->GetNdaughters();
         n != ne; ++n )
   {
      NodeInfo nodeInfo;
      nodeInfo.name = topVolume->GetNode(0)->GetDaughter(n)->GetName();
      nodeInfo.title = topVolume->GetNode(0)->GetDaughter(n)->GetTitle();
      m_nodeInfo.push_back(nodeInfo);
   }

   std::cout<<"fillNodeInfo out"<<std::endl;
   reset();
}
  

void FWGeometryTableManager::changeSelection(int iRow, int iColumn)
{      
   if (iRow == m_selectedRow && iColumn == m_selectedColumn)
      return;
      
   m_selectedRow = iRow;
   m_selectedColumn = iColumn;

   indexSelected_(iRow, iColumn);
   visualPropertiesChanged();
}    
