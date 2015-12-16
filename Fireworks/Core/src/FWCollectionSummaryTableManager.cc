// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCollectionSummaryTableManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sun Feb 22 10:13:39 CST 2009
//

// system include files
#include <sstream>
#include <boost/bind.hpp>
#include "TClass.h"

// user include files
#include "Fireworks/Core/src/FWCollectionSummaryTableManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWItemValueGetter.h"
#include "Fireworks/Core/src/FWCollectionSummaryWidget.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWCollectionSummaryTableManager::FWCollectionSummaryTableManager(FWEventItem* iItem, const TGGC* iContext, const TGGC* iHighlightContext,
                                                                 FWCollectionSummaryWidget* iWidget):
   m_collection(iItem),
   m_renderer(iContext,iHighlightContext),
   m_bodyRenderer(iContext, iHighlightContext, FWTextTableCellRenderer::kJustifyRight),
   m_widget(iWidget)
{
   m_collection->changed_.connect(boost::bind(&FWCollectionSummaryTableManager::modelIdChanges,this));
   m_collection->itemChanged_.connect(boost::bind(&FWCollectionSummaryTableManager::dataChanged,this));
   
   //try to find the default columns
   std::vector<std::pair<std::string,std::string> > s_names;
   edm::TypeWithDict type(*(m_collection->modelType()->GetTypeInfo()));

   
   dataChanged();
}

void FWCollectionSummaryTableManager::modelIdChanges()
{
   if (m_collection->showFilteredEntries() || m_collection->filterExpression().empty())
       dataChanged();
   else
      FWTableManagerBase::dataChanged();
}


// FWCollectionSummaryTableManager::FWCollectionSummaryTableManager(const FWCollectionSummaryTableManager& rhs)
// {
//    // do actual copying here;
// }

FWCollectionSummaryTableManager::~FWCollectionSummaryTableManager()
{
}

//
// assignment operators
//
// const FWCollectionSummaryTableManager& FWCollectionSummaryTableManager::operator=(const FWCollectionSummaryTableManager& rhs)
// {
//   //An exception safe implementation is
//   FWCollectionSummaryTableManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
namespace {
   template<typename S>
   void doSort(const FWEventItem& iItem,
              const FWItemValueGetter& iGetter, int iCol,
               std::multimap<double,int,S>& iMap, 
               std::vector<int>& oNewSort) {
      int size = iItem.size();
      for(int index = 0; index < size; ++index) {
         if (iItem.modelInfo(index).displayProperties().filterPassed()) {
         iMap.insert(std::make_pair(iGetter.valueFor(iItem.modelData(index), iCol),
                                       index));
         }
      }
      std::vector<int>::iterator itVec = oNewSort.begin();
      for(typename std::map<double,int,S>::iterator it = iMap.begin(), itEnd = iMap.end();
          it != itEnd;
          ++it,++itVec) {
         *itVec = it->second;
      }
   }
}

void 
FWCollectionSummaryTableManager::implSort(int iCol, bool iSortOrder)
{
   if(iSortOrder) {
      std::multimap<double,int, std::greater<double> > s;
      doSort(*m_collection, m_collection->valueGetter(), iCol, s, m_sortedToUnsortedIndicies);
   } else {
      std::multimap<double,int, std::less<double> > s;
      doSort(*m_collection, m_collection->valueGetter(), iCol, s, m_sortedToUnsortedIndicies);
   }
}

void 
FWCollectionSummaryTableManager::buttonReleasedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY)
{
   Int_t realRow = unsortedRowNumber(row);
   int hit = m_renderer.clickHit(relX,relY);
   if(hit == FWCollectionSummaryModelCellRenderer::kMiss) {
      return;
   }
   if(hit == FWCollectionSummaryModelCellRenderer::kHitColor) {
      m_widget->itemColorClicked(realRow,event->fXRoot, event->fYRoot+12-relY);
      return;
   }
   FWEventItem::ModelInfo mi = m_collection->modelInfo(realRow);
   FWDisplayProperties dp = mi.displayProperties();
   if(hit == FWCollectionSummaryModelCellRenderer::kHitCheck) {
      dp.setIsVisible(!dp.isVisible());
   }
   m_collection->setDisplayProperties(realRow,dp);
}

//
// const member functions
//
int 
FWCollectionSummaryTableManager::numberOfRows() const
{
   int cs= m_collection->size();
   if (m_collection->showFilteredEntries() || m_collection->filterExpression().empty())
   {
      return cs;
   }
   else
   {
      
      int n = 0;
      for(int index = 0; index < cs; ++index) {
         if (m_collection->modelInfo(index).displayProperties().filterPassed()) { ++n;}
      }
      return n;
   }
}

int 
FWCollectionSummaryTableManager::numberOfColumns() const {
   return m_collection->valueGetter().numValues();
}

std::vector<std::string> 
FWCollectionSummaryTableManager::getTitles() const {



   //return titles;
      return  m_collection->valueGetter().getTitles();
}

int 
FWCollectionSummaryTableManager::unsortedRowNumber(int iSortedRowNumber) const
{
   return m_sortedToUnsortedIndicies[iSortedRowNumber];
}

FWTableCellRendererBase* 
FWCollectionSummaryTableManager::cellRenderer(int iSortedRowNumber, int iCol) const
{
   if(!m_collection->valueGetter().numValues()) {
      return 0;
   }
   if(iSortedRowNumber >= static_cast<int>(m_collection->size())) {
      m_bodyRenderer.setData("",false);
      return &m_bodyRenderer;
   }
   int index = m_sortedToUnsortedIndicies[iSortedRowNumber];
   std::stringstream s;
   s.setf(std::ios_base::fixed,std::ios_base::floatfield);
   s.precision( m_collection->valueGetter().precision(iCol));
   double v = m_collection->valueGetter().valueFor(m_collection->modelData(index), iCol);
   s <<v;
   m_bodyRenderer.setData(s.str(), m_collection->modelInfo(index).isSelected());
   return &m_bodyRenderer;
}

bool 
FWCollectionSummaryTableManager::hasRowHeaders() const
{
   return true;
}

FWTableCellRendererBase* 
FWCollectionSummaryTableManager::rowHeader(int iSortedRowNumber) const
{
   if(iSortedRowNumber >= static_cast<int>(numberOfRows())) {
      return 0;
   }
   int index = m_sortedToUnsortedIndicies[iSortedRowNumber];
   m_renderer.setData(m_collection,
                      index);
   return &m_renderer;
}

void
FWCollectionSummaryTableManager::dataChanged() 
{
   m_sortedToUnsortedIndicies.clear();
   size_t n = numberOfRows();
   m_sortedToUnsortedIndicies.reserve(n);
   for(int i=0; i< static_cast<int>(m_collection->size());++i) {
      if (m_collection->filterExpression().empty() || m_collection->showFilteredEntries() || m_collection->modelInfo(i).displayProperties().filterPassed()) {
          m_sortedToUnsortedIndicies.push_back(i);
      }
   }
   FWTableManagerBase::dataChanged();
}
//
// static member functions
//
