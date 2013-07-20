// $Id: FWTriggerTableViewTableManager.cc,v 1.5 2012/09/27 16:51:25 eulisse Exp $

#include <cassert>
#include "TGClient.h"
#include "Fireworks/Core/interface/FWTriggerTableViewTableManager.h"
#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"

FWTriggerTableViewTableManager::FWTriggerTableViewTableManager (const FWTriggerTableView *view)
   : m_view(view),
     m_graphicsContext(0),
     m_renderer(0)
{
   GCValues_t gc = *(m_view->m_tableWidget->GetWhiteGC().GetAttributes());
   m_graphicsContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&gc,kTRUE);
   m_renderer = new FWTextTableCellRenderer(m_graphicsContext,m_graphicsContext);
}

FWTriggerTableViewTableManager::~FWTriggerTableViewTableManager ()
{
   delete m_renderer;
}

int
FWTriggerTableViewTableManager::numberOfRows() const
{
   if ( !m_view->m_columns.empty() )
      return m_view->m_columns.front().values.size();
   else
      return 0;
}

int
FWTriggerTableViewTableManager::numberOfColumns() const
{
   return m_view->m_columns.size();
}

std::vector<std::string>
FWTriggerTableViewTableManager::getTitles () const
{
   unsigned int n = numberOfColumns();
   std::vector<std::string> ret;
   ret.reserve(n);
   for (unsigned int i = 0; i < n; ++i) {
      ret.push_back(m_view->m_columns.at(i).title);
   }
   return ret;
}

int
FWTriggerTableViewTableManager::unsortedRowNumber(int iSortedRowNumber) const
{
   if (iSortedRowNumber >= (int)m_sortedToUnsortedIndices.size())
      return 0;
   return m_sortedToUnsortedIndices[iSortedRowNumber];
}

FWTableCellRendererBase
*FWTriggerTableViewTableManager::cellRenderer(int iSortedRowNumber,
                                              int iCol) const
{
   const int realRowNumber = unsortedRowNumber(iSortedRowNumber);
   const int acceptColumn = 1;
   if ( !m_view->m_columns.empty() &&
        int(m_view->m_columns.size())>iCol &&
        int(m_view->m_columns.front().values.size())>realRowNumber ) {
      bool accepted = std::string(m_view->m_columns.at(acceptColumn).values.at(realRowNumber)) == "1";
      if ((m_view->backgroundColor() == kBlack) == accepted)
	m_graphicsContext->SetForeground(0xe0e0e0);
      else
	m_graphicsContext->SetForeground(0x404040);
      m_renderer->setData(m_view->m_columns.at(iCol).values.at(realRowNumber), false);
   } else {
      m_renderer->setData("invalid", false);
   }
   return m_renderer;
}

void
FWTriggerTableViewTableManager::dataChanged()
{
   m_sortedToUnsortedIndices.clear();
   for ( int i=0; i< numberOfRows(); ++i)
      m_sortedToUnsortedIndices.push_back(i);
   FWTableManagerBase::dataChanged();
}

namespace {
   template <typename TMap>
   void doSort(int col,
               const std::vector<FWTriggerTableView::Column>& iData,
               TMap& iOrdered,
               std::vector<int>& oRowToIndex)
   {
      unsigned int index=0;
      for(std::vector<std::string>::const_iterator it = iData.at(col).values.begin(),
          itEnd = iData.at(col).values.end(); it!=itEnd; ++it,++index) {
         iOrdered.insert(std::make_pair(*it,index));
      }
      unsigned int row = 0;
      for(typename TMap::iterator it = iOrdered.begin(),
          itEnd = iOrdered.end();
          it != itEnd;
          ++it,++row) {
         oRowToIndex[row]=it->second;
      }
   }
}

void
FWTriggerTableViewTableManager::implSort(int col, bool sortOrder)
{
   if(sortOrder) {
      std::multimap<std::string,int,std::greater<std::string> > ordered;
      doSort(col, m_view->m_columns, ordered, m_sortedToUnsortedIndices);
   } else {
      std::multimap<std::string,int> ordered;
      doSort(col, m_view->m_columns, ordered, m_sortedToUnsortedIndices);
   }
}

