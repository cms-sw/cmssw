
#include <math.h>
#include <sstream>
#include <cassert>

#include "TClass.h"
#include "TGClient.h"

#include "Fireworks/Core/interface/FWTableViewTableManager.h"
#include "Fireworks/Core/interface/FWTableView.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/interface/FWFramedTextTableCellRenderer.h"

FWTableViewTableManager::FWTableViewTableManager (const FWTableView *view)
     : m_view(view),
       m_graphicsContext(0),
       m_renderer(0),
       m_rowContext(0),
       m_rowRenderer(0),
       m_tableFormats(0),
       m_caughtExceptionInCellRender(false)
{
     GCValues_t gc = *(m_view->m_tableWidget->GetWhiteGC().GetAttributes());
     m_graphicsContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&gc,kTRUE);
     m_highlightContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&gc,kTRUE);
     m_highlightContext->SetForeground(gVirtualX->GetPixel(kBlue));
     m_highlightContext->SetBackground(gVirtualX->GetPixel(kBlue));
     m_renderer = new FWTextTableCellRenderer(m_graphicsContext,
					      m_highlightContext,
					      FWTextTableCellRenderer::kJustifyRight);
     //m_rowContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&gc,kTRUE);
     //m_rowContext->SetForeground(gVirtualX->GetPixel(kWhite));
     //m_rowContext->SetBackground(gVirtualX->GetPixel(kBlack));
     m_rowFillContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&gc,kTRUE);
     m_rowRenderer = new FWFramedTextTableCellRenderer(m_graphicsContext,
                                                      m_rowFillContext,
                                                      FWFramedTextTableCellRenderer::kJustifyRight);
   
}

FWTableViewTableManager::~FWTableViewTableManager ()
{
     delete m_renderer;
     delete m_rowRenderer;
}

const FWEventItem*
 FWTableViewTableManager::collection() const
{
   return m_view->item();
}

int FWTableViewTableManager::numberOfRows() const
{
   if (collection () != 0) {
      if (collection()->showFilteredEntries() || collection()->filterExpression().empty())
      {
         return collection()->size();
      }
      else
      {
         int cs = collection()->size();
         int n = 0;
         for(int index = 0; index < cs; ++index) {
            if (collection()->modelInfo(index).displayProperties().filterPassed()) { ++n;}
         }
         return n;
      }
   }
     else return 0;
}

int FWTableViewTableManager::numberOfColumns() const
{
     return m_evaluators.size();
}

std::vector<std::string> FWTableViewTableManager::getTitles () const
{
     unsigned int n = numberOfColumns();
     std::vector<std::string> ret;
     ret.reserve(n);
     for (unsigned int i = 0; i < n; ++i) {
	  ret.push_back(m_tableFormats->at(i).name);
// 	  printf("%s\n", ret.back().c_str());
     }
     return ret;
}

int FWTableViewTableManager::unsortedRowNumber(int iSortedRowNumber) const
{
     if (iSortedRowNumber >= (int)m_sortedToUnsortedIndices.size())
	  return 0;
     return m_sortedToUnsortedIndices[iSortedRowNumber];
}

FWTableCellRendererBase *FWTableViewTableManager::cellRenderer(int iSortedRowNumber, int iCol) const
{
     const int realRowNumber = unsortedRowNumber(iSortedRowNumber);
     if (m_view->item() != 0 &&
         m_view->item()->size() &&
	 m_view->item()->modelData(realRowNumber) != 0 &&
	 iCol < (int)m_evaluators.size()) {
	  double ret;
	  try {
// 	       printf("iCol %d, size %d\n", iCol, m_evaluators.size());
	       ret = m_evaluators[iCol].evalExpression(m_view->item()->modelData(realRowNumber));
	  } catch (...) {
	    if (!m_caughtExceptionInCellRender){
               fwLog(fwlog::kError) << "Error: caught exception in the cell renderer while evaluating an expression. Return -999. Error is suppressed in future\n";
	    }
	    m_caughtExceptionInCellRender = true;
	    ret = -999;
	  }
	  int precision = m_tableFormats->at(iCol).precision;
	  char s[100];
	  char fs[100];
	  switch (precision) {
	  case FWTableViewManager::TableEntry::INT:
	       snprintf(s, sizeof(s), "%d", int(rint(ret)));
	       break;
	  case FWTableViewManager::TableEntry::INT_HEX:
	       snprintf(s, sizeof(s), "0x%x", int(rint(ret)));
	       break;
	  case FWTableViewManager::TableEntry::BOOL:
	       snprintf(s, sizeof(s), int(rint(ret)) != 0 ? "true" : "false");
	       break;
	  default: 
	       snprintf(fs, sizeof(fs), "%%.%df", precision);
	       snprintf(s, sizeof(s), fs, ret);
	       break;
	  }
	  if (not m_view->item()->modelInfo(realRowNumber).isSelected()) {
	       if (m_view->item()->modelInfo(realRowNumber).displayProperties().isVisible())
                  if (m_view->m_manager->colorManager().background() == kBlack) {
                     m_graphicsContext->
                     SetForeground(gVirtualX->GetPixel(kWhite));
                  } else {
                     m_graphicsContext->
                     SetForeground(gVirtualX->GetPixel(kBlack));
                  }
	       else {
                  if (m_view->m_manager->colorManager().background() == kBlack) {
                     m_graphicsContext->SetForeground(0x888888);
                  } else { 
                     m_graphicsContext->SetForeground(0x888888);
                  }
	       }
	       m_renderer->setGraphicsContext(m_graphicsContext);
	  } else {
	       m_graphicsContext->
		    SetForeground(0xffffff);
	       m_renderer->setGraphicsContext(m_graphicsContext);
	  }
	  m_renderer->setData(s, m_view->item()->modelInfo(realRowNumber).isSelected());
     } else { 
	  m_renderer->setData("invalid", false);
     }
     return m_renderer;
}

namespace {
     struct itemOrderGt {
	  bool operator () (const std::pair<bool, double> &i1, 
			    const std::pair<bool, double> &i2) 
	       {
		    // sort first by visibility
		    if (i1.first and not i2.first)
			 return true;
		    if (i2.first and not i1.first)
			 return false;
		    // then by value
		    else return i1.second > i2.second;
	       }
     };
     struct itemOrderLt {
	  bool operator () (const std::pair<bool, double> &i1, 
			    const std::pair<bool, double> &i2) 
	       {
		    // sort first by visibility
		    if (i1.first and not i2.first)
			 return true;
		    if (i2.first and not i1.first)
			 return false;
		    // then by value
		    else return i1.second < i2.second;
	       }
     };
     template<typename S>
     void doSort(const FWEventItem& iItem,
		 int iCol,
		 const std::vector<FWExpressionEvaluator> &evaluators,
		 std::multimap<std::pair<bool, double>, int, S>& iMap,
		 std::vector<int>& oNewSort) 
     {
	  int size = iItem.size();
	  for(int index = 0; index < size; ++index) {
        if (iItem.showFilteredEntries() || iItem.modelInfo(index).displayProperties().filterPassed())
        {
	       double ret;
	       try {
// 	       printf("iCol %d, size %d\n", iCol, m_evaluators.size());
		    ret = evaluators[iCol].evalExpression(iItem.modelData(index));
	       } catch (...) {
		 ret = -999;
	       }
	       iMap.insert(std::make_pair(
				std::make_pair(
				     iItem.modelInfo(index).displayProperties().isVisible(), ret), 
				index));
        }
	  }
	  std::vector<int>::iterator itVec = oNewSort.begin();
	  for(typename std::multimap<std::pair<bool, double>,int,S>::iterator 
		   it = iMap.begin(), 
		   itEnd = iMap.end();
	      it != itEnd;
	      ++it,++itVec) {
	       *itVec = it->second;
	  }
     }
}

void FWTableViewTableManager::implSort(int iCol, bool iSortOrder)
{
   static const bool sort_down = true;
   if (iCol >= (int)m_evaluators.size())
      return;
   if (0!=m_view->item()) {
      //      printf("sorting %s\n", iSortOrder == sort_down ? "down" : "up");
      if (iSortOrder == sort_down) {
         std::multimap<std::pair<bool, double>, int, itemOrderGt> s;
         doSort(*m_view->item(), iCol, m_evaluators, s, m_sortedToUnsortedIndices);
      } else {
         std::multimap<std::pair<bool, double>, int, itemOrderLt> s;
         doSort(*m_view->item(), iCol, m_evaluators, s, m_sortedToUnsortedIndices);
      }
   }
   m_view->m_tableWidget->dataChanged();
}

void
FWTableViewTableManager::dataChanged() 
{
   if (0!=m_view->item()) {
      std::vector<int> visible;
      visible.reserve(m_view->item()->size());
      std::vector<int> invisible;
      invisible.reserve(m_view->item()->size());
      m_sortedToUnsortedIndices.clear();
      m_sortedToUnsortedIndices.reserve(m_view->item()->size());
      for(int i=0; i< static_cast<int>(m_view->item()->size()); ++i) {
           if (collection()->showFilteredEntries() || collection()->modelInfo(i).displayProperties().filterPassed())
           {
         if (m_view->item()->modelInfo(i).displayProperties().isVisible())
            visible.push_back(i);
         else invisible.push_back(i);
           }
      }
      m_sortedToUnsortedIndices.insert(m_sortedToUnsortedIndices.end(),
                                       visible.begin(), visible.end());
      m_sortedToUnsortedIndices.insert(m_sortedToUnsortedIndices.end(),
                                       invisible.begin(), invisible.end());
      
      if (collection()->showFilteredEntries() || collection()->filterExpression().empty())
          assert(m_sortedToUnsortedIndices.size() == m_view->item()->size());
      
   } else {
      m_sortedToUnsortedIndices.clear();
   }
   FWTableManagerBase::dataChanged();
}

void FWTableViewTableManager::updateEvaluators ()
{
     if (m_view->m_iColl == -1) {
	  //printf("what should I do with collection -1?\n");
          m_evaluators.clear();
	  return;
     }
     const FWEventItem *item = m_view->m_manager->items()[m_view->m_iColl];
     if(0==item) { return;}
     std::vector<FWExpressionEvaluator> &ev = m_evaluators;
     ev.clear();
     for (std::vector<FWTableViewManager::TableEntry>::const_iterator 
	       i = m_tableFormats->begin(),
	       end = m_tableFormats->end();
	  i != end; ++i) {
	  try {
	       ev.push_back(FWExpressionEvaluator(i->expression, item->modelType()->GetName()));
	  } catch (...) {
             fwLog(fwlog::kError) << "expression "<< i->expression << " is not valid, skipping\n";
	       ev.push_back(FWExpressionEvaluator("0", item->modelType()->GetName()));
	  }
     }
     //printf("Got evaluators\n");
}

bool FWTableViewTableManager::hasRowHeaders() const
{
   return true;
}
FWTableCellRendererBase* FWTableViewTableManager::rowHeader(int iSortedRowNumber) const
{
   const int realRowNumber = unsortedRowNumber(iSortedRowNumber);
   if (m_view->item() != 0 &&
       m_view->item()->size() &&
       m_view->item()->modelData(realRowNumber) != 0) {
      if (m_view->item()->modelInfo(realRowNumber).displayProperties().isVisible()) {
         if (m_view->m_manager->colorManager().background() == kBlack) {
            m_graphicsContext->
            SetForeground(gVirtualX->GetPixel(kWhite));
         } else {
            m_graphicsContext->
            SetForeground(gVirtualX->GetPixel(kBlack));
         }
         m_rowFillContext->
         SetForeground(gVirtualX->GetPixel(m_view->item()->modelInfo(realRowNumber).
                                           displayProperties().color()));
      } else {
         m_graphicsContext->SetForeground(0x888888);
         m_rowFillContext->SetForeground(m_view->m_manager->colorManager().background());
      }
      
      std::ostringstream s;
      s<<realRowNumber;
      m_rowRenderer->setData(s.str().c_str());
   } else {
      m_rowRenderer->setData("");
   }
   return m_rowRenderer;
}

