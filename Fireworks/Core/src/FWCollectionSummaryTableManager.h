#ifndef Fireworks_Core_FWCollectionSummaryTableManager_h
#define Fireworks_Core_FWCollectionSummaryTableManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCollectionSummaryTableManager
// 
/**\class FWCollectionSummaryTableManager FWCollectionSummaryTableManager.h Fireworks/Core/interface/FWCollectionSummaryTableManager.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun Feb 22 10:13:37 CST 2009
// $Id: FWCollectionSummaryTableManager.h,v 1.2 2011/08/20 03:48:40 amraktad Exp $
//

// system include files
#include <vector>
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/Core/src/FWCollectionSummaryModelCellRenderer.h"

// forward declarations
class FWEventItem;
//class FWItemValueGetter;
class FWCollectionSummaryWidget;

class FWCollectionSummaryTableManager : public FWTableManagerBase {
   
public:
   FWCollectionSummaryTableManager(FWEventItem* iCollection, const TGGC* iContext, const TGGC* iHighlightContext, FWCollectionSummaryWidget*);
   virtual ~FWCollectionSummaryTableManager();
   
   // ---------- const member functions ---------------------
   virtual  int numberOfRows() const ;
   virtual  int numberOfColumns() const ;
   virtual std::vector<std::string> getTitles() const;
   virtual int unsortedRowNumber(int iSortedRowNumber) const;

   virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const;

   virtual bool hasRowHeaders() const ;
   virtual FWTableCellRendererBase* rowHeader(int iSortedRowNumber) const ;
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   virtual void buttonReleasedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY);

protected:
   virtual void implSort(int iCol, bool iSortOrder);
private:
   FWCollectionSummaryTableManager(const FWCollectionSummaryTableManager&); // stop default
   
   const FWCollectionSummaryTableManager& operator=(const FWCollectionSummaryTableManager&); // stop default
   
   void dataChanged();
   // ---------- member data --------------------------------
   FWEventItem* m_collection;
   std::vector<int> m_sortedToUnsortedIndicies;
   
   mutable FWCollectionSummaryModelCellRenderer m_renderer;
   mutable FWTextTableCellRenderer m_bodyRenderer;
   FWCollectionSummaryWidget* m_widget;
};


#endif
