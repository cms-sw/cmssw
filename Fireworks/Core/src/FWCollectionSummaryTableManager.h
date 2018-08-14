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
//

// system include files
#include <vector>
#include <memory>

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
   ~FWCollectionSummaryTableManager() override;
   
   // ---------- const member functions ---------------------
    int numberOfRows() const override ;
    int numberOfColumns() const override ;
   std::vector<std::string> getTitles() const override;
   int unsortedRowNumber(int iSortedRowNumber) const override;

   FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const override;

   bool hasRowHeaders() const override ;
   FWTableCellRendererBase* rowHeader(int iSortedRowNumber) const override ;
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void buttonReleasedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY) override;

protected:
   void implSort(int iCol, bool iSortOrder) override;
private:
   FWCollectionSummaryTableManager(const FWCollectionSummaryTableManager&) = delete; // stop default
   
   const FWCollectionSummaryTableManager& operator=(const FWCollectionSummaryTableManager&) = delete; // stop default
   
   void dataChanged();
   // ---------- member data --------------------------------
   FWEventItem* m_collection;
   std::vector<int> m_sortedToUnsortedIndicies;
   
   mutable FWCollectionSummaryModelCellRenderer m_renderer;
   mutable FWTextTableCellRenderer m_bodyRenderer;
   FWCollectionSummaryWidget* m_widget;
};


#endif
