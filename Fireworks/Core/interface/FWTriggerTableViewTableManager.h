// -*- C++ -*-
#ifndef Fireworks_Core_FWTriggerTableViewTableManager_h
#define Fireworks_Core_FWTriggerTableViewTableManager_h
//
// Package:     Core
// Class  :     FWTriggerTableViewTableManager
//
/**\class FWTriggerTableViewTableManager FWTriggerTableViewTableManager.h Fireworks/Core/interface/FWTriggerTableViewTableManager.h

 */
//
// Original Author:
//         Created:  Mon Feb  2 16:40:52 EST 2009
//

// system include files
#include <vector>
#include <string>
#include "TQObject.h"
#include "GuiTypes.h"

// user include files
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/Core/interface/FWTriggerTableViewManager.h"
#include "Fireworks/Core/interface/FWExpressionEvaluator.h"

// forward declarations
class FWTriggerTableView;

class FWTriggerTableViewTableManager : public FWTableManagerBase {
   friend class FWTriggerTableView;
public:
   FWTriggerTableViewTableManager(const FWTriggerTableView *);
   ~FWTriggerTableViewTableManager() override;

   // ---------- const member functions ---------------------
   ///Number of rows in the table
   int numberOfRows() const override;
   int numberOfColumns() const override;
   std::vector<std::string> getTitles() const override;
   int unsortedRowNumber(int iSortedRowNumber) const override;

   FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const override;
   void dataChanged();   //*SIGNAL*

protected:
   ///Called by 'sort' method to actually handle the sorting of the
   ///rows. Arguments are the same as 'sort'
   void implSort(int iCol, bool iSortOrder) override;
   std::vector<int> m_sortedToUnsortedIndices;

   const FWTriggerTableView *m_view;
   TGGC *m_graphicsContext;
   FWTextTableCellRenderer *m_renderer;

private:
   FWTriggerTableViewTableManager(const FWTriggerTableViewTableManager&) = delete; // stop default     
   const FWTriggerTableViewTableManager& operator=(const FWTriggerTableViewTableManager&) = delete; // stop default
};


#endif
