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
// $Id: FWTriggerTableViewTableManager.h,v 1.3 2011/11/18 02:57:07 amraktad Exp $
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
   virtual ~FWTriggerTableViewTableManager();

   // ---------- const member functions ---------------------
   ///Number of rows in the table
   virtual int numberOfRows() const;
   virtual int numberOfColumns() const;
   virtual std::vector<std::string> getTitles() const;
   virtual int unsortedRowNumber(int iSortedRowNumber) const;

   virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const;
   void dataChanged();   //*SIGNAL*

protected:
   ///Called by 'sort' method to actually handle the sorting of the
   ///rows. Arguments are the same as 'sort'
   virtual void implSort(int iCol, bool iSortOrder);
   std::vector<int> m_sortedToUnsortedIndices;

   const FWTriggerTableView *m_view;
   TGGC *m_graphicsContext;
   FWTextTableCellRenderer *m_renderer;

private:
   FWTriggerTableViewTableManager(const FWTriggerTableViewTableManager&); // stop default     
   const FWTriggerTableViewTableManager& operator=(const FWTriggerTableViewTableManager&); // stop default
};


#endif
