// -*- C++ -*-
#ifndef Fireworks_Core_FWTableViewTableManager_h
#define Fireworks_Core_FWTableViewTableManager_h
//
// Package:     Core
// Class  :     FWTableViewTableManager
// 
/**\class FWTableViewTableManager FWTableViewTableManager.h Fireworks/Core/interface/FWTableViewTableManager.h

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
#include "Fireworks/Core/interface/FWTableViewManager.h"
#include "Fireworks/Core/interface/FWExpressionEvaluator.h"

// forward declarations
class FWTableView;
class FWFramedTextTableCellRenderer;

class FWTableViewTableManager : public FWTableManagerBase {
     friend class FWTableView;
public:
     FWTableViewTableManager(const FWTableView *);
     ~FWTableViewTableManager() override;

     // ---------- const member functions ---------------------
     ///Number of rows in the table
      int numberOfRows() const override;
     ///Number of columns in the table
      int numberOfColumns() const override;
     
     ///returns the title names for each column
     std::vector<std::string> getTitles() const override;
     
     ///when passed the index to the sorted order of the rows it
     ///returns the original row number from the underlying data
     int unsortedRowNumber(int iSortedRowNumber) const override;
     
     /** Returns the particular renderer used to handle the requested
	 cell.  Arguments: iSortedRowNumber: the row number from the
	 present sort (i.e. the cell number of the view) iCol: the column
	 number of the cell.  The returned value must be used immediately
	 and not held onto since the same Renderer can be used for
	 subsequent calls
     */
     FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const override;
     
     ///require all cells to be the same height
     // virtual unsigned int cellHeight() const;
     
     ///for each column in the table this returns the present maximum width for that column
     // virtual std::vector<unsigned int> maxWidthForColumns() const;

     ///Returns 'true' if this table has row headers. Defaults return value is false.
     bool hasRowHeaders() const override ;
     ///Returns the renderer for the row header for the sorted row number iSortedRowNumber
     FWTableCellRendererBase* rowHeader(int iSortedRowNumber) const override ;

     ///Called if mouse button pressed in Row Header, defaults is to do nothing
     //virtual void buttonPressedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY);
     //virtual void buttonReleasedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY);

     // ---------- static member functions --------------------
     
     // ---------- member functions ---------------------------
     ///Call to have table sorted on values in column iCol with the
     ///sort order being descending if iSortOrder is 'true'
     // void sort(int iCol, bool iSortOrder);

     ///Classes which inherit from FWTableViewTableManager must call
     ///this when their underlying data changes
     void dataChanged(); //*SIGNAL*
      
     ///Classes which inherit from FWTableViewTableManager must call
     ///this when how the data is shown (e.g. color) changes
     // void visualPropertiesChanged(); //*SIGNAL*
     
     // ClassDef(FWTableViewTableManager,0);
     void updateEvaluators ();

protected:
     ///Called by 'sort' method to actually handle the sorting of the
     ///rows. Arguments are the same as 'sort'
     void implSort(int iCol, bool iSortOrder) override;
     std::vector<int> m_sortedToUnsortedIndices;

     const FWTableView *m_view;
     TGGC *m_graphicsContext;
     TGGC *m_highlightContext;
     FWTextTableCellRenderer *m_renderer;

     TGGC *m_rowContext;
     TGGC *m_rowFillContext;
     FWFramedTextTableCellRenderer *m_rowRenderer;
   
     std::vector<FWExpressionEvaluator> m_evaluators;
     std::vector<FWTableViewManager::TableEntry> *m_tableFormats;
     
     // ---------- member data --------------------------------
     // int m_sortColumn;
     // bool m_sortOrder;
     mutable bool m_caughtExceptionInCellRender;

private:
     FWTableViewTableManager(const FWTableViewTableManager&) = delete; // stop default     
     const FWTableViewTableManager& operator=(const FWTableViewTableManager&) = delete; // stop default
};


#endif
