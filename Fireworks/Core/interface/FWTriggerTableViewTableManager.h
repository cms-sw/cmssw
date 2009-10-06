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
// $Id: FWTriggerTableViewTableManager.h,v 1.4 2009/08/26 22:21:45 dmytro Exp $
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
   ///Number of columns in the table
   virtual int numberOfColumns() const;

   ///returns the title names for each column
   virtual std::vector<std::string> getTitles() const;

   ///when passed the index to the sorted order of the rows it
   ///returns the original row number from the underlying data
   virtual int unsortedRowNumber(int iSortedRowNumber) const;

   /** Returns the particular renderer used to handle the requested
       cell.  Arguments: iSortedRowNumber: the row number from the
       present sort (i.e. the cell number of the view) iCol: the column
       number of the cell.  The returned value must be used immediately
       and not held onto since the same Renderer can be used for
       subsequent calls
    */
   virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const;

   ///require all cells to be the same height
   // virtual unsigned int cellHeight() const;

   ///for each column in the table this returns the present maximum width for that column
   // virtual std::vector<unsigned int> maxWidthForColumns() const;

   ///Returns 'true' if this table has row headers. Defaults return value is false.
   // virtual bool hasRowHeaders() const ;
   ///Returns the renderer for the row header for the sorted row number iSortedRowNumber
   // virtual FWTableCellRendererBase* rowHeader(int iSortedRowNumber) const ;

   ///Called if mouse button pressed in Row Header, defaults is to do nothing
   //virtual void buttonPressedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY);
   //virtual void buttonReleasedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY);

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   ///Call to have table sorted on values in column iCol with the
   ///sort order being descending if iSortOrder is 'true'
   // void sort(int iCol, bool iSortOrder);

   ///Classes which inherit from FWTriggerTableViewTableManager must call
   ///this when their underlying data changes
   void dataChanged();   //*SIGNAL*

   ///Classes which inherit from FWTriggerTableViewTableManager must call
   ///this when how the data is shown (e.g. color) changes
   // void visualPropertiesChanged(); //*SIGNAL*

   // ClassDef(FWTriggerTableViewTableManager,0);
   void updateEvaluators ();

protected:
   ///Called by 'sort' method to actually handle the sorting of the
   ///rows. Arguments are the same as 'sort'
   virtual void implSort(int iCol, bool iSortOrder);
   std::vector<int> m_sortedToUnsortedIndices;

   const FWTriggerTableView *m_view;
   TGGC *m_graphicsContext;
   TGGC *m_highlightContext;
   FWTextTableCellRenderer *m_renderer;
};


#endif
