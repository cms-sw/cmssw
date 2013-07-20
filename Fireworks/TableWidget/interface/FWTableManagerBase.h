#ifndef Fireworks_TableWidget_FWTableManagerBase_h
#define Fireworks_TableWidget_FWTableManagerBase_h
// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWTableManagerBase
// 
/**\class FWTableManagerBase FWTableManagerBase.h Fireworks/TableWidget/interface/FWTableManagerBase.h

 Description: Base class for classes that work as interfaces that translate underlying data into a table form

 Usage:
    Classes which inherit from FWTableManagerBase are used as adapters to allow external data to be shown in tabular form
    via the FWTableWidget.  The table is made of three parts
    1) The column headers: Each column is described by a 'title' and the title is drawn in the column header
    2) The body: the actual data of the table laid out in rows and columns
    3) the row headers: optional identifier for a row. If given, the row header will always be visible on the screen if any part
        of the row is visible
        
    The FWTableWidget actually draws the cells in the table by asking the FWTableManagerBase for a FWTableCellRendererBase for
    a particular cell.  The renderer will then be asked to draw the cell into the appropriate part of the graphics window.  Therfore
    it is the FWTableManagerBase's responsibility to create FWTableCellRendererBases which are appropriate for the data to be
    shown in each cell of the table.  See the documentation of FWTableCellRendererBase for further information.
    
    FWTableManagerBase must also be able to sort the rows of data based on the values in a specified column.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:40:52 EST 2009
// $Id: FWTableManagerBase.h,v 1.8 2012/02/22 00:15:44 amraktad Exp $
//

// system include files
#include <vector>
#include <string>
#include "TQObject.h"
#include "GuiTypes.h"

// user include files

// forward declarations
class FWTableCellRendererBase;

class FWTableManagerBase : public TQObject 
{

   public:
      FWTableManagerBase();
      virtual ~FWTableManagerBase();

      // ---------- const member functions ---------------------
      ///Number of rows in the table
      virtual  int numberOfRows() const = 0;
      ///Number of columns in the table
      virtual  int numberOfColumns() const = 0;
      
      ///returns the title names for each column
      virtual std::vector<std::string> getTitles() const = 0;
      
      ///when passed the index to the sorted order of the rows it returns the original row number from the underlying data
      virtual int unsortedRowNumber(int iSortedRowNumber) const = 0;
      
      /** Returns the particular renderer used to handle the requested cell.  Arguments:
      iSortedRowNumber: the row number from the present sort (i.e. the cell number of the view)
      iCol: the column number of the cell.
      The returned value must be used immediately and not held onto since the same Renderer can be used for subsequent calls
      */
      virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const =0;
      
      ///require all cells to be the same height
      virtual unsigned int cellHeight() const;
      
      ///for each column in the table this returns the present maximum width for that column
      virtual std::vector<unsigned int> maxWidthForColumns() const;

       virtual bool hasLabelHeaders() const ;
    
      ///Returns 'true' if this table has row headers. Defaults return value is false.
      virtual bool hasRowHeaders() const ;
      ///Returns the renderer for the row header for the sorted row number iSortedRowNumber
      virtual FWTableCellRendererBase* rowHeader(int iSortedRowNumber) const ;
  
      virtual bool cellDataIsSortable() const { return true ; } 
  

      ///Called if mouse button pressed in Row Header, defaults is to do nothing
      virtual void buttonPressedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY);
      virtual void buttonReleasedInRowHeader(Int_t row, Event_t* event, Int_t relX, Int_t relY);

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      ///Call to have table sorted on values in column iCol with the sort order being descending if iSortOrder is 'true'
      void sort(int iCol, bool iSortOrder);

      ///Classes which inherit from FWTableManagerBase must call this when their underlying data changes
      void dataChanged(); //*SIGNAL*
      
      ///Classes which inherit from FWTableManagerBase must call this when how the data is shown (e.g. color) changes
      void visualPropertiesChanged(); //*SIGNAL*
      
      ClassDef(FWTableManagerBase,0);

      /// The current sort order for the table.
      bool sortOrder(void) { return m_sortOrder; }
      
      /// The current sort column
      int sortColumn(void) { return m_sortColumn; }

   protected:
      ///Called by 'sort' method to actually handle the sorting of the rows. Arguments are the same as 'sort'
      virtual void implSort(int iCol, bool iSortOrder) = 0;
      
   private:
      //FWTableManagerBase(const FWTableManagerBase&); // stop default

      //const FWTableManagerBase& operator=(const FWTableManagerBase&); // stop default

      // ---------- member data --------------------------------
      int  m_sortColumn;
      bool m_sortOrder;
};


#endif
