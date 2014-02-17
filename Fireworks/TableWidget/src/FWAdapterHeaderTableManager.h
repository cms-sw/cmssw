#ifndef Fireworks_TableWidget_FWAdapterHeaderTableManager_h
#define Fireworks_TableWidget_FWAdapterHeaderTableManager_h
// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWAdapterHeaderTableManager
// 
/**\class FWAdapterHeaderTableManager FWAdapterHeaderTableManager.h Fireworks/TableWidget/interface/FWAdapterHeaderTableManager.h

 Description: a TableManager used to pass the header info of another table as the body of this table

 Usage:
    This class is an implementation detail of how the FWTableWidget handles the column headers.  The drawing
    of the column headers is done by the same widget as handles the drawing of the body. The
    FWAdapterHeaderTableManager is used to make the header information appear to be just another table so
    that it works with the above mentioned widget.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:44:43 EST 2009
// $Id: FWAdapterHeaderTableManager.h,v 1.1 2009/02/03 20:33:03 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"

// forward declarations
class FWColumnLabelCellRenderer;

class FWAdapterHeaderTableManager : public FWTableManagerBase 
{

   public:
      FWAdapterHeaderTableManager(const FWTableManagerBase*);
      virtual ~FWAdapterHeaderTableManager();

      // ---------- const member functions ---------------------
      virtual  int numberOfRows() const ;
      virtual  int numberOfColumns() const ;
      virtual std::vector<std::string> getTitles() const;
      virtual FWTableCellRendererBase* cellRenderer(int iRow, int iCol) const;
      int unsortedRowNumber(int) const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void implSort(int col, bool sortOrder) ;

   private:
      FWAdapterHeaderTableManager(const FWAdapterHeaderTableManager&); // stop default

      const FWAdapterHeaderTableManager& operator=(const FWAdapterHeaderTableManager&); // stop default

      // ---------- member data --------------------------------
      const FWTableManagerBase* m_table;
      FWColumnLabelCellRenderer* m_renderer;
      int m_sortedColumn;
      bool m_descendingSort;

};


#endif
