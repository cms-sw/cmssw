#ifndef Fireworks_TableWidget_FWAdapterRowHeaderTableManager_h
#define Fireworks_TableWidget_FWAdapterRowHeaderTableManager_h
// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWAdapterRowHeaderTableManager
// 
/**\class FWAdapterRowHeaderTableManager FWAdapterRowHeaderTableManager.h Fireworks/TableWidget/interface/FWAdapterRowHeaderTableManager.h

 Description: a TableManager used to pass the row header info of another table as the body of this table

 Usage:
   This class is an implementation detail of how the FWTableWidget handles the row headers.  The drawing
   of the row headers is done by the same widget as handles the drawing of the body. The
   FWAdapterRowHeaderTableManager is used to make the row header information appear to be just another table so
   that it works with the above mentioned widget.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:44:59 EST 2009
// $Id: FWAdapterRowHeaderTableManager.h,v 1.1 2009/02/03 20:33:04 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"

// forward declarations

class FWAdapterRowHeaderTableManager : public FWTableManagerBase 
{

   public:
      FWAdapterRowHeaderTableManager(FWTableManagerBase*);
      virtual ~FWAdapterRowHeaderTableManager();

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
      FWAdapterRowHeaderTableManager(const FWAdapterRowHeaderTableManager&); // stop default

      const FWAdapterRowHeaderTableManager& operator=(const FWAdapterRowHeaderTableManager&); // stop default

      // ---------- member data --------------------------------
      const FWTableManagerBase* m_table;

};


#endif
