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
//

// system include files

// user include files
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"

// forward declarations

class FWAdapterRowHeaderTableManager : public FWTableManagerBase {
public:
  FWAdapterRowHeaderTableManager(FWTableManagerBase*);
  ~FWAdapterRowHeaderTableManager() override;

  // ---------- const member functions ---------------------
  int numberOfRows() const override;
  int numberOfColumns() const override;
  std::vector<std::string> getTitles() const override;
  FWTableCellRendererBase* cellRenderer(int iRow, int iCol) const override;
  int unsortedRowNumber(int) const override;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void implSort(int col, bool sortOrder) override;

  FWAdapterRowHeaderTableManager(const FWAdapterRowHeaderTableManager&) = delete;  // stop default

  const FWAdapterRowHeaderTableManager& operator=(const FWAdapterRowHeaderTableManager&) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  const FWTableManagerBase* m_table;
};

#endif
