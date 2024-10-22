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
//

// system include files

// user include files
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"

// forward declarations
class FWColumnLabelCellRenderer;

class FWAdapterHeaderTableManager : public FWTableManagerBase {
public:
  FWAdapterHeaderTableManager(const FWTableManagerBase*);
  ~FWAdapterHeaderTableManager() override;

  // ---------- const member functions ---------------------
  int numberOfRows() const override;
  int numberOfColumns() const override;
  std::vector<std::string> getTitles() const override;
  FWTableCellRendererBase* cellRenderer(int iRow, int iCol) const override;
  int unsortedRowNumber(int) const override;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void implSort(int col, bool sortOrder) override;

  FWAdapterHeaderTableManager(const FWAdapterHeaderTableManager&) = delete;  // stop default

  const FWAdapterHeaderTableManager& operator=(const FWAdapterHeaderTableManager&) = delete;  // stop default

private:
  // ---------- member data --------------------------------
  const FWTableManagerBase* m_table;
  FWColumnLabelCellRenderer* m_renderer;
  int m_sortedColumn;
  bool m_descendingSort;
};

#endif
