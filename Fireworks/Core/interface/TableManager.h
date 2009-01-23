// -*- C++ -*-
#ifndef Fireworks_Core_TableManager_h
#define Fireworks_Core_TableManager_h
//
// Package:     Core
// Class  :     TableManager
//
/**\class TableManager TableManager.h Fireworks/Core/interface/TableManager.h

   Description:  Abstract class interface which is used by TableWidget to access data for a table.

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Sun Jul  6 23:28:39 EDT 2008
// $Id: TableManager.h,v 1.4 2008/11/06 22:05:23 amraktad Exp $
//

// system include files
#include <vector>
#include <string>

// user include files

// forward declarations
class TGFrame;

class TableManager {
public:
   TableManager(void) {
   }
   virtual ~TableManager() {
   }
   //      virtual const std::string GetTitle(int col) = 0;
   virtual int NumberOfRows() const = 0;
   virtual int NumberOfCols() const = 0;
   virtual void Sort(int col, bool sortOrder) = 0; // sortOrder=true means desc order
   virtual std::vector<std::string> GetTitles(int col) = 0;
   virtual void FillCells(int rowStart, int colStart,
                          int rowEnd, int colEnd, std::vector<std::string>& oToFill) = 0;
   virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame) = 0;
   virtual void UpdateRowCell(int row, TGFrame *rowCell) = 0;
   virtual const std::string title() const {
      return "";
   }
   virtual void Selection (int row, int mask) {
   }
   virtual void selectRows () {
   }
   virtual int table_row_to_index (int) const {
      return 0;
   }
   virtual int index_to_table_row (int) const {
      return 0;
   }
};


#endif
