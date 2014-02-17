#ifndef Fireworks_TableWidget_FWColumnLabelCellRenderer_h
#define Fireworks_TableWidget_FWColumnLabelCellRenderer_h
// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWColumnLabelCellRenderer
// 
/**\class FWColumnLabelCellRenderer FWColumnLabelCellRenderer.h Fireworks/TableWidget/interface/FWColumnLabelCellRenderer.h

 Description: Cell Renderer which handles the labels at the top of columns

 Usage:
    This renderer will draw both the text of the column's label and if the sort order has been set to kAscendingSort or kDescendingSort
    it will also draw the appropriate symbol denoting the sort order of the column.

*/
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:44:11 EST 2009
// $Id: FWColumnLabelCellRenderer.h,v 1.3 2012/02/22 00:15:44 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/TableWidget/interface/SortOrder.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"

// forward declarations

class FWColumnLabelCellRenderer : public FWTextTableCellRenderer
{

   public:
      FWColumnLabelCellRenderer(const TGGC* iContext=&(getDefaultGC()), bool isSortable = true);
      virtual ~FWColumnLabelCellRenderer();

      // ---------- const member functions ---------------------
      fireworks::table::SortOrder sortOrder() const;

      virtual UInt_t width() const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void setSortOrder(fireworks::table::SortOrder);

      virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight);

   private:
      //FWColumnLabelCellRenderer(const FWColumnLabelCellRenderer&); // stop default

      //const FWColumnLabelCellRenderer& operator=(const FWColumnLabelCellRenderer&); // stop default

      // ---------- member data --------------------------------
      static const UInt_t kGap = 2;
      fireworks::table::SortOrder m_sortOrder;
      int m_sizeOfOrderIcon;
      int m_sizeOfOrderIconStartX;
  
      bool m_isSortable;

};


#endif
