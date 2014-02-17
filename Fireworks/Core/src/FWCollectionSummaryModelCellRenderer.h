#ifndef Fireworks_Core_FWCollectionSummaryModelCellRenderer_h
#define Fireworks_Core_FWCollectionSummaryModelCellRenderer_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCollectionSummaryModelCellRenderer
// 
/**\class FWCollectionSummaryModelCellRenderer FWCollectionSummaryModelCellRenderer.h Fireworks/Core/interface/FWCollectionSummaryModelCellRenderer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Feb 25 10:03:21 CST 2009
// $Id: FWCollectionSummaryModelCellRenderer.h,v 1.1 2009/03/04 16:40:51 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"

// forward declarations
class FWColorBoxIcon;
class FWCheckBoxIcon;
class FWEventItem;

class FWCollectionSummaryModelCellRenderer : public FWTextTableCellRenderer
{
public:
   FWCollectionSummaryModelCellRenderer(const TGGC* iContext, const TGGC* iSelectContext);
   virtual ~FWCollectionSummaryModelCellRenderer();
   
   enum ClickHit {
      kMiss,
      kHitCheck,
      kHitColor
   };
   // ---------- const member functions ---------------------
   virtual UInt_t width() const;
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   virtual void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight);
   
   void setData(const FWEventItem* iItem, int iIndex);
   
   ClickHit clickHit(int iX, int iY) const;
   
private:
   FWCollectionSummaryModelCellRenderer(const FWCollectionSummaryModelCellRenderer&); // stop default
   
   const FWCollectionSummaryModelCellRenderer& operator=(const FWCollectionSummaryModelCellRenderer&); // stop default
   
   // ---------- member data --------------------------------
   FWColorBoxIcon* m_colorBox;
   FWCheckBoxIcon* m_checkBox;
   TGGC* m_colorContext;
};


#endif
