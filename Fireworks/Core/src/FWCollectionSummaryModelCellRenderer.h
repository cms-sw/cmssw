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
//

// system include files

// user include files
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"

// forward declarations
class FWColorBoxIcon;
class FWCheckBoxIcon;
class FWEventItem;

class FWCollectionSummaryModelCellRenderer : public FWTextTableCellRenderer {
public:
  FWCollectionSummaryModelCellRenderer(const TGGC* iContext, const TGGC* iSelectContext);
  ~FWCollectionSummaryModelCellRenderer() override;

  enum ClickHit { kMiss, kHitCheck, kHitColor };
  // ---------- const member functions ---------------------
  UInt_t width() const override;

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight) override;

  void setData(const FWEventItem* iItem, int iIndex);

  ClickHit clickHit(int iX, int iY) const;

private:
  FWCollectionSummaryModelCellRenderer(const FWCollectionSummaryModelCellRenderer&) = delete;  // stop default

  const FWCollectionSummaryModelCellRenderer& operator=(const FWCollectionSummaryModelCellRenderer&) =
      delete;  // stop default

  // ---------- member data --------------------------------
  FWColorBoxIcon* m_colorBox;
  FWCheckBoxIcon* m_checkBox;
  TGGC* m_colorContext;
};

#endif
