#ifndef Fireworks_Core_FWGeoEveOverlap_h
#define Fireworks_Core_FWGeoEveOverlap_h

#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "TString.h"
#include <Rtypes.h>

class FWGeometryTableManagerBase;
class TGLViewer;
class FWEveOverlap : public FWGeoTopNode {
public:
  FWEveOverlap(FWOverlapTableView* v);
  ~FWEveOverlap() override {}

  void Paint(Option_t* option = "") override;
  TString GetHighlightTooltip() override;

  FWGeometryTableManagerBase* tableManager() override;
  FWGeometryTableViewBase* browser() override;
  void popupMenu(int x, int y, TGLViewer* v) override;

private:
  FWOverlapTableView* m_browser;

#ifndef __CINT__
  bool paintChildNodesRecurse(FWGeometryTableManagerBase::Entries_i pIt, Int_t idx, const TGeoHMatrix& mtx);
#endif
  ClassDefOverride(FWEveOverlap, 0);
};

#endif
