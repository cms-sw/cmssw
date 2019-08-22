#ifndef Fireworks_Core_FWGeoEveDetectorGeo_h
#define Fireworks_Core_FWGeoEveDetectorGeo_h

#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "TString.h"
#include <Rtypes.h>

class FWGeometryTableManagerBase;
class TGLViewer;
class FWEveDetectorGeo : public FWGeoTopNode {
public:
  FWEveDetectorGeo(FWGeometryTableView* v);
  ~FWEveDetectorGeo() override {}

  void Paint(Option_t* option = "") override;

  TString GetHighlightTooltip() override;

  FWGeometryTableManagerBase* tableManager() override;
  FWGeometryTableViewBase* browser() override;

  void popupMenu(int x, int y, TGLViewer*) override;

#ifndef __CINT__
  // virtual void paintShape(bool visLevel, FWGeometryTableManagerBase::NodeInfo& data,  Int_t tableIndex, const TGeoHMatrix& nm, bool volumeColor);
#endif

protected:
private:
#ifndef __CINT__
  bool paintChildNodesRecurse(FWGeometryTableManagerBase::Entries_i pIt, Int_t idx, const TGeoHMatrix& mtx);
#endif
  FWGeometryTableView* m_browser;
  int m_maxLevel;
  bool m_filterOff;

  ClassDefOverride(FWEveDetectorGeo, 0);
};

#endif
