#ifndef Fireworks_Core_FWGeoEveOverlap_h
#define Fireworks_Core_FWGeoEveOverlap_h

#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "TString.h"
#include <Rtypes.h>

class FWGeometryTableManagerBase;
class TGLViewer;
class FWEveOverlap : public FWGeoTopNode
{
public:
   FWEveOverlap(FWOverlapTableView* v);
   virtual ~FWEveOverlap(){}

   virtual void Paint(Option_t* option="");
   virtual TString     GetHighlightTooltip();

   virtual FWGeometryTableManagerBase* tableManager();
   virtual void popupMenu(int x, int y, TGLViewer* v);
private:
   FWOverlapTableView       *m_browser;

#ifndef __CINT__
   void paintChildNodesRecurse(FWGeometryTableManagerBase::Entries_i pIt, Int_t idx,  const TGeoHMatrix& mtx);
#endif
   ClassDef(FWEveOverlap, 0);
};

#endif
