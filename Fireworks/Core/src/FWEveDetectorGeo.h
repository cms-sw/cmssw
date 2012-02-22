#ifndef Fireworks_Core_FWGeoEveDetectorGeo_h
#define Fireworks_Core_FWGeoEveDetectorGeo_h

#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "TString.h"
#include <Rtypes.h>

class FWGeometryTableManagerBase;

class FWEveDetectorGeo : public FWGeoTopNode
{
public:

   enum MenuOptions {
      kGeoSetTopNode,
      kGeoSetTopNodeCam,
      kGeoVisOn,
      kGeoVisOff,
      kGeoInspectMaterial,
      kGeoInspectShape,
      kGeoCamera
   };

   FWEveDetectorGeo(FWGeometryTableView* v); 
   virtual ~FWEveDetectorGeo() {}

   virtual void Paint(Option_t* option="");

   virtual TString     GetHighlightTooltip();

   virtual FWGeometryTableManagerBase* tableManager();
   virtual void popupMenu(int x, int y);
private:
#ifndef __CINT__
   void paintChildNodesRecurse(FWGeometryTableManagerBase::Entries_i pIt, Int_t idx,  const TGeoHMatrix& mtx);
#endif
   FWGeometryTableView       *m_browser;
   int m_maxLevel;
   bool m_filterOff;

   ClassDef(FWEveDetectorGeo, 0);
};

#endif
