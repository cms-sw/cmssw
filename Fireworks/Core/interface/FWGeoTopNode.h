#ifndef Fireworks_Core_FWGeoTopNode_h
#define Fireworks_Core_FWGeoTopNode_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeoTopNode
// 
/**\class FWGeoTopNode FWGeoTopNode.h Fireworks/Core/interface/FWGeoTopNode.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Matevz Tadel, Alja Mrak Tadel
//         Created:  Thu Jun 23 01:25:00 CEST 2011
// $Id: FWGeoTopNode.h,v 1.5 2011/07/01 23:33:53 amraktad Exp $
//

#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "TEveElement.h"

class TGeoHMatrix;

class FWGeometryTableManager;
class FWGeometryBrowser;
class TBuffer3D;
class TGeoNode;


class FWGeoTopNode : public TEveElementList
{
public:
   FWGeoTopNode(FWGeometryBrowser*);
   virtual ~FWGeoTopNode();
   virtual void Paint(Option_t* option="");

private:
   FWGeoTopNode(const FWGeoTopNode&); // stop default
   const FWGeoTopNode& operator=(const FWGeoTopNode&); // stop default


   void setupBuffMtx(TBuffer3D& buff, const TGeoHMatrix& mat);

   void paintChildNodesRecurse(FWGeometryTableManager::Entries_i pIt, TGeoHMatrix& mtx);
   void  paintShape(FWGeometryTableManager::NodeInfo& nodeInfo, const TGeoHMatrix& nm);
   FWGeometryBrowser       *m_geoBrowser;

   // cached
   FWGeometryTableManager::Entries_v* m_entries;
   int m_maxLevel;
   bool m_filterOff;
};


#endif
