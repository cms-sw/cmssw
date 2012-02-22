#ifndef Fireworks_Core_FWGeometryTableManager_h
#define Fireworks_Core_FWGeometryTableManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableManager
// 
/**\class FWGeometryTableManager FWGeometryTableManager.h Fireworks/Core/interface/FWGeometryTableManager.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Jan  4 20:34:22 CET 2012
// $Id: FWGeometryTableManager.h,v 1.1.2.3 2012/02/16 04:50:21 amraktad Exp $
//

#include "Fireworks/Core/interface/FWGeometryTableManagerBase.h"

class FWGeometryTableViewBase;
class FWGeometryTableView;
#include <string>
class FWGeometryTableManager : public FWGeometryTableManagerBase
{
public:
   enum GeometryBits
   {
      kMatches         =  BIT(5),
      kChildMatches    =  BIT(6),
      kFilterCached    =  BIT(7)
   };

   enum   ECol   { kNameColumn, kColorColumn,  kVisSelfColumn, kVisChildColumn, kMaterialColumn, kNumColumn };

   FWGeometryTableManager(FWGeometryTableView*);
   virtual ~FWGeometryTableManager();

   virtual void recalculateVisibility();
   void recalculateVisibilityNodeRec(int);
   void recalculateVisibilityVolumeRec(int);
   // geo 
   void  loadGeometry( TGeoNode* iGeoTopNode, TObjArray* iVolumes);
   void checkChildMatches(TGeoVolume* v,  std::vector<TGeoVolume*>&);
   void importChildren(int parent_idx);
   void checkHierarchy();

   // signal callbacks
   void updateFilter();
   void printMaterials();

   virtual void setVisibility(NodeInfo& nodeInfo, bool );
   virtual void setVisibilityChld(NodeInfo& nodeInfo, bool);

   virtual bool getVisibilityChld(const NodeInfo& nodeInfo) const;
   virtual bool getVisibility (const NodeInfo& nodeInfo) const;

   void assertNodeFilterCache(NodeInfo& data);
 
   virtual int numberOfColumns() const { return kNumColumn; }
   virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const;

protected:
   virtual bool nodeIsParent(const NodeInfo&) const;
   //   virtual FWGeometryTableManagerBase::ESelectionState nodeSelectionState(int) const;
   virtual const char* cellName(const NodeInfo& data) const;

private:
   FWGeometryTableManager(const FWGeometryTableManager&); // stop default
   const FWGeometryTableManager& operator=(const FWGeometryTableManager&); // stop default


   FWGeometryTableView* m_browser;

   mutable Volumes_t  m_volumes;

   bool               m_filterOff; //cached

};


#endif
