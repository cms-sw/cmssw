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
//

#include "Fireworks/Core/interface/FWGeometryTableManagerBase.h"
#include <string>
#include <unordered_map>

class FWGeometryTableViewBase;
class FWGeometryTableView;

#include "TGeoVolume.h"


class FWGeometryTableManager : public FWGeometryTableManagerBase
{
public:
   enum ECol   { kNameColumn, kColorColumn, kTranspColumn, kVisSelfColumn, kVisChildColumn, kMaterialColumn, kNumColumn };

   enum GeometryBits
   {
      kMatches         =  BIT(5),
      kChildMatches    =  BIT(6),
      kFilterCached    =  BIT(7)
   };

   struct Match
   {
      bool m_matches;
      bool m_childMatches;
      Match() : m_matches(false), m_childMatches(false) {}

      bool accepted() const { return m_matches || m_childMatches; }
   };


   typedef std::unordered_map<TGeoVolume*, Match>  Volumes_t;
   typedef Volumes_t::iterator               Volumes_i; 

   FWGeometryTableManager(FWGeometryTableView*);
   ~FWGeometryTableManager() override;

   void recalculateVisibility() override;
   void recalculateVisibilityNodeRec(int);
   void recalculateVisibilityVolumeRec(int);
   // geo 
   void loadGeometry( TGeoNode* iGeoTopNode, TObjArray* iVolumes);
   void checkChildMatches(TGeoVolume* v,  std::vector<TGeoVolume*>&);
   void importChildren(int parent_idx);
   void checkHierarchy();

   // signal callbacks
   void updateFilter(int);
   void printMaterials();

   void setDaughtersSelfVisibility(int i, bool v) override;
   void setVisibility(NodeInfo& nodeInfo, bool ) override;
   void setVisibilityChld(NodeInfo& nodeInfo, bool) override;

   bool getVisibilityChld(const NodeInfo& nodeInfo) const override;
   bool getVisibility (const NodeInfo& nodeInfo) const override;

   void assertNodeFilterCache(NodeInfo& data);
 
   int numberOfColumns() const override { return kNumColumn; }
   FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const override;
   
   void checkRegionOfInterest(double* center, double radius, long algo);
   void resetRegionOfInterest();
   
protected:
   bool nodeIsParent(const NodeInfo&) const override;
   //   virtual FWGeometryTableManagerBase::ESelectionState nodeSelectionState(int) const;
   const char* cellName(const NodeInfo& data) const override;

private:
   FWGeometryTableManager(const FWGeometryTableManager&) = delete; // stop default
   const FWGeometryTableManager& operator=(const FWGeometryTableManager&) = delete; // stop default


   FWGeometryTableView *m_browser;

   mutable Volumes_t    m_volumes;

   bool                 m_filterOff; //cached
};

#endif
