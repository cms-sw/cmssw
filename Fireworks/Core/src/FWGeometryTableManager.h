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
// $Id: FWGeometryTableManager.h,v 1.10 2012/05/10 23:57:52 amraktad Exp $
//

#include "Fireworks/Core/interface/FWGeometryTableManagerBase.h"
#include <string>
#include <boost/tr1/unordered_map.hpp>

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

      bool accepted() { return m_matches || m_childMatches; }
   };


   typedef boost::unordered_map<TGeoVolume*, Match>  Volumes_t;
   typedef Volumes_t::iterator               Volumes_i; 

   FWGeometryTableManager(FWGeometryTableView*);
   virtual ~FWGeometryTableManager();

   virtual void recalculateVisibility();
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

   virtual void setDaughtersSelfVisibility(int i, bool v);
   virtual void setVisibility(NodeInfo& nodeInfo, bool );
   virtual void setVisibilityChld(NodeInfo& nodeInfo, bool);

   virtual bool getVisibilityChld(const NodeInfo& nodeInfo) const;
   virtual bool getVisibility (const NodeInfo& nodeInfo) const;

   void assertNodeFilterCache(NodeInfo& data);
 
   virtual int numberOfColumns() const { return kNumColumn; }
   virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const;
   
   void checkRegionOfInterest(double* center, double radius, long algo);
   void resetRegionOfInterest();
   
protected:
   virtual bool nodeIsParent(const NodeInfo&) const;
   //   virtual FWGeometryTableManagerBase::ESelectionState nodeSelectionState(int) const;
   virtual const char* cellName(const NodeInfo& data) const;

private:
   FWGeometryTableManager(const FWGeometryTableManager&); // stop default
   const FWGeometryTableManager& operator=(const FWGeometryTableManager&); // stop default


   FWGeometryTableView *m_browser;

   mutable Volumes_t    m_volumes;

   bool                 m_filterOff; //cached
};

#endif
