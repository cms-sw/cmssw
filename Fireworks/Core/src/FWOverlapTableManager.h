#ifndef Fireworks_Core_FWOverlapTableManager_h
#define Fireworks_Core_FWOverlapTableManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWOverlapTableManager
// 
/**\class FWOverlapTableManager FWOverlapTableManager.h Fireworks/Core/interface/FWOverlapTableManager.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Wed Jan  4 20:34:38 CET 2012
// $Id: FWOverlapTableManager.h,v 1.2 2012/02/22 03:45:59 amraktad Exp $
//

#include "Fireworks/Core/interface/FWGeometryTableManagerBase.h"
#include "TGeoMatrix.h"
#include "TGeoNode.h"
#include "TGeoOverlap.h"
#include <map>


class FWOverlapTableView;
class TGeoOverlap;
class TGeoIterator;

class FWOverlapTableManager : public FWGeometryTableManagerBase
{
public:
   enum OverlapBits
   {
      kVisMarker    =  BIT(5),
      kOverlap      =  BIT(6),
      kOverlapChild      =  BIT(7)
   };
   FWOverlapTableManager(FWOverlapTableView*);
   virtual ~FWOverlapTableManager();

  virtual void recalculateVisibility();
  virtual void recalculateVisibilityNodeRec(int);
  void importOverlaps(std::string path, double precision);
   virtual int numberOfColumns() const {return 6;}

   virtual std::vector<std::string> getTitles() const;
 
  FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const;
  
  void getOverlapTitles(int,TString&) const;
  void printOverlaps(int) const;

   void setDaughtersSelfVisibility(int i, bool v);
protected:
   virtual bool nodeIsParent(const NodeInfo&) const;
   //   virtual  const char* cellName(const NodeInfo& data) const;

private:
   FWOverlapTableManager(const FWOverlapTableManager&); // stop default
   const FWOverlapTableManager& operator=(const FWOverlapTableManager&); // stop default

   void addOverlapEntry(TGeoOverlap*, int, int , TGeoHMatrix*);
   FWOverlapTableView* m_browser;
  
  std::multimap<int, int> m_mapNodeOverlaps;
};


#endif
