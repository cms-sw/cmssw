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
//

#include "Fireworks/Core/interface/FWGeometryTableManagerBase.h"
#include "TGeoMatrix.h"
#include "TGeoNode.h"
#include "TGeoOverlap.h"
#include <map>

class FWOverlapTableView;
class TGeoOverlap;
class TGeoIterator;

class FWOverlapTableManager : public FWGeometryTableManagerBase {
public:
  enum OverlapBits { kVisMarker = BIT(5), kOverlap = BIT(6), kOverlapChild = BIT(7) };

  class QuadId : public TNamed {
  public:
    QuadId() : m_ovl(nullptr), m_parentIdx(-1) {}
    QuadId(TGeoOverlap* ovl, int idx) {
      m_ovl = ovl;
      m_parentIdx = idx;
    }

    ~QuadId() override {}
    const char* GetName() const override { return m_ovl->GetTitle(); }
    const char* GetTitle() const override { return m_ovl->GetTitle(); }

    TGeoOverlap* m_ovl;
    int m_parentIdx;
    std::vector<int> m_nodes;
  };

  FWOverlapTableManager(FWOverlapTableView*);
  ~FWOverlapTableManager() override;

  void recalculateVisibility() override;
  virtual void recalculateVisibilityNodeRec(int);
  void importOverlaps(std::string path, double precision);
  int numberOfColumns() const override { return 6; }

  std::vector<std::string> getTitles() const override;

  FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const override;

  void getOverlapTitles(int, TString&) const;
  void printOverlaps(int) const;

  void setDaughtersSelfVisibility(int i, bool v) override;

protected:
  bool nodeIsParent(const NodeInfo&) const override;
  //   virtual  const char* cellName(const NodeInfo& data) const;

private:
  FWOverlapTableManager(const FWOverlapTableManager&) = delete;                   // stop default
  const FWOverlapTableManager& operator=(const FWOverlapTableManager&) = delete;  // stop default

  void addOverlapEntry(TGeoOverlap*, int, int, TGeoHMatrix*);
  FWOverlapTableView* m_browser;

  std::multimap<int, int> m_mapNodeOverlaps;
};

#endif
