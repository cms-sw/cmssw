#include "Fireworks/Core/src/FWEveOverlap.h"
#include "TGeoOverlap.h"
#include "TEveGeoShape.h"
#include "Fireworks/Core/src/FWOverlapTableView.h"
#include "Fireworks/Core/src/FWOverlapTableManager.h"
#include "Fireworks/Core/interface/FWGeometryTableViewManager.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/src/FWPopupMenu.cc"
//==============================================================================
//==============================================================================
//==============================================================================
FWEveOverlap::FWEveOverlap(FWOverlapTableView* v) : m_browser(v) {}
FWGeometryTableManagerBase* FWEveOverlap::tableManager() { return m_browser->getTableManager(); }

FWGeometryTableViewBase* FWEveOverlap::browser() { return m_browser; }
//______________________________________________________________________________

void FWEveOverlap::Paint(Option_t*) {
  if (m_browser->getTableManager()->refEntries().empty())
    return;

  FWGeoTopNode::Paint();

  TEveGeoManagerHolder gmgr(FWGeometryTableViewManager::getGeoMangeur());

  int topNodeIdx = m_browser->getTopNodeIdx();

  FWGeometryTableManagerBase::Entries_i sit = m_browser->getTableManager()->refEntries().begin();
  std::advance(sit, topNodeIdx);
  TGeoHMatrix mtx;
  m_browser->getTableManager()->getNodeMatrix(*sit, mtx);

  bool drawsChildren = false;

  if ((*sit).testBit(FWGeometryTableManagerBase::kVisNodeChld))
    drawsChildren = paintChildNodesRecurse(sit, topNodeIdx, mtx);

  if (sit->testBit(FWGeometryTableManagerBase::kVisNodeSelf))
    paintShape(topNodeIdx, mtx, false, drawsChildren);
}

// ______________________________________________________________________
bool FWEveOverlap::paintChildNodesRecurse(FWGeometryTableManagerBase::Entries_i pIt,
                                          Int_t cnt,
                                          const TGeoHMatrix& parentMtx) {
  TGeoNode* parentNode = pIt->m_node;
  int nD = parentNode->GetNdaughters();

  int dOff = 0;

  pIt++;
  int pcnt = cnt + 1;

  bool drawsChildren = false;

  FWGeometryTableManagerBase::Entries_i it;
  for (int n = 0; n != nD; ++n) {
    it = pIt;
    std::advance(it, n + dOff);
    cnt = pcnt + n + dOff;

    TGeoHMatrix nm = parentMtx;
    nm.Multiply(it->m_node->GetMatrix());

    bool drawsChildrenSecondGen = false;
    if (it->testBit(FWGeometryTableManagerBase::kVisNodeChld) && it->testBit(FWOverlapTableManager::kOverlapChild))
      drawsChildrenSecondGen = paintChildNodesRecurse(it, cnt, nm);

    if (it->testBit(FWGeometryTableManagerBase::kVisNodeSelf)) {
      if (it->testBit(FWOverlapTableManager::kOverlap)) {
        int nno;
        it->m_node->GetOverlaps(nno);
        if ((m_browser->m_rnrOverlap.value() && ((nno & BIT(1)) == BIT(1))) ||
            (m_browser->m_rnrExtrusion.value() && ((nno & BIT(2)) == BIT(2)))) {
          paintShape(cnt, nm, false, drawsChildrenSecondGen);
          drawsChildren = true;
        }

      } else {
        paintShape(cnt, nm, false, drawsChildrenSecondGen);
        drawsChildren = true;
      }
    }

    drawsChildren |= drawsChildrenSecondGen;
    FWGeometryTableManagerBase::getNNodesTotal(parentNode->GetDaughter(n), dOff);
  }
  return drawsChildren;
}

//______________________________________________________________________________

TString FWEveOverlap::GetHighlightTooltip() {
  //   printf("highlight tooltio \n");
  std::set<TGLPhysicalShape*>::iterator it = fHted.begin();
  int idx = tableIdx(*it);
  if (idx < 0) {
    return Form("TopNode ");
  }
  FWGeometryTableManagerBase::NodeInfo& data = m_browser->getTableManager()->refEntries().at(idx);

  TString name = data.name();
  if (data.testBit(FWOverlapTableManager::kOverlap)) {
    ((FWOverlapTableManager*)m_browser->getTableManager())->getOverlapTitles(idx, name);
    return name;
  }

  return data.name();
}

//_____________________________________________________________________________

void FWEveOverlap::popupMenu(int x, int y, TGLViewer* v) {
  FWPopupMenu* nodePopup = setPopupMenu(x, y, v, true);

  if (nodePopup)
    nodePopup->Connect("Activated(Int_t)", "FWOverlapTableView", m_browser, "chosenItem(Int_t)");
}
