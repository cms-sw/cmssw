#include "Fireworks/Core/src/FWEveDetectorGeo.h"
#include "Fireworks/Core/src/FWGeometryTableView.h"
#include "Fireworks/Core/src/FWGeometryTableManager.h"
#include "Fireworks/Core/interface/FWGeometryTableViewManager.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/src/FWPopupMenu.cc"

#include "TGeoMatrix.h"

//==============================================================================
//==============================================================================
//==============================================================================
FWEveDetectorGeo::FWEveDetectorGeo(FWGeometryTableView* v):
   m_browser(v), m_maxLevel(0), m_filterOff(0)
{
} 

FWGeometryTableManagerBase* FWEveDetectorGeo::tableManager()
{
   return m_browser->getTableManager();
}

FWGeometryTableViewBase* FWEveDetectorGeo::browser()
{
   return m_browser;
}
//______________________________________________________________________________

void FWEveDetectorGeo::Paint(Option_t* opt)
{
   FWGeoTopNode::Paint();

   // printf("PAINPAINTPAINTPAINTPAINTPAINTPAINTPAINTPAINTPAINTT  %d/%d \n",  m_browser->getTopNodeIdx(),  (int)m_browser->getTableManager()->refEntries().size());
   if (m_browser->getTableManager()->refEntries().empty()) return; 

   TEveGeoManagerHolder gmgr( FWGeometryTableViewManager::getGeoMangeur());

   m_maxLevel = m_browser->getVisLevel() + m_browser->getTableManager()->getLevelOffset();

   m_filterOff = m_browser->getFilter().empty();

   Int_t topIdx = m_browser->getTopNodeIdx();
   FWGeometryTableManagerBase::Entries_i sit = m_browser->getTableManager()->refEntries().begin(); 

   TGeoHMatrix mtx;
   if (topIdx >= 0)
   {
      std::advance(sit, topIdx);
      m_browser->getTableManager()->getNodeMatrix(*sit, mtx);
   }

   bool drawsChildren = 0;
   
   if ( ((FWGeometryTableManager*)tableManager())->getVisibilityChld(*sit))
      drawsChildren = paintChildNodesRecurse( sit, topIdx, mtx);
   
   if (sit->testBit(FWGeometryTableManagerBase::kVisNodeSelf) && ((FWGeometryTableManager*)tableManager())->getVisibility(*sit))
      paintShape( topIdx,mtx, m_browser->getVolumeMode(), drawsChildren );
   
   
   fflush(stdout);
}


// ______________________________________________________________________
bool FWEveDetectorGeo::paintChildNodesRecurse (FWGeometryTableManagerBase::Entries_i pIt, Int_t cnt, const TGeoHMatrix& parentMtx)
{ 
   TGeoNode* parentNode =  pIt->m_node;
   int nD = parentNode->GetNdaughters();

   int dOff=0;

   pIt++;
   int pcnt = cnt+1;
   
   bool drawsChildNodes = 0;

   FWGeometryTableManagerBase::Entries_i it;
   for (int n = 0; n != nD; ++n)
   {
      it =  pIt;
      std::advance(it,n + dOff);
      cnt = pcnt + n+dOff;

      TGeoHMatrix nm = parentMtx;
      nm.Multiply(it->m_node->GetMatrix());

      bool drawsChildNodesSecondGen = false;
      if (m_filterOff || m_browser->isSelectedByRegion())
      {
         if  ( ((FWGeometryTableManager*)tableManager())->getVisibilityChld(*it) && ( it->m_level < m_maxLevel)) {
           drawsChildNodesSecondGen = paintChildNodesRecurse(it,cnt , nm);
         }
         
         if ( ((FWGeometryTableManager*)tableManager())->getVisibility(*it))
         {
            paintShape(cnt , nm, m_browser->getVolumeMode(),  drawsChildNodesSecondGen );
            drawsChildNodes = true;
         }

      }
      else
      {
         if ( ((FWGeometryTableManager*)tableManager())->getVisibilityChld(*it) && ( it->m_level < m_maxLevel || m_browser->getIgnoreVisLevelWhenFilter() ))
         {
            drawsChildNodesSecondGen = paintChildNodesRecurse(it,cnt , nm);
         }
         
         ((FWGeometryTableManager*)tableManager())->assertNodeFilterCache(*it);
         if ( ((FWGeometryTableManager*)tableManager())->getVisibility(*it))
         {
            paintShape(cnt , nm, m_browser->getVolumeMode(), drawsChildNodesSecondGen );
            drawsChildNodes = true;
         }
      }

      drawsChildNodes |= drawsChildNodesSecondGen;
      FWGeometryTableManagerBase::getNNodesTotal(parentNode->GetDaughter(n), dOff);  
   }
   
   return  drawsChildNodes;
}

//______________________________________________________________________________

TString  FWEveDetectorGeo::GetHighlightTooltip()
{
   std::set<TGLPhysicalShape*>::iterator it = fHted.begin();
   int idx = tableIdx(*it);
   if (idx > 0)
   {
      FWGeometryTableManagerBase::NodeInfo& data = m_browser->getTableManager()->refEntries().at(idx);
      return data.name();
   }
   return "error";
}

//_____________________________________________________________________________

void FWEveDetectorGeo::popupMenu(int x, int y, TGLViewer* v)
{
   FWPopupMenu* nodePopup = FWGeoTopNode::setPopupMenu(x, y, v, false);
   
 if (nodePopup)  nodePopup->Connect("Activated(Int_t)",
                      "FWGeometryTableView",
                      m_browser,
                      "chosenItem(Int_t)");
}