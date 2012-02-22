// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableManager
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Wed Jan  4 20:31:25 CET 2012
// $Id: FWGeometryTableManager.cc,v 1.44 2012/02/22 03:45:59 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/src/FWGeometryTableManager.h"
#include "Fireworks/Core/interface/FWGeometryTableViewBase.h"
#include "Fireworks/Core/interface/FWGeometryTableViewManager.h"
#include "Fireworks/Core/src/FWGeometryTableView.h"

#include "TEveUtil.h"

FWGeometryTableManager::FWGeometryTableManager(FWGeometryTableView* v):
   FWGeometryTableManagerBase(),
   m_browser(v),
   m_filterOff(true)
{
}


FWGeometryTableManager::~FWGeometryTableManager()
{
}

const char* FWGeometryTableManager::cellName(const NodeInfo& data) const
{
   if (m_browser->getVolumeMode())
      return Form("%s [%d]", data.m_node->GetVolume()->GetName(), data.m_node->GetNdaughters());
   else    
      return Form("%s [%d]", data.m_node->GetName(), data.m_node->GetNdaughters()); 
}

//______________________________________________________________________________

FWTableCellRendererBase* FWGeometryTableManager::cellRenderer(int iSortedRowNumber, int iCol) const
{
   FWTextTreeCellRenderer* renderer = &m_renderer;
   if (m_row_to_index.empty()) return renderer;


   int unsortedRow =  m_row_to_index[iSortedRowNumber];
   if (unsortedRow < 0) printf("!!!!!!!!!!!!!!!! error %d %d \n",unsortedRow,  iSortedRowNumber);


   const NodeInfo& data = m_entries[unsortedRow];
   TGeoNode& gn = *data.m_node;
   bool isSelected = data.testBit(kHighlighted) ||  data.testBit(kSelected);
   // printf("cell render %s \n", data.name());
   if (data.testBit(kSelected))
   {
      m_highlightContext->SetBackground(0xc86464);
   }
   else if (data.testBit(kHighlighted) )
   {
      m_highlightContext->SetBackground(0x6464c8);
   }
   else if (iCol == kMaterialColumn && data.testBit(kMatches) )
   {
         m_highlightContext->SetBackground(0xdddddd);
   }


   if (iCol == kNameColumn)
   {
      renderer->setData(cellName(data), isSelected); 


      renderer->setIsParent(nodeIsParent(data));

      renderer->setIsOpen( data.testBit(FWGeometryTableManagerBase::kExpanded));

      int level = data.m_level - m_levelOffset;
      if (nodeIsParent(data))
         renderer->setIndentation(20*level);
      else
         renderer->setIndentation(20*level + FWTextTreeCellRenderer::iconWidth());

      return renderer;
   }
   else
   {
      // printf("title %s \n",data.m_node->GetTitle() );
      renderer->setIsParent(false);
      renderer->setIndentation(0);
      if (iCol == kColorColumn)
      {
         // m_colorBoxRenderer.setData(data.m_node->GetVolume()->GetLineColor(), isSelected);
         m_colorBoxRenderer.setData(data.m_color, isSelected);
         return  &m_colorBoxRenderer;
      }
      else if (iCol == kVisSelfColumn )
      {
         renderer->setData(getVisibility(data)  ? "On" : "-",  isSelected );
         return renderer;
      }
      else if (iCol == kVisChildColumn )
      {
         renderer->setData( getVisibilityChld(data) ? "On" : "-",  isSelected);
         return renderer;
      }
      else if (iCol == kMaterialColumn )
      { 
         renderer->setData( gn.GetVolume()->GetMaterial()->GetName(),  isSelected);
         return renderer;
      }
      else
      {  renderer->setData("ERROR", false);
         return renderer;
      }

   }
}
//____________________________________________________________________________

void FWGeometryTableManager::importChildren(int parent_idx)
{
   NodeInfo& parent        = m_entries[parent_idx];
   TGeoNode* parentGeoNode = parent.m_node; 
   int       parentLevel   = parent.m_level; 
   
   int nV = parentGeoNode->GetNdaughters();
   int dOff = 0; 
   for (int n = 0; n != nV; ++n)
   {         
      NodeInfo& data = m_entries[parent_idx + n + 1 + dOff];    
      data.m_node =   parentGeoNode->GetDaughter(n);
      data.m_level =  parentLevel + 1;
      data.m_parent = parent_idx;
      data.m_color =  data.m_node->GetVolume()->GetLineColor();
      if (data.m_level <=  m_browser->getAutoExpand()) data.setBit(kExpanded);
      
 
      importChildren(parent_idx + n + 1 + dOff);         
      getNNodesTotal(parentGeoNode->GetDaughter(n), dOff);            
   }  
}

//==============================================================================

void FWGeometryTableManager::checkHierarchy()
{
   // Used for debug: in a NodeInfo entry look TGeoNode children from parent index and check
   // if child is found.
   
   for ( size_t i = 0,  e = m_entries.size(); i != e; ++i )
   {
      if ( m_entries[i].m_level > 0)
      {
         TGeoNode* pn = m_entries[m_entries[i].m_parent].m_node;
         bool ok = false;
         for (int d = 0; d < pn->GetNdaughters(); ++d )
         {
            if (m_entries[i].m_node ==  pn->GetDaughter(d))
            {
               ok = true;
               break;
            }
         }
         if (!ok) printf("!!!!!! node %s has false parent %s \n", m_entries[i].name(), pn->GetName());
      }   
   }
}

void FWGeometryTableManager::checkChildMatches(TGeoVolume* vol,  std::vector<TGeoVolume*>& pstack)
{
   if (m_volumes[vol].m_matches)
   {
      for (std::vector<TGeoVolume*>::iterator i = pstack.begin(); i!= pstack.end(); ++i)
      {
         Match& pm =  m_volumes[*i];
         pm.m_childMatches = true; 
      }
   }

   pstack.push_back(vol);

   int nD =  vol->GetNdaughters();//TMath::Min(m_browser->getMaxDaughters(), vol->GetNdaughters());
   for (int i = 0; i!=nD; ++i)
      checkChildMatches(vol->GetNode(i)->GetVolume(), pstack);
   
   pstack.pop_back();
}

// callbacks ______________________________________________________________________________

void FWGeometryTableManager::updateFilter()
{
   std::string filterExp =  m_browser->getFilter();
   m_filterOff =  filterExp.empty();
   //   printf("update filter %s  OFF %d volumes size %d\n",filterExp.c_str(),  m_filterOff , (int)m_volumes.size());

   if (m_filterOff || m_entries.empty()) return;
   
   // update volume-match entries
   //   m_numVolumesMatched = 0;
   for (Volumes_i i = m_volumes.begin(); i!= m_volumes.end(); ++i) 
   {
      if (strcasestr(i->first->GetMaterial()->GetName(), filterExp.c_str()) > 0) {
         i->second.m_matches = true;
         //    m_numVolumesMatched++;
      }
      else {
         i->second.m_matches = false;
      }
      i->second.m_childMatches = false;
   }  

   std::vector<TGeoVolume*> pstack;
   checkChildMatches(m_entries[TMath::Max(0,m_browser->getTopNodeIdx())].m_node->GetVolume(), pstack);
 

   for (Entries_i ni = m_entries.begin(); ni != m_entries.end(); ++ni)
      ni->resetBit(kFilterCached);

}

//==============================================================================

void FWGeometryTableManager::loadGeometry( TGeoNode* iGeoTopNode, TObjArray* iVolumes)
{
#ifdef PERFTOOL_GEO_TABLE  
   ProfilerStart("loadGeo");
#endif
   
   // Prepare data for cell render.
   
   // clear entries
   m_entries.clear();
   m_row_to_index.clear();
   m_volumes.clear();
   m_levelOffset = 0;

   // set volume table for filters
   boost::unordered_map<TGeoVolume*, Match>  pipi(iVolumes->GetSize());
   m_volumes.swap(pipi);
   TIter next( iVolumes);
   TGeoVolume* v;
   while ((v = (TGeoVolume*) next()) != 0)
      m_volumes.insert(std::make_pair(v, Match()));

   if (!m_filterOff)
      updateFilter();  

   // add top node to init
 
   int nTotal = 0;
   NodeInfo topNodeInfo;
   topNodeInfo.m_node   = iGeoTopNode;
   topNodeInfo.m_level  = 0;
   topNodeInfo.m_parent = -1;
   topNodeInfo.setBitVal(kExpanded, m_browser->getAutoExpand());
   topNodeInfo.setBitVal(kVisNodeSelf, m_browser->drawTopNode());

   getNNodesTotal(topNodeInfo.m_node , nTotal);
   m_entries.resize(nTotal+1);
   m_entries[0] = topNodeInfo;

   importChildren(0);

   // checkHierarchy();
   
#ifdef PERFTOOL_GEO_TABLE  
   ProfilerStop();
#endif
}



//______________________________________________________________________________
void FWGeometryTableManager::printMaterials()
{
   std::cerr << "not implemented \n";
}
//______________________________________________________________________________
      
void FWGeometryTableManager::recalculateVisibility()
{
   m_row_to_index.clear();


   int i = TMath::Max(0, m_browser->getTopNodeIdx());
   m_row_to_index.push_back(i);

   NodeInfo& data = m_entries[i];

   if (!m_filterOff)
      assertNodeFilterCache(data);

   if ((m_filterOff && data.testBit(kExpanded) == false) ||
       (m_filterOff == false && data.testBit(kChildMatches) == false) )
      return;

   if (m_browser->getVolumeMode())
      recalculateVisibilityVolumeRec(i);
   else
      recalculateVisibilityNodeRec(i);

   //  printf (" child [%d] FWGeometryTableManagerBase::recalculateVisibility table size %d \n", (int)m_row_to_index.size());
}

//______________________________________________________________________________

void FWGeometryTableManager::recalculateVisibilityVolumeRec(int pIdx)
{
   TGeoNode* parentNode = m_entries[pIdx].m_node;
   int nD = parentNode->GetNdaughters();
   int dOff=0;

   // printf("----------- parent %s\n", parentNode->GetName() );

   std::vector<int> vi; 
   vi.reserve(nD);


   for (int n = 0; n != nD; ++n)
   {
      int idx = pIdx + 1 + n + dOff;
      NodeInfo& data = m_entries[idx];

      bool toAdd = true;
      for (std::vector<int>::iterator u = vi.begin(); u != vi.end(); ++u )
      {
         TGeoVolume* neighbourVolume =  parentNode->GetDaughter(*u)->GetVolume();
         if (neighbourVolume == data.m_node->GetVolume())
         {
            toAdd = false;
            // printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
            break;
         }
      }

      if (toAdd)
      {
         vi.push_back(n);
         if (m_filterOff)
         {
            //    std::cout << data.nameIndent() << std::endl;
            m_row_to_index.push_back(idx);
            if (data.testBit(kExpanded)) recalculateVisibilityVolumeRec(idx);
         }
         else
         {
            assertNodeFilterCache(data);
            if (data.testBitAny(kMatches | kChildMatches)) m_row_to_index.push_back(idx); 
            if (data.testBit(kChildMatches) && data.testBit(kExpanded)) recalculateVisibilityVolumeRec(idx);
         }
      }
      FWGeometryTableManagerBase::getNNodesTotal(parentNode->GetDaughter(n), dOff);
   }
}

//______________________________________________________________________________

void FWGeometryTableManager::recalculateVisibilityNodeRec( int pIdx)
{
   TGeoNode* parentNode = m_entries[pIdx].m_node;
   int nD = parentNode->GetNdaughters();
   int dOff=0;
   for (int n = 0; n != nD; ++n)
   {
      int idx = pIdx + 1 + n + dOff;
      NodeInfo& data = m_entries[idx];

      if (m_filterOff)
      {
         m_row_to_index.push_back(idx);
         if (data.testBit(kExpanded)) recalculateVisibilityNodeRec(idx);
      }
      else
      {
         assertNodeFilterCache(data);
         if (data.testBitAny(kMatches | kChildMatches)) m_row_to_index.push_back(idx); 
         if (data.testBit(kChildMatches) && data.testBit(kExpanded) ) recalculateVisibilityNodeRec(idx);
      }

      FWGeometryTableManagerBase::getNNodesTotal(parentNode->GetDaughter(n), dOff);
   }
} 

//______________________________________________________________________________
void FWGeometryTableManager::assertNodeFilterCache(NodeInfo& data)
{
   if (!data.testBit(kFilterCached))
   {
      bool matches = m_volumes[data.m_node->GetVolume()].m_matches;
      data.setBitVal(kMatches, matches);
      setVisibility(data, matches);

      bool childMatches = m_volumes[data.m_node->GetVolume()].m_childMatches;
      data.setBitVal(kChildMatches, childMatches);
      data.setBitVal(kExpanded, childMatches);
      setVisibilityChld(data, childMatches);

      data.setBit(kFilterCached);
      //  printf("%s matches [%d] childMatches [%d] ................ %d %d \n", data.name(), data.testBit(kMatches), data.testBit(kChildMatches), matches , childMatches);
   }
}

//______________________________________________________________________________
void FWGeometryTableManager::setVisibility(NodeInfo& data, bool x)
{
   if (m_browser->getVolumeMode())
   {
      if (data.m_node->GetVolume()->IsVisible() != x)
      {
         FWGeometryTableViewManager::getGeoMangeur();
         data.m_node->GetVolume()->SetVisibility(x);
      }
   }
   else
   {
      data.setBitVal(kVisNodeSelf, x);
   }
}

//______________________________________________________________________________

void FWGeometryTableManager::setVisibilityChld(NodeInfo& data, bool x)
{
   if (m_browser->getVolumeMode())
   {
      if (data.m_node->GetVolume()->IsVisibleDaughters() != x)
      {
         TEveGeoManagerHolder gmgr( FWGeometryTableViewManager::getGeoMangeur());
         data.m_node->GetVolume()->VisibleDaughters(x);
      }
   }
   else
      data.setBitVal(kVisNodeChld, x);
}
//______________________________________________________________________________

void FWGeometryTableManager::setDaughtersSelfVisibility(int selectedIdx, bool v)
{
   int dOff = 0;
   TGeoNode* parentNode = m_entries[selectedIdx].m_node;
   int nD = parentNode->GetNdaughters();
   for (int n = 0; n != nD; ++n)
   {
      int idx = selectedIdx + 1 + n + dOff;
      NodeInfo& data = m_entries[idx];

      setVisibility(data, v);
      setVisibilityChld(data, v);

      FWGeometryTableManager::getNNodesTotal(parentNode->GetDaughter(n), dOff);
   }
}

//______________________________________________________________________________

bool  FWGeometryTableManager::getVisibility(const NodeInfo& data) const
{
   if (m_browser->getVolumeMode())
      return data.m_node->GetVolume()->IsVisible();

   return  data.testBit(kVisNodeSelf);   
}

bool  FWGeometryTableManager::getVisibilityChld(const NodeInfo& data) const
{
   if (m_browser->getVolumeMode())
      return data.m_node->GetVolume()->IsVisibleDaughters();

   return  data.testBit(kVisNodeChld);   
}

//______________________________________________________________________________

bool  FWGeometryTableManager::nodeIsParent(const NodeInfo& data) const
{
   return   (data.m_node->GetNdaughters() != 0) && (m_filterOff || data.testBit(kChildMatches) );
}

