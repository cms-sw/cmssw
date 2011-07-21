// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableManager
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel, Matevz Tadel
//         Created:  Thu Jan 27 14:50:57 CET 2011
// $Id: FWGeometryTableManager.cc,v 1.40 2011/07/20 23:18:59 amraktad Exp $
//

//#define PERFTOOL_GEO_TABLE

// user include files
#include <iostream>
#include <boost/bind.hpp>
#include <stack>
#ifdef PERFTOOL_GEO_TABLE 
#include <google/profiler.h>
#endif
#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "Fireworks/Core/interface/FWGeometryTableView.h"
#include "Fireworks/Core/interface/FWGeometryTableViewManager.h"
#include "Fireworks/Core/src/FWColorBoxIcon.h"
#include "Fireworks/TableWidget/interface/GlobalContexts.h"
#include "Fireworks/TableWidget/src/FWTabularWidget.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "TMath.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"
#include "TGeoShape.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TEveUtil.h"

#include "TGFrame.h"

static const char* redTxt   = "\033[01;31m";
//static const char* greenTxt = "\033[01;32m";
// static const char* cyanTxt  = "\033[22;36m";
// static const char* whiteTxt = "\033[0m";

const char* FWGeometryTableManager::NodeInfo::name() const
{
   return m_node->GetName();
}

const char* FWGeometryTableManager::NodeInfo::nameIndent() const
{
   std::string x(m_level*3 , ' ');
   x += m_node->GetName();
   return x.c_str();
}

FWGeometryTableManager::ColorBoxRenderer::ColorBoxRenderer():
   FWTableCellRendererBase(),
   m_width(1),
   m_height(1),
   m_color(0xffffff),
   m_isSelected(false)
{
   GCValues_t gval; 
   gval.fMask       = kGCForeground | kGCBackground | kGCStipple | kGCFillStyle  | kGCGraphicsExposures;
   gval.fStipple    = gClient->GetResourcePool()->GetCheckeredBitmap();
   gval.fGraphicsExposures = kFALSE;
   gval.fBackground = gVirtualX->GetPixel(kGray);
   m_colorContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&gval,kTRUE);

}

FWGeometryTableManager::ColorBoxRenderer::~ColorBoxRenderer()
{
   gClient->GetResourcePool()->GetGCPool()->FreeGC(m_colorContext->GetGC());
}

void FWGeometryTableManager::ColorBoxRenderer::setData(Color_t c, bool s)
{
   m_color = gVirtualX->GetPixel(c);
   m_isSelected = s;
}


void FWGeometryTableManager::ColorBoxRenderer::draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
{
   iX -= FWTabularWidget::kTextBuffer;
   iY -= FWTabularWidget::kTextBuffer;
   iWidth += 2*FWTabularWidget::kTextBuffer;
   iHeight += 2*FWTabularWidget::kTextBuffer;

   m_colorContext->SetFillStyle(kFillSolid);
   Pixel_t baq =  m_colorContext->GetForeground();
   m_colorContext->SetForeground(m_color);
   gVirtualX->FillRectangle(iID, m_colorContext->GetGC(), iX, iY, iWidth, iHeight);

   if (m_isSelected)
   {
      m_colorContext->SetFillStyle(kFillOpaqueStippled);
      gVirtualX->FillRectangle(iID, m_colorContext->GetGC(), iX, iY, iWidth, iHeight);
   }
   m_colorContext->SetForeground(baq);
}

//==============================================================================
//==============================================================================
//
// class FWGeometryTableManager
//
//==============================================================================
//==============================================================================

FWGeometryTableManager::FWGeometryTableManager(FWGeometryTableView* browser)
:    m_selectedRow(-1),
     m_selectedIdx(0),
     m_selectedColumn(-1),
     m_browser(browser),
     m_filterOff(true),
     m_numVolumesMatched(-1),
     m_levelOffset(0)
{ 
   m_colorBoxRenderer.m_width  =  50;
   m_colorBoxRenderer.m_height =  m_renderer.height();
}

FWGeometryTableManager::~FWGeometryTableManager()
{
}


int FWGeometryTableManager::unsortedRowNumber(int unsorted) const
{
   return unsorted;
}

int FWGeometryTableManager::numberOfRows() const 
{
   return m_row_to_index.size();
}

int FWGeometryTableManager::numberOfColumns() const 
{
   return kNumCol;
}
   

std::vector<std::string> FWGeometryTableManager::getTitles() const 
{
   std::vector<std::string> returnValue;
   returnValue.reserve(numberOfColumns());

   if (m_browser->getVolumeMode() )
      returnValue.push_back("Volume Name");
   else
      returnValue.push_back("Node Name");

   returnValue.push_back("Color");
   returnValue.push_back("RnrSelf");
   returnValue.push_back("RnrChildren");
   returnValue.push_back("Material");
   returnValue.push_back("X");
   returnValue.push_back("Y");
   returnValue.push_back("Z");
   //   returnValue.push_back("Diagonal");

   return returnValue;
}
  
void FWGeometryTableManager::setSelection (int row, int column, int mask) 
{
   changeSelection(row, column);
}

const std::string FWGeometryTableManager::title() const 
{
   return "Geometry";
}

int FWGeometryTableManager::selectedRow() const 
{
   return m_selectedRow;
}

int FWGeometryTableManager::selectedColumn() const 
{
   return m_selectedColumn;
}
 
bool FWGeometryTableManager::rowIsSelected(int row) const 
{
   return m_selectedRow == row;
}

void FWGeometryTableManager::changeSelection(int iRow, int iColumn)
{     
   if (iRow < 0) return; 

   m_selectedRow = iRow;
   m_selectedColumn = iColumn;
   if (m_row_to_index.size() > 0)   m_selectedIdx = m_row_to_index[iRow];

   // printf("WGeometryTableManager::changeSelecti row %d index %d \n", m_selectedIdx,  m_selectedIdx );
   visualPropertiesChanged();
}    

void  FWGeometryTableManager::setBackgroundToWhite(bool iToWhite )
{
   if(iToWhite) {
      m_renderer.setGraphicsContext(&TGFrame::GetBlackGC());
      m_renderer.setHighlightContext(&(FWTextTableCellRenderer::getDefaultHighlightGC()));
   } else {
      static const TGGC* s_blackHighLight = 0;
      if (!s_blackHighLight) {
         GCValues_t gval;
         gval.fMask = kGCForeground | kGCBackground | kGCStipple | kGCFillStyle  | kGCGraphicsExposures;
         gval.fForeground = 0xbbbbbb;
         gval.fBackground = 0x000000;
         gval.fFillStyle  = kFillOpaqueStippled;
         gval.fStipple    = gClient->GetResourcePool()->GetCheckeredBitmap();
         gval.fGraphicsExposures = kFALSE;
         s_blackHighLight = gClient->GetGC(&gval, kTRUE);
      }
      m_renderer.setHighlightContext(s_blackHighLight);
      m_renderer.setGraphicsContext(&TGFrame::GetWhiteGC());
   }
   m_renderer.setBlackIcon(iToWhite);
}

//______________________________________________________________________________

FWTableCellRendererBase* FWGeometryTableManager::cellRenderer(int iSortedRowNumber, int iCol) const
{
   struct Cache
   {
      TGeoHMatrix mtx;
      double pos[3];
      int row;

   };

   static Cache mxCache;
   if (static_cast<int>(m_row_to_index.size()) <= iSortedRowNumber)
   {
      m_renderer.setData(std::string("FWGeometryTableManager::cellRenderer() Error!"), false);
      return &m_renderer;
   }       

   static char sval[12];
   static std::string zero = "0.000";

   FWTextTreeCellRenderer* renderer = &m_renderer;

   int unsortedRow =  m_row_to_index[iSortedRowNumber];
   const NodeInfo& data = m_entries[unsortedRow];
   TGeoNode& gn = *data.m_node;

   bool isSelected = (iCol == kMaterial ) && (!m_filterOff &&  m_volumes[gn.GetVolume()].m_matches);//(m_selectedRow == unsortedRow);

   if (iCol == kName)
   {
      //int nD = getNdaughtersLimited(data.m_node);
      int nD = data.m_node->GetNdaughters();
      if (m_browser->getVolumeMode())
         renderer->setData(Form("%s [%d]", gn.GetVolume()->GetName(), nD), isSelected);
      else    
         renderer->setData(Form("%s [%d]", gn.GetName(), nD), isSelected); 

      renderer->setIsParent((gn.GetNdaughters() > 0) && (m_filterOff || data.testBit(kChildMatches) ));

      renderer->setIsOpen( data.testBit(FWGeometryTableManager::kExpanded));

      int level = data.m_level - m_levelOffset;

      if (data.m_node->GetNdaughters())
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
      if (iCol == kColor)
      {
         // m_colorBoxRenderer.setData(data.m_node->GetVolume()->GetLineColor(), isSelected);
         m_colorBoxRenderer.setData(data.m_color, isSelected);
         return  &m_colorBoxRenderer;
      }
      else if (iCol == kVisSelf )
      {
         renderer->setData(getVisibility(data)  ? "On" : "-",  isSelected );
         return renderer;
      }
      else if (iCol == kVisChild )
      {
         renderer->setData( getVisibilityChld(data) ? "On" : "-",  isSelected);
         return renderer;
      }
      else if (iCol == kMaterial )
      { 
         renderer->setData( gn.GetVolume()->GetMaterial()->GetName(),  isSelected);
         return renderer;
      }
      
      else if (iCol == kPosX || iCol == kPosY || iCol == kPosZ)
      { 
         if (mxCache.row != iSortedRowNumber) { 
            mxCache.row = iSortedRowNumber;
            mxCache.pos[0] = 0; mxCache.pos[1] = 0; mxCache.pos[2] = 0;
            mxCache.mtx.Clear();
            getNodeMatrix(data, mxCache.mtx);
            TGeoBBox* bb = static_cast<TGeoBBox*>(data.m_node->GetVolume()->GetShape());
            const double* origin = bb->GetOrigin();
            mxCache.mtx.LocalToMaster(origin, mxCache.pos);
         }
         float val =  mxCache.pos[iCol - kPosX];
         if (val < 0.001)  {
            renderer->setData(zero, isSelected);
         }
         else {
            snprintf(sval, sizeof(sval), "%.3f", val);
            renderer->setData(sval, isSelected);

         }

         return renderer;
      }
      else
      { 
         TGeoBBox* gs = static_cast<TGeoBBox*>( gn.GetVolume()->GetShape());
         float val = TMath::Sqrt(gs->GetDX()*gs->GetDX() + gs->GetDY()*gs->GetDY() +gs->GetDZ()*gs->GetDZ() );
         //            renderer->setData( Form("%.3f", TMath::Sqrt(gs->GetDX()*gs->GetDX() + gs->GetDY()*gs->GetDY() +gs->GetDZ()*gs->GetDZ() )),  isSelected);
         if (val < 0.001)  {
            renderer->setData(zero, isSelected);
         }
         else {
            snprintf(sval, sizeof(sval), "%.3f", val);
            renderer->setData(sval, isSelected);
         }
         return renderer;
      }

   }
}

//______________________________________________________________________________
void FWGeometryTableManager::firstColumnClicked(int row)
{
   if (row == -1)
      return;

   int idx = rowToIndex()[row];
   // printf("click %s \n", m_entries[idx].name());

   m_entries[idx].switchBit(kExpanded);
 
   recalculateVisibility();
   dataChanged();
   visualPropertiesChanged();
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

   //  printf (" child [%d] FWGeometryTableManager::recalculateVisibility table size %d \n", (int)m_row_to_index.size());
}


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

      FWGeometryTableManager::getNNodesTotal(parentNode->GetDaughter(n), dOff);
   }
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
      FWGeometryTableManager::getNNodesTotal(parentNode->GetDaughter(n), dOff);
   }
}


//______________________________________________________________________________

void FWGeometryTableManager::redrawTable() 
{
   if (m_entries.empty()) return;

   changeSelection(0, 0);

   recalculateVisibility();

   // if (m_filterOff)  
   //   m_statusMessage = Form("level:%d  rows %d ",   getLevelOffset(), (int)m_row_to_index.size());
   // else
   //   m_statusMessage = Form("level:%d  rows %d volumes %d (%.2f %%) selected ", getLevelOffset(), (int)m_row_to_index.size(), m_numVolumesMatched, 100.0* m_numVolumesMatched/m_volumes.size());

   dataChanged();
   visualPropertiesChanged();
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
   if (m_browser->getAutoExpand())
      topNodeInfo.setBit(kExpanded);

   getNNodesTotal(topNodeInfo.m_node , nTotal);
   m_entries.resize(nTotal+1);
   m_entries[0] = topNodeInfo;

   importChildren(0);

   // checkHierarchy();
   
#ifdef PERFTOOL_GEO_TABLE  
   ProfilerStop();
#endif
}

//==============================================================================


//______________________________________________________________________________

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
         if (!ok) printf("%s!!!!!! node %s has false parent %s \n", redTxt, m_entries[i].name(), pn->GetName());
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
    m_numVolumesMatched = 0;
   for (Volumes_i i = m_volumes.begin(); i!= m_volumes.end(); ++i) 
   {
      if (strcasestr(i->first->GetMaterial()->GetName(), filterExp.c_str()) > 0) {
         i->second.m_matches = true;
         m_numVolumesMatched++;
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


FWGeometryTableManager::Entries_i FWGeometryTableManager::refSelected()
{
   Entries_i i = m_entries.begin();
   std::advance(i,m_selectedIdx );
   return  i; //m_entries[m_selectedIdx];
}

void FWGeometryTableManager::getNodePath(int idx, std::string& path) const
{
   std::vector<std::string> relPath;
   while(idx >= 0)
   { 
      relPath.push_back( m_entries[idx].name());
      // printf("push %s \n",m_entries[idx].name() );
      idx  =  m_entries[idx].m_parent;
   }

   size_t ns = relPath.size();
   for (size_t i = 1; i < ns; ++i )
   {
      path +="/";
      path += relPath[ns-i -1];
      // printf("push_back add to path %s\n", path.c_str());
   }
}

//______________________________________________________________________________

void  FWGeometryTableManager::checkExpandLevel()
{
   // check expand state
   int ae = m_browser->getAutoExpand() +  m_levelOffset;
   for (Entries_i i = m_entries.begin(); i != m_entries.end(); ++i)
   {
      if (i->m_level  < ae)
         i->setBit(kExpanded);
      else
         i->resetBit(kExpanded);
   } 
}

//______________________________________________________________________________


void  FWGeometryTableManager::topGeoNodeChanged(int idx)
{
   // cached 
   if (idx >= 0)
      m_levelOffset = m_entries[idx].m_level;
   else
      m_levelOffset = 0;

}

void FWGeometryTableManager::printChildren(int idx) const
{
   //   static double pnt[3];
  std::string path;
  getNodePath(idx, path);
  printf("My %s parent %s path  %d \n",m_entries[idx].name(),path.c_str(), m_entries[idx].m_parent );
  /*
   for (int i =0, k = m_entries.size(); i < k; ++i)
   {
     const  NodeInfo&  ni = m_entries[i];
      if (ni.m_parent == idx)
      {
         std::string indent(ni.m_level*3 , ' ');

         //  printf("%s%s (%.2f, %.2f, %.2f)\n",indent.c_str(), ni.name(), pnt[0], pnt[1], pnt[2]);
         printf("%s%s \n",indent.c_str(), ni.name());

         printChildren(i);
      }
      }*/
}

void FWGeometryTableManager::setDaughtersSelfVisibility(bool v)
{
   int dOff = 0;
   TGeoNode* parentNode = m_entries[m_selectedIdx].m_node;
   int nD = parentNode->GetNdaughters();
   for (int n = 0; n != nD; ++n)
   {
      int idx = m_selectedIdx + 1 + n + dOff;
      NodeInfo& data = m_entries[idx];

      setVisibility(data, v);
      setVisibilityChld(data, v);

      FWGeometryTableManager::getNNodesTotal(parentNode->GetDaughter(n), dOff);
   }
}

//______________________________________________________________________________
void FWGeometryTableManager::getNodeMatrix(const NodeInfo& data, TGeoHMatrix& mtx) const
{
   // utility used by browser and FWGeoNode

   int pIdx  = data.m_parent;
   int endl =  data.m_level -1;
   for (int l = 0 ; l < endl ; ++l)
   {
      pIdx = m_entries.at(pIdx).m_parent;
      mtx.MultiplyLeft(m_entries.at(pIdx).m_node->GetMatrix());
   }

   mtx.Multiply(data.m_node->GetMatrix());
}
//==============================================================================

void FWGeometryTableManager::setVisibility(NodeInfo& data, bool x)
{
   if (m_browser->getVolumeMode())
   {
      if (data.m_node->GetVolume()->IsVisible() != x)
      {
         TEveGeoManagerHolder gmgr( FWGeometryTableViewManager::getGeoMangeur());
         data.m_node->GetVolume()->SetVisibility(x);
      }
   }
   else
   {
      data.setBitVal(kVisNode, x);
   }
}

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
   {
      data.setBitVal(kVisNodeChld, x);
   }
}

//______________________________________________________________________________

bool  FWGeometryTableManager::getVisibility(const NodeInfo& data) const
{
   if (m_browser->getVolumeMode())
      return data.m_node->GetVolume()->IsVisible();
   else
      return  data.testBit(kVisNode);
   
}

bool  FWGeometryTableManager::getVisibilityChld(const NodeInfo& data) const
{
   if (m_browser->getVolumeMode())
      return data.m_node->GetVolume()->IsVisibleDaughters();
   else
      return  data.testBit(kVisNodeChld);
   
}
//==============================================================================

//______________________________________________________________________________

void FWGeometryTableManager::printMaterials()
{
   std::map<TGeoMaterial*, std::string> mlist;
   Entries_i it = m_entries.begin();
   std::advance(it, m_selectedIdx );
   int nLevel = it->m_level;
   it++;
   while (it->m_level > nLevel)
   {
      TGeoMaterial* m = it->m_node->GetVolume()->GetMaterial();
      if (mlist.find(m) == mlist.end())
      {
         mlist[m] = m->GetName();
      } 
      it++;
   }

   printf("size %d \n", (int)mlist.size());
   for(std::map<TGeoMaterial*, std::string>::iterator i = mlist.begin(); i != mlist.end(); ++i)
   {
      printf("material %s \n", i->second.c_str());
   }

}

