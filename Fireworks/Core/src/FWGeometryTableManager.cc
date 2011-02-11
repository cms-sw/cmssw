// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableManager
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Thu Jan 27 14:50:57 CET 2011
// $Id: FWGeometryTableManager.cc,v 1.1.2.4 2011/02/11 19:42:16 amraktad Exp $
//

// system include files

//#define PERFTOOL

// user include files
#include <iostream>
#include <boost/bind.hpp>
#ifdef PERFTOOL 
#include <google/profiler.h>
#endif
#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "Fireworks/Core/interface/FWGeometryTable.h"
#include "Fireworks/Core/src/FWColorBoxIcon.h"
#include "Fireworks/TableWidget/interface/GlobalContexts.h"
#include "Fireworks/TableWidget/src/FWTabularWidget.h"

#include "TMath.h"
#include "TGeoManager.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"
#include "TGeoShape.h"
#include "TGeoBBox.h"


static const char* redTxt   = "\033[01;31m";
static const char* greenTxt = "\033[01;32m";
static const char* cyanTxt  = "\033[22;36m";
//static const char* whiteTxt = "\033[0m";

const char* FWGeometryTableManager::NodeInfo::name() const
{
   return m_node->GetName();
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
   m_colorContext->SetForeground(m_color);
   gVirtualX->FillRectangle(iID, m_colorContext->GetGC(), iX, iY, iWidth, iHeight);

   if (m_isSelected)
   {
     m_colorContext->SetFillStyle(kFillOpaqueStippled);
     gVirtualX->FillRectangle(iID, m_colorContext->GetGC(), iX, iY, iWidth, iHeight);
   }
}

//==============================================================================
//==============================================================================
//
// class FWGeometryTableManager
//
//==============================================================================
//==============================================================================

FWGeometryTableManager::FWGeometryTableManager(FWGeometryTable* browser)
   : m_browser(browser),
     m_geoManager(0),
     m_selectedRow(-1),
     m_maxLevel(0),
     m_maxDaughters(10000)
{ 
   setGrowInWidth(false);

   m_colorBoxRenderer.m_width  =  50;
   m_colorBoxRenderer.m_height =  m_renderer.height();

   m_browser->m_filter.changed_.connect(boost::bind(&FWGeometryTableManager::updateFilter,this));
   m_browser->m_mode.changed_.connect(boost::bind(&FWGeometryTableManager::updateMode,this));
   m_browser->m_maxExpand.changed_.connect(boost::bind(&FWGeometryTableManager::updateMaxExpand,this));
   m_browser->m_maxDepth.changed_.connect(boost::bind(&FWGeometryTableManager::updateMaxDepth,this));
   m_browser->m_maxDaughters.changed_.connect(boost::bind(&FWGeometryTableManager::updateMaxDepth,this));
}

FWGeometryTableManager::~FWGeometryTableManager()
{
}


void FWGeometryTableManager::implSort(int, bool)
{ 
}

void FWGeometryTableManager::runFilter()
{
   printf("=======================================  runFilter[%d]\n", (int)m_entries.size());

   // Decide whether or not items match the filter.
   for (size_t i = 0, e = m_entries.size(); i != e; ++i)
   {
      NodeInfo &data = m_entries[i];
      // First of all decide whether or not we match conditions.
      data.m_matches = true;

    

      if (!strstr(data.m_node->GetVolume()->GetMaterial()->GetName(), m_browser->m_filter.value().c_str()))
         data.m_matches = false;
      else
         data.m_matches = true;
      //  printf("%s maches %d\n", data.name(),  data.m_matches);

   }

   // We reset whether or not a given parent has children that match the
   // filter, and we recompute the whole information by checking all the
   // children.
   for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      m_entries[i].m_childMatches = false;

   std::vector<int> stack;
   int previousLevel = 0;
   for (size_t i = 0, e = m_entries.size(); i != e; ++i)
   {
      NodeInfo &data = m_entries[i];
      // Top level.
      if (data.m_parent == -1)
      {
         previousLevel = 0;
         continue;
      }
      // If the level is greater than the previous one,
      // it means we are among the children of the 
      // previous level, hence we push the parent to
      // the stack.
      // If the level is not greater than the previous
      // one it means we have popped out n levels of
      // parents, where N is the difference between the 
      // new and the old level. In this case we
      // pop up N parents from the stack.
      if (data.m_level > previousLevel)
         stack.push_back(data.m_parent);
      else
         for (size_t pi = 0, pe = previousLevel - data.m_level; pi != pe; ++pi)
            stack.pop_back();
 
      if (data.m_matches)
      {
         for (size_t pi = 0, pe = stack.size(); pi != pe; ++pi)
         {
             m_entries[stack[pi]].m_childMatches = true;
            // printf("%s CHILDmaches %d\n", m_entries[stack[pi]].name(),  m_entries[stack[pi]].m_matches);
         }

      }
      previousLevel = data.m_level;
   }
       
   recalculateVisibility();
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

   if (m_browser->m_mode.value() == FWGeometryTable::kNode )
      returnValue.push_back("Node Name");
   else
      returnValue.push_back("Volume Name");

   returnValue.push_back("Color");
   returnValue.push_back("RnrSelf");
   returnValue.push_back("RnrChildren");
   returnValue.push_back("Material");
   returnValue.push_back("Position");
   returnValue.push_back("Diagonal");

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
//______________________________________________________________________________

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
   if (iRow == m_selectedRow && iColumn == m_selectedColumn)
   {
      m_selectedRow = -1;
      m_selectedColumn = -1;
   }  
   else
   {
      m_selectedRow = iRow;
      m_selectedColumn = iColumn;
   }
   indexSelected_(iRow, iColumn);
   visualPropertiesChanged();
}    

void FWGeometryTableManager::refresh(bool rerunFilters) 
{

   if (rerunFilters) runFilter();

   changeSelection(-1, -1);
   recalculateVisibility();
   dataChanged();
   visualPropertiesChanged();
}

  
FWTableCellRendererBase* FWGeometryTableManager::cellRenderer(int iSortedRowNumber, int iCol) const
{
   if (static_cast<int>(m_row_to_index.size()) <= iSortedRowNumber)
   {
      m_renderer.setData(std::string("FWGeometryTableManager::cellRenderer() Error!"), false);
      return &m_renderer;
   }       

   FWTextTreeCellRenderer* renderer = &m_renderer;
  


   int unsortedRow =  m_row_to_index[iSortedRowNumber];
   const NodeInfo& data = m_entries[unsortedRow];
   if (0)
   {
      TGGC* gc = ( TGGC*)m_renderer.graphicsContext();
  
   if (m_browser->m_mode.value() == FWGeometryTable::kVolume)
      gc->SetForeground(gVirtualX->GetPixel(kGray));
   else
      gc->SetForeground(gVirtualX->GetPixel(kBlack));
   }

   bool isSelected = data.m_imported; // debug
   TGeoNode& gn = *data.m_node;

   if (iCol == kName)
   {
      //   printf("redere\n");
      int nD = getNdaughtersLimited(data.m_node);
      if (m_browser->m_mode.value() == FWGeometryTable::kVolume)
         renderer->setData(Form("%s [%d]", gn.GetVolume()->GetName(), nD), isSelected);
      else    
         renderer->setData(Form("%s [%d]", gn.GetName(), nD ), isSelected); 

      renderer->setIsParent(gn.GetNdaughters() > 0 && ((filterOn() && data.m_childMatches) || !filterOn())) ;
      renderer->setIsOpen(data.m_expanded);
      if (data.m_node->GetNdaughters())
         renderer->setIndentation(10*data.m_level);
      else
         renderer->setIndentation(10*data.m_level + FWTextTreeCellRenderer::iconWidth());

      return renderer;
   }
   else
   {
      // printf("title %s \n",data.m_node->GetTitle() );
      renderer->setIsParent(false);
      renderer->setIndentation(0);
      if (iCol == kColor)
      {
         //renderer->setData(Form("level .. %d", data.m_level),  isSelected);
         m_colorBoxRenderer.setData(gn.GetVolume()->GetLineColor(), isSelected);
         return  &m_colorBoxRenderer;
      }
      else if (iCol == kVisSelf )
      {
         renderer->setData( gn.IsVisible() ? "on" : "off",  isSelected);
         return renderer;
      }
      else if (iCol == kVisChild )
      {
         renderer->setData( gn.IsVisDaughters() ? "on" : "off",  isSelected);
         return renderer;
      }
      else if (iCol == kMaterial )
      { 
         renderer->setData( gn.GetVolume()->GetMaterial()->GetName(),  isSelected);
         return renderer;
      }
      else if (iCol == kPosition )
      { 
         const Double_t* p = gn.GetMatrix()->GetTranslation();
         renderer->setData(Form("[%.3f, %.3f, %.3f]", p[0], p[1], p[2]),  isSelected);
         return renderer;
      }
      else// if (iCol == kPosition  )
      { 
         TGeoBBox* gs = static_cast<TGeoBBox*>( gn.GetVolume()->GetShape());
         renderer->setData( Form("%f", TMath::Sqrt(gs->GetDX()*gs->GetDX() + gs->GetDY()*gs->GetDY() +gs->GetDZ()*gs->GetDZ() )),  isSelected);
         return renderer;
      }
   }
}
//______________________________________________________________________________



void FWGeometryTableManager:: setExpanded(int row)
{
   if (row == -1)
      return;

     
   int idx = rowToIndex()[row];
   // printf("click %s \n", m_entries[idx].name());
   Entries_i it = m_entries.begin();
   std::advance(it, idx);
   NodeInfo& data = *it;
   data.m_expanded = !data.m_expanded;
   if (data.m_expanded  &&  data.m_imported == false)
   {
      importChildren(idx, false);
   }

   recalculateVisibility();
   dataChanged();
   visualPropertiesChanged();
}

void FWGeometryTableManager::recalculateVisibility()
{
   m_row_to_index.clear();

   for ( size_t i = 0,  e = m_entries.size(); i != e; ++i )
   {   
      NodeInfo &data = m_entries[i];
      // printf("visiblity for %s \n", data.m_node->GetName() );
      if (data.m_parent == -1)
      {
         data.m_visible = data.m_childMatches || data.m_matches || !filterOn();
      }
      else 
      {
         data.m_visible = data.m_matches || data.m_childMatches || ( !filterOn() && m_entries[data.m_parent].m_expanded && m_entries[data.m_parent].m_visible);
      }
   }

   // Put in the index only the entries which are visible.
   for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      if (m_entries[i].m_visible)
         m_row_to_index.push_back(i);

   // printf("entries %d \n", m_entries.size());
} 

bool
FWGeometryTableManager::filterOn() const
{
   return !m_browser->m_filter.value().empty();
}

//______________________________________________________________________________

void FWGeometryTableManager::updateMode()
{
   fillNodeInfo(m_geoManager);
}

void FWGeometryTableManager::updateFilter()
{
   refresh(true);
}
void FWGeometryTableManager::updateMaxExpand()
{
   printf("TableManager::updateMaxExpan \n");
   int expL = (int)m_browser->m_maxExpand.value();
   for (Entries_i i = m_entries.begin(); i != m_entries.end(); ++i)
   {
      if ((*i).m_level < TMath::Min( m_maxLevel-1, expL) )
         (*i).m_expanded = true;
      else
         (*i).m_expanded = false;
   }
   refresh();
}
void FWGeometryTableManager::updateMaxDepth()
{
   printf("TableManager::updateMaxDepth \n");
   fillNodeInfo(m_geoManager);
}

//==============================================================================

void FWGeometryTableManager::fillNodeInfo(TGeoManager* geoManager)
{
#ifdef PERFTOOL
   ProfilerStop();
   ProfilerStart(Form("draw_%d_%d", m_maxLevel, m_browser->m_maxExpand.value()));
#endif
   m_geoManager = geoManager;

   m_maxLevel = m_browser->m_maxDepth.value();
   m_maxDaughters =  m_browser->m_maxDaughters.value();

   // clear entries
   if (0) {
      Entries_v x;        m_entries.swap(x);
      std::vector<int> y; m_row_to_index.swap(y);
   }
   else
   {
      m_entries.clear();
      m_row_to_index.clear();
   }
   NodeInfo topNodeInfo;
   topNodeInfo.m_node   = geoManager->GetTopNode()->GetDaughter(0);
   topNodeInfo.m_level  = 0;
   topNodeInfo.m_parent = -1;
   m_entries.push_back(topNodeInfo);

   importChildren(0, true);
   if (1)
   {
      for (Entries_i i = m_entries.begin(); i != m_entries.end(); ++i)
         if ((*i).m_level < TMath::Min( m_maxLevel-1, (int)m_browser->m_maxExpand.value()) )
            (*i).m_expanded = true;
   }

   // checkHierarchy(); // debug
   printf("size %d \n", m_entries.size());
   refresh(filterOn());
}

//==============================================================================


void
FWGeometryTableManager::getNVolumesTotal(TGeoNode* geoNode, int level, int& off, bool) const
{
   int nD =  getNdaughtersLimited(geoNode);
   std::vector<int> vi; vi.reserve(nD);
   vi.reserve(nD);
   for (int n = 0; n != nD; ++n)
   {
      if (strstr(geoNode->GetDaughter(n)->GetName(), "_1"))
         vi.push_back(n);
   }

   int nV = vi.size();
   if (level <  (m_maxLevel))
   {
      off += nV;
      for (int i = 0; i < nV; ++i )
      {
         getNVolumesTotal(geoNode->GetDaughter(vi[i]), level+1, off, false);
      }
   }
}

void FWGeometryTableManager::importChildren(int parent_idx, bool recurse)
{
   if (m_browser->m_mode.value() == FWGeometryTable::kNode)
      importChildNodes(parent_idx, recurse);
   else
      importChildVolumes(parent_idx, recurse);

}
//______________________________________________________________________________

void FWGeometryTableManager::importChildVolumes(int parent_idx, bool recurse)
{ 
   bool debug = false;

   // printf("importing from parent %d entries size %d \n",  parent_idx, m_entries.size());
   NodeInfo& parent  = m_entries[parent_idx];
   TGeoNode* geoNode = parent.m_node; 
   parent.m_imported = true;

   // get indices of nodes with unique volumes
   int nD = getNdaughtersLimited(geoNode);
   std::vector<int> vi; 
   vi.reserve(nD);
   for (int n = 0; n != nD; ++n)
   {
      if (strstr(geoNode->GetDaughter(n)->GetName(), "_1"))
      {
         vi.push_back(n);
      }
   }

   int nV =  vi.size();

   // add nodes with unique volumes
   Entries_i it = m_entries.begin();
   std::advance(it, parent_idx+1);
   m_entries.insert(it, nV, NodeInfo());
   // setup
   for (int n = 0; n != nV; ++n)
   {
      int childIdx = vi[n];
      NodeInfo &nodeInfo = m_entries[parent_idx + n + 1];
      nodeInfo.m_node =   geoNode->GetDaughter(childIdx);
      nodeInfo.m_level =  parent.m_level + 1;
      nodeInfo.m_parent = parent_idx;
      if (debug)  printf(" add %s\n", nodeInfo.name());
   }

  
   // recurse
   if (recurse)
   {
      int dOff = 0;
      if ((parent.m_level+1) < m_maxLevel)
      {
         for (int n = 0; n != nV; ++n)
         {
            int childIdx = vi[n];
            importChildVolumes(parent_idx + n + 1 + dOff, recurse);       
            if (geoNode->GetNdaughters() > 0)
               getNVolumesTotal(geoNode->GetDaughter(childIdx), parent.m_level+1, dOff, debug);

         }
      }
   }
}

//______________________________________________________________________________


void
FWGeometryTableManager::getNNodesTotal(TGeoNode* geoNode, int level, int& off, bool debug) const
{
   if (debug) printf("%s %s (c:%d)\033[22;0m ", cyanTxt, geoNode->GetName(), level);

   int nD = getNdaughtersLimited(geoNode);
   if (level <  (m_maxLevel))
   {
      off += nD;
      for (int i = 0; i < nD; ++i )
      {
         getNNodesTotal(geoNode->GetDaughter(i), level+1, off, debug);
      }
      if (debug) printf("%d \n", off);
   }
}

void FWGeometryTableManager::importChildNodes(int parent_idx, bool recurse)
{  
   int nEntries = (int)m_entries.size();
   assert( parent_idx < nEntries);
 
   bool debug = false;


   // printf("importing from parent %d entries size %d \n",  parent_idx, m_entries.size());
   NodeInfo& parent  = m_entries[parent_idx];
   parent.m_imported = true;

   TGeoNode* parentGeoNode = parent.m_node; 
   int       parentLevel   = parent.m_level;

   if (debug) printf("%s START level[%d] >  %s[%d]   \033[0m\n" ,greenTxt,  parentLevel+1, parentGeoNode->GetName(), parent_idx);
   // add children
   int nD = getNdaughtersLimited(parentGeoNode);
   Entries_i it = m_entries.begin();
   std::advance(it, parent_idx+1);
   m_entries.insert(it, nD, NodeInfo());
   for (int n = 0; n != nD; ++n)
   {
      NodeInfo &nodeInfo = m_entries[parent_idx + n + 1];
      nodeInfo.m_node    =   parentGeoNode->GetDaughter(n);
      nodeInfo.m_level   =  parentLevel + 1;




      nodeInfo.m_parent  = parent_idx;
      if (debug)  printf(" add %s\n", nodeInfo.name());
   }
   
   // shift parent indices for array succesors ....
   if (debug)  printf("\ncheck shhift for level  evel %d  import %s ", parent.m_level +1,parentGeoNode->GetName() ); 
   for (int i = (parent_idx + nD + 1); i < nEntries; ++i)
   {
      if (m_entries[i].m_parent > m_entries[parent_idx].m_parent)
      {
         if (debug)  printf("%s %s", redTxt,  m_entries[i].name());       
         m_entries[i].m_parent +=  nD;

      }
   }
   if (debug) printf(" \033[0m\n");

   // recurse
   if (recurse)
   {
      int dOff = 0;
      if ((parentLevel+1) < m_maxLevel)
      {
         for (int n = 0; n != nD; ++n)
         {
            if (debug)  printf("begin [%d]to... recursive import of daughter %d %s parent-index-offset %d\n",parent.m_level+1, n, parentGeoNode->GetDaughter(n)->GetName(), dOff );
            importChildNodes(parent_idx + n + 1 + dOff, recurse);
            if (debug)   printf("end [%d]... recursive import of daughter %d %s parent-index-offset %d\n\n\n", parent.m_level+1, n, parentGeoNode->GetDaughter(n)->GetName(), dOff );
       
            getNNodesTotal(parentGeoNode->GetDaughter(n), parentLevel+1, dOff, debug);
            if (debug)   printf("\n");
         }
      }
   }
   if (debug) printf("\n%s END level[%d] >  %s[%d]   \033[0m\n" ,greenTxt,  parentLevel+1, parentGeoNode->GetName(), parent_idx);
   fflush(stdout);
}// end importChildNodes

//==============================================================================

void FWGeometryTableManager::checkHierarchy()
{
   // function only for debug purposes 

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
