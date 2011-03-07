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
// $Id: FWGeometryTableManager.cc,v 1.7 2011/02/14 22:27:59 amraktad Exp $
//

//#define PERFTOOL

// user include files
#include <iostream>
#include <boost/bind.hpp>
#include <stack>
#ifdef PERFTOOL 
#include <google/profiler.h>
#endif
#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "Fireworks/Core/interface/FWGeometryBrowser.h"
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

FWGeometryTableManager::FWGeometryTableManager(FWGeometryBrowser* browser)
:    m_selectedRow(-1),
     m_browser(browser),
     m_geoManager(0),
     m_autoExpand(0),
     m_maxDaughters(10000),
     m_modeVolume(false)
{ 
   m_colorBoxRenderer.m_width  =  50;
   m_colorBoxRenderer.m_height =  m_renderer.height();
   
   m_browser->m_mode.changed_.connect(boost::bind(&FWGeometryTableManager::updateMode,this));
   m_browser->m_autoExpand.changed_.connect(boost::bind(&FWGeometryTableManager::updateAutoExpand,this));
   m_browser->m_maxDaughters.changed_.connect(boost::bind(&FWGeometryTableManager::updateAutoExpand,this));
   m_browser->m_filter.changed_.connect(boost::bind(&FWGeometryTableManager::updateFilter,this));
   
   // debug
   m_browser->m_maxDaughters.changed_.connect(boost::bind(&FWGeometryTableManager::updateAutoExpand,this));
   
   // init here if no config
   m_modeVolume   =  m_browser->m_mode.value();
   m_autoExpand   =  m_browser->m_autoExpand.value();
   m_maxDaughters =  m_browser->m_maxDaughters.value();
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

   if (m_modeVolume )
      returnValue.push_back("Volume Name");
   else
      returnValue.push_back("Node Name");

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
   if (iRow == m_selectedRow )
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
   TGeoNode& gn = *data.m_node;

   bool isSelected =  (!filterOff() &&  m_volumes[gn.GetVolume()].m_matches);//(m_selectedRow == unsortedRow);
   if (0) 
   {
      TGGC* gc = ( TGGC*)m_renderer.graphicsContext();
  
      if (!filterOff() &&  m_volumes[gn.GetVolume()].m_matches)
         gc->SetForeground(gVirtualX->GetPixel(kRed));
      else
         gc->SetForeground(gVirtualX->GetPixel(kBlack));
   }

   if (iCol == kName)
   {
      //   printf("redere\n");
      int nD = getNdaughtersLimited(data.m_node);
      if (m_modeVolume)
         renderer->setData(Form("%s [%d]", gn.GetVolume()->GetName(), nD), isSelected);
      else    
         renderer->setData(Form("%s [%d]", gn.GetName(), nD ), isSelected); 

      renderer->setIsParent((gn.GetNdaughters() > 0) && (filterOff() || m_volumes[gn.GetVolume()].accepted()));
      // printf("%s isParent %d\n", gn.GetVolume()->GetName(), isParent);

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
void FWGeometryTableManager::firstColumnClicked(int row)
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
         data.m_visible = true;
      }
      else 
      {
         data.m_visible = m_entries[data.m_parent].m_expanded && m_entries[data.m_parent].m_visible;
      }
   }

   // Put in the index only the entries which are visible.
   for (size_t i = 0, e = m_entries.size(); i != e; ++i)
      if (m_entries[i].m_visible)
         m_row_to_index.push_back(i);

   // printf("entries %d \n", m_entries.size());
} 


void FWGeometryTableManager::redrawTable() 
{
   changeSelection(-1, -1);
   recalculateVisibility();
   dataChanged();
   visualPropertiesChanged();
}



//==============================================================================

void FWGeometryTableManager::checkUniqueVolume(TGeoVolume* v)
{
   Volumes_i it  = m_volumes.find(v);
   if (it == m_volumes.end())
   {
      m_volumes.insert(std::make_pair(v, Match()));
   }
   for (int i =0, nD = v->GetNdaughters(); i != nD; ++i) {
      checkUniqueVolume(v->GetNode(i)->GetVolume());
   }
}

void FWGeometryTableManager::loadGeometry(TGeoManager* geoManager)
{
   m_geoManager = geoManager;
   
   m_volumes.clear();
   checkUniqueVolume(geoManager->GetTopVolume());
   if (!filterOff())
      updateFilter();
   
   m_browser->updateStatusBar(Form("FWGeometryTableManager::loadGeometry() %d unique volumes", (int)m_volumes.size()));
   
   setTableContent();
}
//______________________________________________________________________________


void FWGeometryTableManager::setTableContent()
{
   // Prepare data for cell render.
  
   m_browser->updateStatusBar("Set table content ...");

#ifdef PERFTOOL  
   if (filterOff())
      ProfilerStart(Form("SetTableContent.expand_%d.modeV_%d", m_browser->m_autoExpand.value(), ( m_browser->m_mode.value() == FWGeometryBrowser::kVolume)));
   else  
      ProfilerStart(Form("SetTableContent.expand_%d.modeV_%d.filter_%s", m_browser->m_autoExpand.value(), ( m_browser->m_mode.value() == FWGeometryBrowser::kVolume),  m_browser->m_filter.value().c_str()));

#endif
   bool debug = 1;
   
   // clear entries
   m_entries.clear();
   m_row_to_index.clear();
   
   // cache browser values   
   m_autoExpand   = m_browser->m_autoExpand.value();
   m_maxDaughters =  m_browser->m_maxDaughters.value();
   m_modeVolume   =  m_browser->m_mode.value() == FWGeometryBrowser::kVolume;

   // add top node to init
   NodeInfo topNodeInfo;
   topNodeInfo.m_node   = m_geoManager->GetTopNode();
   topNodeInfo.m_level  = 0;
   topNodeInfo.m_parent = -1;
   m_entries.push_back(topNodeInfo);

   importChildren(0, true);
   
   if (debug)
      checkHierarchy();
 
   redrawTable();
   
#ifdef PERFTOOL  
   ProfilerStop();
#endif

   if (filterOff())
   {
      m_browser->updateStatusBar(Form("%d entries imported ", (int)m_entries.size()));
   }
   else
   {
      {
         // get status
         int na = 0;
         int n = 0;
         for (Volumes_i i = m_volumes.begin(); i!= m_volumes.end(); ++i) 
         {
            n++;
            if ( i->second.m_matches)
            {
               na++;
               // printf("[%d] %s matches material %s \n", na, i->first->GetName(), i->first->GetMaterial()->GetName());
            }
         }

         m_browser->updateStatusBar(Form("%d entries imported, filter: %d volumes (%.2f %%) selected ", (int)m_entries.size(), na, na*1.f/n));
      }
   }
}

//==============================================================================



void
FWGeometryTableManager::getNNodesTotal(TGeoNode* geoNode, int level, int& off, bool debug) const
{   
   // Get number of nested children recursively.
   
   if (debug) printf("getNNodesTotal %s %s (c:%d)\033[22;0m \n", cyanTxt, geoNode->GetName(), level);
   
   int nD =  getNdaughtersLimited(geoNode);
   std::vector<int> vi; vi.reserve(nD);
   vi.reserve(nD);
   for (int n = 0; n != nD; ++n)
   {
      TGeoVolume* vTmp = geoNode->GetDaughter(n)->GetVolume();
      if (m_volumes[vTmp].accepted())
      {
         bool toAdd = true;
         if (m_modeVolume)
         {
            for (std::vector<int>::iterator u = vi.begin(); u != vi.end(); ++u )
            {
               TGeoVolume* neighbourVolume = geoNode->GetDaughter(*u)->GetVolume();
               if (neighbourVolume == vTmp)
               {
                  toAdd = false;
                  break;
               }
            }
         } // end volume mode
         if (toAdd) vi.push_back(n);
      }
   }
   
   int nV = vi.size();
   if (level <  m_autoExpand)
   {
      off += nV;
      for (int i = 0; i < nV; ++i )
      {
         getNNodesTotal(geoNode->GetDaughter(vi[i]), level+1, off, false);
      }
      if (debug) printf("%d \n", off);
   }
}

void FWGeometryTableManager::importChildren(int parent_idx, bool recurse)
{
   bool debug = false;
   
   int nEntries = (int)m_entries.size();
   assert( parent_idx < nEntries);
 
   // parnt index not valid in recursive import:  save parent info here
   NodeInfo& parent        = m_entries[parent_idx];
   TGeoNode* parentGeoNode = parent.m_node; 
   int       parentLevel   = parent.m_level;   
   if (debug) printf("%s START level[%d] >  %s[%d]   \033[0m\n" ,greenTxt,  parentLevel+1, parentGeoNode->GetName(), parent_idx);

   parent.m_expanded = true;
   
   // get indices of accepted nodes
   int nD = getNdaughtersLimited(parentGeoNode);
   std::vector<int> vi; 
   vi.reserve(nD);
   TGeoVolume* vTmp;

   for (int n = 0; n != nD; ++n)
   {
      vTmp = parentGeoNode->GetDaughter(n)->GetVolume();
      
      if (filterOff() || m_volumes[vTmp].accepted())
      {
         bool toAdd = true;
         if (m_modeVolume)
         {
            // check duplicates in added
            for (std::vector<int>::iterator u = vi.begin(); u != vi.end(); ++u )
            {
               TGeoVolume* neighbourVolume =  parentGeoNode->GetDaughter(*u)->GetVolume();
               if (neighbourVolume == vTmp)
               {
                  toAdd = false;
                  break;
               }
            }
         } // end volume mode
         if (toAdd) vi.push_back(n);         
      } // end checke filters
      
      
   } // end daughter loop
   int nV =  vi.size();
   
   // add  accepted nodes
   Entries_i it = m_entries.begin();
   std::advance(it, parent_idx+1);
   m_entries.insert(it, nV, NodeInfo());
   nEntries += nV; 
   if (debug)  printf(" accpted %d of %d entries size %d \n", nV, nD, (int)m_entries.size());
   
   // child nodes setup
   for (int n = 0; n != nV; ++n)
   {
      int childIdx = vi[n];
      NodeInfo &nodeInfo = m_entries[parent_idx + 1 + n ];
      nodeInfo.m_node =   parentGeoNode->GetDaughter(childIdx);
      nodeInfo.m_level =  parent.m_level + 1;
      nodeInfo.m_parent = parent_idx;
      if (debug)  printf(" add %s\n", nodeInfo.name());
   }
   
   if (recurse)
   {
      // change of autoExpand parameter
      int dOff = 0;
      if ((parent.m_level+1) < m_autoExpand)
      {
         for (int n = 0; n != nV; ++n)
         {
            importChildren(parent_idx + n + 1 + dOff, recurse);       
            if (parentGeoNode->GetNdaughters() > 0)
            {
               getNNodesTotal(parentGeoNode->GetDaughter(vi[n]), parentLevel+1, dOff, debug);
            }
            
         }
      }
   }
   else
   {  
      // expand on double-click, possibly shift parents
      if (debug)  printf("\ncheck shhift for level  evel %d  import %s ", parent.m_level +1,parentGeoNode->GetName() ); 
      
      for (int i = (parent_idx + nV + 1); i < nEntries; ++i)
      {
         if (m_entries[i].m_parent > m_entries[parent_idx].m_parent)
         {
            if (debug)  printf("%s %s", redTxt,  m_entries[i].name());       
            m_entries[i].m_parent +=  nV;
            
         }
      }      
      if (debug) printf(" \033[0m\n");
   }
   
   
   fflush(stdout);
}// end importChildren

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
         //  if (0 && pm.m_childMatches)
         //   break;

         pm.m_childMatches = true;         
      }
   }

   pstack.push_back(vol);

   int nD = TMath::Min(m_maxDaughters, vol->GetNdaughters());
   for (int i = 0; i!=nD; ++i)
      checkChildMatches(vol->GetNode(i)->GetVolume(), pstack);
   
   pstack.pop_back();
}

// callbacks ______________________________________________________________________________

void FWGeometryTableManager::updateFilter()
{
   if (!m_geoManager) return;
   
#ifdef PERFTOOL
   ProfilerStart(m_browser->m_filter.value().c_str());
#endif

   for (Volumes_i i = m_volumes.begin(); i!= m_volumes.end(); ++i) 
   {
      i->second.m_matches = m_browser->m_filter.value().empty() || strstr(i->first->GetMaterial()->GetName(), m_browser->m_filter.value().c_str());
      i->second.m_childMatches = false;
   }  
  
   std::vector<TGeoVolume*> pstack;
   checkChildMatches(m_geoManager->GetTopVolume(), pstack);
 
   //   printf("filterChanged \n");
#ifdef PERFTOOL
   ProfilerStop();
#endif
   setTableContent();
}

void FWGeometryTableManager::updateAutoExpand()
{
   if (!m_geoManager) return;
   
   setTableContent();
}

void FWGeometryTableManager::updateMode()
{
   if (!m_geoManager) return;
   
   setTableContent();
}

bool FWGeometryTableManager::filterOff() const
{
  return  m_browser->m_filter.value().empty();
  //   printf("%d empty off \n", m_browser->m_filter.value().empty());
}
