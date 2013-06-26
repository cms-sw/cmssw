// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableManagerBase
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel, Matevz Tadel
//         Created:  Thu Jan 27 14:50:57 CET 2011
// $Id: FWGeometryTableManagerBase.cc,v 1.8 2013/04/14 20:39:19 amraktad Exp $
//

//#define PERFTOOL_GEO_TABLE

// user include files
#include <iostream>
#include <boost/bind.hpp>
#include <stack>
#ifdef PERFTOOL_GEO_TABLE 
#include <google/profiler.h>
#endif
#include "Fireworks/Core/interface/FWGeometryTableManagerBase.h"
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

#include "TGFrame.h"
#include "TEveUtil.h"
#include "boost/lexical_cast.hpp"


const char* FWGeometryTableManagerBase::NodeInfo::name() const
{
   return m_node->GetName();
}


FWGeometryTableManagerBase::ColorBoxRenderer::ColorBoxRenderer():
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

FWGeometryTableManagerBase::ColorBoxRenderer::~ColorBoxRenderer()
{
   gClient->GetResourcePool()->GetGCPool()->FreeGC(m_colorContext->GetGC());
}

void FWGeometryTableManagerBase::ColorBoxRenderer::setData(Color_t c, bool s)
{
   m_color = gVirtualX->GetPixel(c);
   m_isSelected = s;
}


void FWGeometryTableManagerBase::ColorBoxRenderer::draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
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
// class FWGeometryTableManagerBase
//
//==============================================================================
//==============================================================================

FWGeometryTableManagerBase::FWGeometryTableManagerBase()
   :   
   m_highlightIdx(-1),
   m_levelOffset(0),
   m_editor(0),
   m_editTransparencyIdx(-1)
{ 
   m_colorBoxRenderer.m_width  =  50;
   m_colorBoxRenderer.m_height =  m_renderer.height();

   GCValues_t gval;
   gval.fMask = kGCForeground | kGCBackground | kGCStipple | kGCFillStyle  | kGCGraphicsExposures;
   gval.fForeground = gVirtualX->GetPixel(kGray);//gClient->GetResourcePool()->GetFrameHiliteColor();
   gval.fBackground = gVirtualX->GetPixel(kWhite);//gClient->GetResourcePool()->GetFrameBgndColor();
   gval.fFillStyle  = kFillOpaqueStippled; // kFillTiled;
   gval.fStipple    = gClient->GetResourcePool()->GetCheckeredBitmap();
   gval.fGraphicsExposures = kFALSE;
   m_highlightContext = gClient->GetGC(&gval, kTRUE);

   m_renderer.setHighlightContext( m_highlightContext);
}

FWGeometryTableManagerBase::~FWGeometryTableManagerBase()
{
}


int FWGeometryTableManagerBase::unsortedRowNumber(int unsorted) const
{
   return unsorted;
}

int FWGeometryTableManagerBase::numberOfRows() const 
{
   return m_row_to_index.size();
}


std::vector<std::string> FWGeometryTableManagerBase::getTitles() const 
{
   std::vector<std::string> returnValue;
   returnValue.reserve(numberOfColumns());

   returnValue.push_back("Name");
   returnValue.push_back("Color");
   returnValue.push_back("Opcty");
   returnValue.push_back("RnrSelf");
   returnValue.push_back("RnrChildren");
   returnValue.push_back("Material");
   return returnValue;
}
 
const std::string FWGeometryTableManagerBase::title() const 
{
   return "Geometry";
}


void  FWGeometryTableManagerBase::setBackgroundToWhite(bool iToWhite )
{
   if(iToWhite) {
      m_renderer.setGraphicsContext(&TGFrame::GetBlackGC());
   } else {
      m_renderer.setGraphicsContext(&TGFrame::GetWhiteGC());
   }
   m_renderer.setBlackIcon(iToWhite);
}

//______________________________________________________________________________
bool FWGeometryTableManagerBase::firstColumnClicked(int row, int xPos)
{
   if (row == -1)
      return false;

   int idx = rowToIndex()[row];
   // printf("click %s \n", m_entries[idx].name());

   int off = 0;
   if (idx >= 0)
      off = (m_entries[idx].m_level - m_levelOffset)* 20;

   //   printf("compare %d %d level %d\n" , xPos, off, idx);
   if (xPos >  off &&  xPos < (off + 20))
   {
      m_entries[idx].switchBit(kExpanded);
 
      recalculateVisibility();
      dataChanged();
      visualPropertiesChanged();
      return false;
   }

   return true;
}


 

//______________________________________________________________________________

void FWGeometryTableManagerBase::getNodeMatrix(const NodeInfo& data, TGeoHMatrix& mtx) const
{
   // utility used by browser and FWGeoNode
   //   printf("================ FWGeometryTableManagerBase::getNodeMatri \n");
   int pIdx  = data.m_parent;

   while (pIdx > 0)
   {
      // printf("%s [%d]\n",m_entries.at(pIdx).name(), m_entries.at(pIdx).m_level );
      mtx.MultiplyLeft(m_entries.at(pIdx).m_node->GetMatrix());
      pIdx = m_entries.at(pIdx).m_parent;
   }

   //   printf("right %s [%d]\n",data.name(), data.m_level );
   mtx.Multiply(data.m_node->GetMatrix());
}

//______________________________________________________________________________
void FWGeometryTableManagerBase::redrawTable(bool setExpand) 
{
   //   std::cerr << "GeometryTableManagerBase::redrawTable ------------------------------------- \n";
   if (m_entries.empty()) return;

   //   if (setExpand) checkExpandLevel();

   recalculateVisibility();


   dataChanged();
   visualPropertiesChanged();
}


//______________________________________________________________________________

void FWGeometryTableManagerBase::getNodePath(int idx, std::string& path) const
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


void FWGeometryTableManagerBase::setCellValueEditor(TGTextEntry *editor)
{
   m_editor = editor;
   m_renderer.setCellEditor(m_editor);
}

void FWGeometryTableManagerBase::showEditor(int row)
{
   m_editTransparencyIdx = row;
   m_editor->UnmapWindow();
   m_editor->SetText(Form("%d", 100 - m_entries[row].m_transparency));
   m_editor->Resize(40, 17);
   m_editor->SetCursorPosition(2);
   redrawTable();
}



void FWGeometryTableManagerBase::applyTransparencyFromEditor()
{
   printf("transparency idx %d opaci %s \n",m_editTransparencyIdx, m_editor->GetText() );
   if ( m_editTransparencyIdx >= 0)
   {
      using boost::lexical_cast;
      using boost::bad_lexical_cast;
      try {
         int t = lexical_cast<int>(m_editor->GetText());
         if (t > 100 || t < 0 )
         {
            fwLog(fwlog::kError) << "Transparency must be set in procentage [0-100].";
            return;
         }
         m_entries[m_editTransparencyIdx].m_transparency = 100 - t;
         printf("SET !! \n");
         cancelEditor(true);
      }
      catch (bad_lexical_cast &) {
         fwLog(fwlog::kError) << "Bad Lexical cast. Transparency must be set in procentage [0-100].";
      }
   }
}

void FWGeometryTableManagerBase::cancelEditor(bool redraw)
{
   m_editTransparencyIdx = -1;

   if ( m_editor->IsMapped())
   {
      m_editor->UnmapWindow(); 
      if (redraw) redrawTable();
   }
}


//------------------------------------------------------------------------------

void FWGeometryTableManagerBase::setVisibility(NodeInfo& data, bool x)
{
   data.setBitVal(kVisNodeSelf, x);
}

//------------------------------------------------------------------------------

void FWGeometryTableManagerBase::setVisibilityChld(NodeInfo& data, bool x)
{
   data.setBitVal(kVisNodeChld, x);
}
//______________________________________________________________________________

void FWGeometryTableManagerBase::setDaughtersSelfVisibility(int selectedIdx, bool v)
{
   TGeoNode  *parentNode = m_entries[selectedIdx].m_node;
   int nD   = parentNode->GetNdaughters();
   int dOff = 0;
   for (int n = 0; n != nD; ++n)
   {
      int idx = selectedIdx + 1 + n + dOff;
      NodeInfo& data = m_entries[idx];
      
      setVisibility(data, v);
      setVisibilityChld(data, v);
      
      getNNodesTotal(parentNode->GetDaughter(n), dOff);
   }
}

//------------------------------------------------------------------------------

bool FWGeometryTableManagerBase::getVisibility(const NodeInfo& data) const
{
   return data.testBit(kVisNodeSelf);
}

bool FWGeometryTableManagerBase::getVisibilityChld(const NodeInfo& data) const
{
   return data.testBit(kVisNodeChld);
}


bool FWGeometryTableManagerBase::isNodeRendered(int idx, int topNodeIdx) const
{
   const NodeInfo& data = m_entries[idx];
   bool foundParent = false;

   if (data.testBit(kVisNodeSelf))
   {
      int pidx = data.m_parent;
      while (pidx >= 0 ) 
      {
         if (!m_entries[pidx].testBit(kVisNodeChld)) {
            // printf ("parent disallow not visible !!! \n");
            return false;
         }

         if (pidx == topNodeIdx) { foundParent = true; 
            // printf("parent found \n"); 
            break;
         }
         pidx = m_entries[pidx].m_parent;
      }

      return foundParent;
   }
   return false;
}
