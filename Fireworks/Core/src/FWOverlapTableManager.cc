// -*- C++ -*-
//
// Package:     Core
// Class  :     FWOverlapTableManager
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Wed Jan  4 20:31:32 CET 2012
// $Id: FWOverlapTableManager.cc,v 1.9 2013/04/14 20:41:06 amraktad Exp $
//

// system include files

// user include files
#include "Fireworks/Core/src/FWOverlapTableManager.h"
#include "Fireworks/Core/src/FWOverlapTableView.h"
#include "Fireworks/Core/src/FWEveDigitSetScalableMarker.cc"
#include "Fireworks/Core/interface/FWGeometryTableViewManager.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "TEveQuadSet.h"
#include "TGeoVolume.h"
#include "TGeoMatrix.h"
#include "TGeoShape.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TEveUtil.h"
#include "TObjString.h"
#include "TGeoNode.h"
#include "TGeoOverlap.h"
#include "TGeoManager.h"
#include "TPolyMarker3D.h"

#include "TStopwatch.h"
#include "TTimer.h"
#include "TGeoPainter.h"
#include "TPRegexp.h"

FWOverlapTableManager::FWOverlapTableManager(FWOverlapTableView* v ):
   FWGeometryTableManagerBase(),
   m_browser(v)
{
}

FWOverlapTableManager::~FWOverlapTableManager()
{
}



std::vector<std::string> FWOverlapTableManager::getTitles() const 
{
   std::vector<std::string> returnValue;
   returnValue.reserve(numberOfColumns());

   returnValue.push_back("Name");
   returnValue.push_back("Color");
   returnValue.push_back("Opcty");
   returnValue.push_back("RnrSelf");
   returnValue.push_back("RnrChildren");
   returnValue.push_back("Overlap");
   returnValue.push_back("RnrMarker");
   return returnValue;
}




//---------------------------------------------------------------------------------
void FWOverlapTableManager::importOverlaps(std::string iPath, double iPrecision)
{
   m_entries.clear();
   m_mapNodeOverlaps.clear();
   m_browser->getMarker()->Reset(TEveQuadSet::kQT_FreeQuad, kFALSE, 32 );

   TEveGeoManagerHolder mangeur( FWGeometryTableViewManager::getGeoMangeur());
   // gGeoManager->cd();
   NodeInfo topNodeInfo;
   topNodeInfo.m_node   = gGeoManager->GetTopNode();
   topNodeInfo.m_level  = 0;
   topNodeInfo.m_color  = gGeoManager->GetTopNode()->GetVolume()->GetLineColor();
   topNodeInfo.m_transparency  = gGeoManager->GetTopNode()->GetVolume()->GetTransparency();
   topNodeInfo.m_parent = -1;
   topNodeInfo.resetBit(kVisNodeSelf);

   m_entries.resize(gGeoManager->GetNNodes());
   m_entries[0] = topNodeInfo;

   m_entries.resize( gGeoManager->GetNNodes());
  
   TGeoVolume* topVol =  topNodeInfo.m_node->GetVolume();
   Int_t icheck = 0;
   Int_t ncheck = 0;
   TStopwatch *timer;
   Int_t i;  
   bool checkingOverlaps = false;
   TGeoManager *geom = topVol->GetGeoManager();
   ncheck = topNodeInfo.m_node->CountDaughters(kFALSE);
   timer = new TStopwatch();
   geom->ClearOverlaps();
   geom->SetCheckingOverlaps(kTRUE);

   int oldS = 0;
   timer->Start();
   geom->GetGeomPainter()->OpProgress(topVol->GetName(),icheck,ncheck,timer,kFALSE);
//   topVol->CheckOverlaps(iPrecision);
   icheck++;
   TGeoIterator git(topVol);
   Entries_i eit = m_entries.begin();
   /*
     if (gGeoManager->GetListOfOverlaps()->GetEntriesFast()) {
     int newCnt =  gGeoManager->GetListOfOverlaps()->GetEntriesFast();
     for (int i=0; i<newCnt; ++i) {
     addOverlapEntry((TGeoOverlap*)gGeoManager->GetListOfOverlaps()->At(i), new TGeoHMatrix(*geom->GetCurrentMatrix()), topNode, next); 
     }
     oldS= newCnt;
     }*/
   eit++;
   TGeoNode *node;
   icheck = 1;
  
   int topNodeIdx =  m_browser->getTopNodeIdx();

   while ((node=git())) {
      if (!eit->testBit(kOverlap)) eit->resetBit(kVisNodeSelf);
      eit->m_node = node;
      eit->m_color = node->GetVolume()->GetLineColor();
      eit->m_transparency = node->GetVolume()->GetTransparency();
      eit->m_level = git.GetLevel();
      eit->m_parent = icheck;
     
     if ((topNodeIdx )== icheck || !topNodeIdx  ) { 
     //  printf("start to check overlaps on topNodeIdx %s \n", eit->name());
       checkingOverlaps=true;
     }
     else if (checkingOverlaps && ( eit->m_level <= m_entries[m_browser->getTopNodeIdx()].m_level)) 
     {
       checkingOverlaps=false;
     }
      // parent index
      Entries_i pit = eit;
      do 
      {
         --pit;
         --(eit->m_parent);
         if (pit->m_level <  eit->m_level) 
            break;
      } while (pit !=  m_entries.begin());

      // overlap bits
      if ( checkingOverlaps) {
         if (!node->GetVolume()->IsSelected()) {
            geom->GetGeomPainter()->OpProgress(node->GetVolume()->GetName(),icheck+1,ncheck,timer,kFALSE);
            node->GetVolume()->SelectVolume(kFALSE);

            node->GetVolume()->CheckOverlaps(iPrecision);

            if (oldS !=  gGeoManager->GetListOfOverlaps()->GetEntriesFast()) {
              // printf("mother %s overlaps \n", node->GetName());
            
              eit->setBit(kOverlapChild);
              eit->setBit(kVisNodeChld);              
              eit->setBit(kVisMarker);
              
               TGeoHMatrix* motherm = new TGeoHMatrix(*geom->GetCurrentMatrix());        
               {
                  TGeoNode* ni = topNodeInfo.m_node;
                  for (Int_t i=1; i<=git.GetLevel(); i++) {
                     ni = ni->GetDaughter(git.GetIndex(i));
                     motherm->Multiply(ni->GetMatrix());
                  }
               }
        
               int newCnt =  gGeoManager->GetListOfOverlaps()->GetEntriesFast();       
          
               for (int i=oldS; i<newCnt; ++i)
               {
                  //                  printf("add %p %p \n", (void*)node->GetVolume(), (void*)m_entries[icheck].m_node->GetVolume());
                  addOverlapEntry((TGeoOverlap*)gGeoManager->GetListOfOverlaps()->At(i), i, icheck, motherm); 
               }
            
               oldS = newCnt;
            } 
         }   
      }
      eit++; 
      icheck ++;    
   } 

   m_browser->getMarker()->RefitPlex();
  
   topVol->SelectVolume(kTRUE);
   geom->SetCheckingOverlaps(kFALSE);
   //   geom->SortOverlaps();
   TObjArray *overlaps = geom->GetListOfOverlaps();
   Int_t novlps = overlaps->GetEntriesFast();     
   TNamed *obj;
   for (i=0; i<novlps; i++) {
      obj = (TNamed*)overlaps->At(i);
      obj->SetName(Form("ov%05d",i));
   }
   geom->GetGeomPainter()->OpProgress("Check overlaps:",icheck,ncheck,timer,kTRUE);
   Info("CheckOverlaps", "Number of illegal overlaps/extrusions : %d\n", novlps);
   delete timer;
}


//______________________________________________________________________________


void FWOverlapTableManager::addOverlapEntry(TGeoOverlap* ovl, int ovlIdx,  Int_t parentIdx, TGeoHMatrix* motherm)
{     

   // printf("add %s \n", ovl->GetTitle());
   // get doughter indices of overlaps
   /* 
      TPMERegexp re(" ", "o");
      re.Split(TString(ovl->GetTitle()));
      printf("add title %s \n", ovl->GetTitle());
   */
   int pcnt = parentIdx+1;
   int dOff =0;
   TGeoNode* mothern = m_entries[parentIdx].m_node;

   QuadId* quid = new QuadId(ovl, parentIdx);

   for (int i = 0; i < mothern->GetNdaughters(); ++i)
   {
      TGeoNode* n = mothern->GetDaughter(i);

      int  cnt = pcnt + i+dOff;

      if (ovl->IsOverlap()) { 
         if (n->GetVolume() == ovl->GetFirstVolume() && (*(ovl->GetFirstMatrix()) == *(n->GetMatrix())))
         {
            // std::string x = re[0].Data();
            //if (x.find(n->GetName()) == std::string::npos) printf("ERROT \n");

            m_entries[cnt].setBit(kOverlap);
            m_entries[cnt].setBit(kVisNodeSelf);
            m_mapNodeOverlaps.insert(std::pair<int, int>(cnt, ovlIdx));
            int nno; n->GetOverlaps(nno); 
            nno |= BIT(1); n->SetOverlaps(0, nno); 
         quid->m_nodes.push_back(cnt);
         }
      }

      if (n->GetVolume() == ovl->GetSecondVolume() && (*(ovl->GetSecondMatrix()) == *(n->GetMatrix())))
      {
         //printf("-----------------------------------------------\n");
         // std::string x = re[2].Data();
         // if (x.find(n->GetName()) == std::string::npos) printf("ERROT \n");

         m_entries[cnt].setBit(kOverlap);
         m_entries[cnt].setBit(kVisNodeSelf);
        
         m_mapNodeOverlaps.insert(std::pair<int, int>(cnt, ovlIdx));        
         int nno; n->GetOverlaps(nno); 
         nno |= (ovl->IsOverlap()  ? BIT(1) : BIT(2)); 
         n->SetOverlaps(0, nno);

         quid->m_nodes.push_back(cnt); 
        
      }


      FWGeometryTableManagerBase::getNNodesTotal(n, dOff);  
   }

   TPolyMarker3D* pm = ovl->GetPolyMarker();
   for (int j=0; j<pm->GetN(); ++j )
   {
      double pl[3];
      double pg[3];
      pm->GetPoint(j, pl[0], pl[1], pl[2]);
      motherm->LocalToMaster(pl, pg);
   
      float dx = TMath::Abs(ovl->GetOverlap());
      if (dx > 1e5) 
      { 
         fwLog(fwlog::kInfo)  << Form("WARNING [%s], overlap size = %.1f \n", ovl->GetTitle(), dx);
         dx =  10;
      }
      float dy = dx, dz = 0;
      float fp[3]; fp[0] = pg[0];fp[1] = pg[1];fp[2] = pg[2];
      float bb[12] = {
         fp[0] +dx, fp[1] -dy, fp[2] -dz,
         fp[0] +dx, fp[1] +dy, fp[2] +dz,
         fp[0] -dx, fp[1] +dy, fp[2] +dz,
         fp[0] -dx, fp[1] -dy, fp[2] -dz
      };
      m_browser->getMarker()->AddQuad(&bb[0]);
      m_browser->getMarker()->QuadId(quid); 
   }


   int aIdx = parentIdx;   int aLev = m_entries[aIdx].m_level;
   int topNodeIdx =  m_browser->getTopNodeIdx();

   while(aIdx > topNodeIdx)
   {
      aIdx--;
      if (m_entries[aIdx].m_level < aLev)
      {
         m_entries[aIdx].setBit(kOverlapChild);
         m_entries[aIdx].setBit(kVisNodeChld);
         //  printf("stamp %s \n", m_entries[aIdx].name());
         aLev--;
      }
   }
}


//_____________________________________________________________________________

void FWOverlapTableManager::recalculateVisibility( )
{
   // printf("overlap recalcuate vis \n");
   m_row_to_index.clear();
   int i = m_browser->getTopNodeIdx();
   m_row_to_index.push_back(i);

   if (m_entries[i].testBit(kExpanded)  )
      recalculateVisibilityNodeRec(i);
}

void FWOverlapTableManager::recalculateVisibilityNodeRec( int pIdx)
{
   TGeoNode* parentNode = m_entries[pIdx].m_node;
   int nD = parentNode->GetNdaughters();
   int dOff=0;
   for (int n = 0; n != nD; ++n)
   {
      int idx = pIdx + 1 + n + dOff;
      NodeInfo& data = m_entries[idx];


      if (m_browser->listAllNodes() || data.testBitAny(kOverlap | kOverlapChild))
         m_row_to_index.push_back(idx);
 
      if ((m_browser->listAllNodes() || data.testBit(kOverlapChild)) && data.testBit(kExpanded) )
         recalculateVisibilityNodeRec(idx);

      FWGeometryTableManagerBase::getNNodesTotal(parentNode->GetDaughter(n), dOff);
   }
} 


//______________________________________________________________________________

bool  FWOverlapTableManager::nodeIsParent(const NodeInfo& data) const
{
   return  m_browser->listAllNodes() ? data.m_node->GetNdaughters() : data.testBit(kOverlapChild) ;
}

void FWOverlapTableManager::printOverlaps(int idx) const
{
  
  TEveGeoManagerHolder gmgr( FWGeometryTableViewManager::getGeoMangeur());
  std::pair<std::multimap<int, int>::const_iterator, std::multimap<int, int>::const_iterator> ppp;
  ppp = m_mapNodeOverlaps.equal_range(idx);
  for (std::multimap<int, int>::const_iterator it2 = ppp.first;it2 != ppp.second;++it2) {
    const TGeoOverlap* ovl = (const TGeoOverlap*) gGeoManager->GetListOfOverlaps()->At((*it2).second);
    if (ovl) ovl->Print();
  }    
}

void FWOverlapTableManager::getOverlapTitles(int idx, TString& txt) const
{
  
   TEveGeoManagerHolder gmgr( FWGeometryTableViewManager::getGeoMangeur());
   std::pair<std::multimap<int, int>::const_iterator, std::multimap<int, int>::const_iterator> ppp;
   ppp = m_mapNodeOverlaps.equal_range(idx);
   for (std::multimap<int, int>::const_iterator it2 = ppp.first;it2 != ppp.second;++it2) {
      const TGeoOverlap* ovl = (const TGeoOverlap*) gGeoManager->GetListOfOverlaps()->At((*it2).second);
      {
         txt += "\n";

         if (ovl) {
            txt += Form("%s: %g, ", ovl->IsOverlap() ? "Ovl" : "Extr",  ovl->GetOverlap());
            txt += ovl->GetTitle();    
         }
      }
   }    
}
//______________________________________________________________________________
/*
  const char* FWOverlapTableManager::cellName(const NodeInfo& data) const
  {
  if (data.m_parent == -1)
  {
  int ne = 0;
  int no = 0;
  TGeoOverlap* ovl;
  TEveGeoManagerHolder gmgr( FWGeometryTableViewManager::getGeoMangeur());
  TIter next_ovl(gGeoManager->GetListOfOverlaps());
  while((ovl = (TGeoOverlap*)next_ovl())) 
  ovl->IsOverlap() ? no++ : ne++;
     
  return Form("%s Ovl[%d] Ext[%d]", data.m_node->GetName(), no, ne);
  }
  else
  {
  return data.name();
  }
  }*/

//______________________________________________________________________________

FWTableCellRendererBase* FWOverlapTableManager::cellRenderer(int iSortedRowNumber, int iCol) const
{  
   if (m_row_to_index.empty()) return &m_renderer;

   int unsortedRow =  m_row_to_index[iSortedRowNumber];

   if (unsortedRow < 0) printf("!!!!!!!!!!!!!!!! error %d %d \n",unsortedRow,  iSortedRowNumber);

   // editor state
   //
   m_renderer.showEditor(unsortedRow == m_editTransparencyIdx && iCol == 2);


   // selection state
   //
   const NodeInfo& data = m_entries[unsortedRow];

   bool isSelected = data.testBit(kHighlighted) ||  data.testBit(kSelected);
   if (m_browser->listAllNodes()) isSelected = isSelected ||  data.testBit(kOverlap);

   if (data.testBit(kSelected))
   {
      m_highlightContext->SetBackground(0xc86464);
   }
   else if (data.testBit(kHighlighted) )
   {
      m_highlightContext->SetBackground(0x6464c8);
   }
   else if (m_browser->listAllNodes() && data.testBit(kOverlap) )
   {
      m_highlightContext->SetBackground(0xdddddd);
   }
  

   // set column content
   //
   if (iCol == 0)
   {
      if (unsortedRow == m_browser->getTopNodeIdx())
      {
         int no = 0, ne =0;
         TEveGeoManagerHolder gmgr( FWGeometryTableViewManager::getGeoMangeur());
         TIter next_ovl(gGeoManager->GetListOfOverlaps());
         const TGeoOverlap* ovl;
         while((ovl = (TGeoOverlap*)next_ovl())) 
            ovl->IsOverlap() ? no++ : ne++;
      
         m_renderer.setData(Form("%s Ovl[%d] Ext[%d]", data.m_node->GetName(), no, ne), isSelected);
      }
      else {
         m_renderer.setData(data.name(), isSelected); 
      }
      m_renderer.setIsParent(nodeIsParent(data));

      m_renderer.setIsOpen( data.testBit(FWGeometryTableManagerBase::kExpanded));

      int level = data.m_level - m_levelOffset;
      if (nodeIsParent(data))
         m_renderer.setIndentation(20*level);
      else
         m_renderer.setIndentation(20*level + FWTextTreeCellRenderer::iconWidth());
   }
   else
   {
      m_renderer.setIsParent(false);
      m_renderer.setIndentation(0);

      if (iCol == 5)
      {
         if (data.testBit(kOverlap) ) 
         {
            std::string x;
            std::pair<std::multimap<int, int>::const_iterator, std::multimap<int, int>::const_iterator> ppp;
            ppp = m_mapNodeOverlaps.equal_range(unsortedRow);

            TEveGeoManagerHolder gmgr( FWGeometryTableViewManager::getGeoMangeur());
           
            for (std::multimap<int, int>::const_iterator it2 = ppp.first;it2 != ppp.second;++it2) {
               const TGeoOverlap* ovl = (const TGeoOverlap*) gGeoManager->GetListOfOverlaps()->At((*it2).second);
               if (ovl)
                  x +=  Form("%s: %g ", ovl->IsOverlap() ? "Ovl" : "Extr", ovl->GetOverlap());
               else
                  x += "err";
             
            }
            m_renderer.setData(x,  isSelected);
         }
         else
         {
            m_renderer.setData("",  isSelected);
         }
      }
      if (iCol == 1)
      {
         m_colorBoxRenderer.setData(data.m_color, isSelected);
         return  &m_colorBoxRenderer;
      }
      else if (iCol == 2 )
      { 
         m_renderer.setData(Form("%d", 100 -data.m_transparency), isSelected);
      }
      else if (iCol == 3 )
      {
         m_renderer.setData(data.testBit(kVisNodeSelf)  ? "On" : "-",  isSelected );

      }
      else if (iCol == 4 )
      {
         m_renderer.setData(data.testBit(kVisNodeChld)  ? "On" : "-",  isSelected);

      }
      else if (iCol == 6)
      { 
         std::cerr << "This shoud not happen! \n"     ;
      }
   }
   return &m_renderer;
}



//______________________________________________________________________________

void FWOverlapTableManager::setDaughtersSelfVisibility(int selectedIdx, bool v)
{
   int dOff = 0;
   TGeoNode* parentNode = m_entries[selectedIdx].m_node;
   int nD = parentNode->GetNdaughters();
   for (int n = 0; n != nD; ++n)
   {
      int idx = selectedIdx + 1 + n + dOff;
      NodeInfo& data = m_entries[idx];

      data.setBitVal(FWGeometryTableManagerBase::kVisNodeChld, v);
      data.setBitVal(FWGeometryTableManagerBase::kVisNodeSelf, v);


      FWGeometryTableManagerBase::getNNodesTotal(parentNode->GetDaughter(n), dOff);
   }
}
