// -*- C++ -*-
//
// Package:     Core
// Class  :     FWOverlapTableView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  
//         Created:  Wed Jan  4 00:06:35 CET 2012
// $Id: FWOverlapTableView.cc,v 1.17 2013/04/19 04:21:29 amraktad Exp $
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "Fireworks/Core/src/FWOverlapTableView.h"
#include "Fireworks/Core/src/FWGeoTopNodeScene.h"
#include "Fireworks/Core/src/FWOverlapTableManager.h"
#include "Fireworks/Core/src/FWEveOverlap.h"
#include "Fireworks/Core/interface/FWGeometryTableViewManager.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/src/FWEveDigitSetScalableMarker.cc"
#include "Fireworks/Core/src/FWPopupMenu.cc"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "Fireworks/Core/src/FWValidatorBase.h"
#include "Fireworks/Core/src/FWEveDigitSetScalableMarker.cc"

#include "TEveScene.h"
#include "TEveSceneInfo.h"
#include "TEveWindow.h"
#include "TEveManager.h"


#include "TGeoVolume.h"
#include "TGeoMatrix.h"
#include "TGeoShape.h"
#include "TGeoBBox.h"
#include "TGeoMatrix.h"
#include "TGeoManager.h"

#include "TGLViewer.h"
#include "KeySymbols.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
#include "TGListBox.h"
#include "TGButton.h"
#include "TEveViewer.h"
#include "TGeoOverlap.h"
#include "TGClient.h"

static const std::string sUpdateMsg = "Please press Apply button to update overlaps.\n";


FWOverlapTableView::FWOverlapTableView(TEveWindowSlot* iParent, FWColorManager* colMng) : 
   FWGeometryTableViewBase(iParent, FWViewType::kOverlapTable, colMng),
   m_applyButton(0),
   m_listOptionButton(0),
   m_tableManager(0),
   m_numEntry(0),
   m_runChecker(true),
   m_path(this,"Path:", std::string("/cms:World_1/cms:CMSE_1")),
   m_precision(this, "Precision", 0.05, 0.000001, 10),
   m_listAllNodes(this, "ListAllNodes", true),
   m_rnrOverlap(this, "Overlap", true),
   m_rnrExtrusion(this, "Extrusion", true),
  m_drawPoints(this, "DrawPoints", true),
  m_pointSize(this, "PointSize", 4l, 0l, 10l),
  m_extrusionMarkerColor(this, "ExtrusionMarkerColor", 0l, 0l, 20l),
  m_overlapMarkerColor(this, "OverlapMarkerColor", 9l, 0l, 20l)
{ 
   // top row
   TGHorizontalFrame* hp =  new TGHorizontalFrame(m_frame);

   {
      m_viewBox = new FWViewCombo(hp, this);
      hp->AddFrame( m_viewBox,new TGLayoutHints(kLHintsExpandY, 2, 2, 0, 0));
   }

   {
      TGTextButton* rb = new TGTextButton (hp, "CdTop");
      hp->AddFrame(rb, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 0) );
      rb->Connect("Clicked()","FWGeometryTableViewBase",this,"cdTop()");
   }
  
   {
      TGTextButton* rb = new TGTextButton (hp, "CdUp");
      hp->AddFrame(rb, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 0));
      rb->Connect("Clicked()","FWGeometryTableViewBase",this,"cdUp()");
   }
   {
      hp->AddFrame(new TGLabel(hp, "Precision:"), new TGLayoutHints(kLHintsBottom, 10, 0, 0, 2));
      m_numEntry = new TGNumberEntry(hp,  m_precision.value(), 5, -1, TGNumberFormat::kNESReal, TGNumberFormat::kNEAAnyNumber, TGNumberFormat::kNELLimitMinMax, m_precision.min(), m_precision.max());
      hp->AddFrame(m_numEntry, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 0));
      m_numEntry->Connect("ValueSet(Long_t)","FWOverlapTableView",this,"precisionCallback(Long_t)");
   }

   { 
      m_listOptionButton  = new TGCheckButton(hp,m_listAllNodes.name().c_str());
      m_listOptionButton->SetState( m_listAllNodes.value() ? kButtonDown : kButtonUp );
      m_listOptionButton->Connect("Clicked()", "FWOverlapTableView", this, "setListAllNodes()");
      hp->AddFrame(m_listOptionButton, new TGLayoutHints(kLHintsLeft|kLHintsCenterY,2,0,1,1));

   }
   {
      m_applyButton = new TGTextButton (hp, "Apply");
      hp->AddFrame( m_applyButton, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 0));
      m_applyButton->Connect("Clicked()","FWOverlapTableView",this,"recalculate()");
   }
   
   m_frame->AddFrame(hp,new TGLayoutHints(kLHintsLeft|kLHintsExpandX, 4, 2, 2, 0));
   m_tableManager = new FWOverlapTableManager(this);

   // std::cerr << " FWOverlapTableView::initGeometry \n";
  
   FWGeoTopNodeGLScene *gls = new FWGeoTopNodeGLScene(0);
#if ROOT_VERSION_CODE < ROOT_VERSION(5,32,0)
   m_eveScene  = new  FWGeoTopNodeEveScene(gls, "TopGeoNodeScene", "");
#else
   m_eveScene  = new  TEveScene(gls, "TopGeoNodeScene", "");
#endif

   gEve->GetScenes()->AddElement(m_eveScene);

   m_eveTopNode = new  FWEveOverlap(this);
   m_eveTopNode->SetElementNameTitle("overlapNode", "opverlapNodetitle");
   m_eveTopNode->IncDenyDestroy();
   m_eveTopNode->SetPickable(true);
   m_eveScene->AddElement(m_eveTopNode);

   gls->m_eveTopNode = m_eveTopNode;
   m_eveTopNode->m_scene   = gls;

   m_marker = new  FWEveDigitSetScalableMarker();
   m_marker->SetMainColor(kRed);
   m_marker->IncDenyDestroy();
   m_marker->Reset(TEveQuadSet::kQT_FreeQuad, kFALSE, 32);
   m_marker->SetOwnIds(kTRUE);
   m_marker->SetAlwaysSecSelect(kTRUE);
   m_marker->SetPickable(kTRUE);
   m_marker->SetOwnIds(kTRUE);


   m_drawPoints.changed_.connect(boost::bind(&FWOverlapTableView::drawPoints,this));
   m_pointSize.changed_.connect(boost::bind(&FWOverlapTableView::pointSize,this));
   m_rnrOverlap.changed_.connect(boost::bind(&FWOverlapTableView::refreshTable3D,this));
   m_overlapMarkerColor.changed_.connect(boost::bind(&FWOverlapTableView::refreshTable3D,this));
   m_extrusionMarkerColor.changed_.connect(boost::bind(&FWOverlapTableView::refreshTable3D,this));
   m_rnrExtrusion.changed_.connect(boost::bind(&FWGeometryTableViewBase::refreshTable3D,this));
  
   postConst();
}
//______________________________________________________________________________


FWOverlapTableView::~FWOverlapTableView()
{
   if (m_marker) m_marker->DecDenyDestroy();
}

//______________________________________________________________________________
FWGeometryTableManagerBase* FWOverlapTableView::getTableManager()
{
   return m_tableManager;
}

//______________________________________________________________________________
bool FWOverlapTableView::listAllNodes() const
{
   return m_listAllNodes.value();
}

//______________________________________________________________________________
void FWOverlapTableView::setListAllNodes()
{
   m_listAllNodes.set( m_listOptionButton->IsOn());
   refreshTable3D();
}
//______________________________________________________________________________

TEveElement* FWOverlapTableView::getEveGeoElement() const
{
   return m_eveTopNode;
}

//______________________________________________________________________________
void FWOverlapTableView::precisionCallback(Long_t )
{
   // std::cout << " ----------------------------- PRECISION \n" <<  m_numEntry->GetNumber();
   setCheckerState(true);
   m_precision.set( m_numEntry->GetNumber());
   std::cout << sUpdateMsg;
}


void FWOverlapTableView::recalculate()
{
   //m_path.set(m_pathEntry->GetText());
   //  m_precision.set(m_numEntry->GetNumber());
   // std::cout << "                             $$$$ " << m_path.value() << std::endl;
   m_tableManager->importOverlaps(m_path.value(), m_precision.value());
   checkExpandLevel();
   getTableManager()->setLevelOffset(getTableManager()->refEntries().at(getTopNodeIdx()).m_level);
   refreshTable3D();

   setCheckerState(false);
}


//______________________________________________________________________________
void FWOverlapTableView::setFrom(const FWConfiguration& iFrom)
{
   m_enableRedraw = false;

   for (const_iterator it =begin(), itEnd = end(); it != itEnd; ++it)
   { 
         (*it)->setFrom(iFrom);
   }  

   m_viewersConfig = iFrom.valueForKey("Viewers");
   m_numEntry->SetNumber(m_precision.value());
  
   //  refreshTable3D();
   m_enableRedraw = true;
   recalculate();
}

//______________________________________________________________________________
void FWOverlapTableView::populateController(ViewerParameterGUI& gui) const
{
   gui.requestTab("Style").
      // addParam(&m_enableHighlight).
      // separator().
      addParam(&m_rnrOverlap).
      addParam(&m_rnrExtrusion).
      separator().
      addParam(&m_extrusionMarkerColor).
     addParam(&m_overlapMarkerColor).
      addParam(&m_pointSize);
   
   FWGeometryTableViewBase::populateController(gui);
}

//______________________________________________________________________________
void FWOverlapTableView::drawPoints()
{
   m_marker->SetRnrSelf(m_drawPoints.value());
   m_marker->ElementChanged();
   gEve->Redraw3D();
}

//______________________________________________________________________________
void FWOverlapTableView::pointSize()
{
   m_marker->SetMarkerSize(m_pointSize.value());
   m_marker->ElementChanged();
   gEve->Redraw3D();
}

//______________________________________________________________________________
void FWOverlapTableView::cdUp()
{
   setCheckerState(true);
   FWGeometryTableViewBase::cdUp();
}
//______________________________________________________________________________
void FWOverlapTableView::cdTop()
{
   if (m_topNodeIdx.value() == -1) return;

   setCheckerState(true);
   FWGeometryTableViewBase::cdTop();
}
//______________________________________________________________________________
void FWOverlapTableView::setCheckerState(bool x)
{
   m_runChecker = x;
   m_applyButton->SetForegroundColor(x ? 0xff0000 : 0x000000);
   gClient->NeedRedraw(m_applyButton);

}
//______________________________________________________________________________

void FWOverlapTableView::chosenItem(int menuIdx)
{
   // printf(" FWOverlapTableView::chosenItem chosen item %s \n", ni->name());
   
   switch (menuIdx) {
      case FWGeoTopNode::kPrintOverlap:
      {
         std::cout << "=============================================================================" <<  std::endl << std::endl;
         m_tableManager->printOverlaps(m_eveTopNode->getFirstSelectedTableIndex());
         break;
      }
      default:
         FWGeometryTableViewBase::chosenItem(menuIdx);
   }
}

//______________________________________________________________________________
void FWOverlapTableView::refreshTable3D()
{
   using namespace TMath;
   if (!m_enableRedraw) return;
   FWGeometryTableViewBase::refreshTable3D();

   for (int i = 0; i < m_marker->GetPlex()->Size(); ++i)
   {
      FWOverlapTableManager::QuadId* id = (FWOverlapTableManager::QuadId*) m_marker->GetId(i);
      TEveQuadSet::QFreeQuad_t* q = (TEveQuadSet::QFreeQuad_t*)m_marker->GetDigit(i);
      q->fValue = -1;

      // check if any of the overlaping nodes is visible -> is in the subtree
      bool rnr = false;
      
      for (std::vector<int>::iterator j = id->m_nodes.begin(); j < id->m_nodes.end(); ++j)
      {
         if ( (id->m_ovl->IsExtrusion() && m_rnrExtrusion.value()) ||  (id->m_ovl->IsOverlap() && m_rnrOverlap.value()))
         {
            if (*j == getTopNodeIdx() || m_tableManager->isNodeRendered(*j, getTopNodeIdx() )) {
               rnr = true;
               break;
            }
         }
      }

      if (rnr) {
         q->fValue = (id->m_ovl->IsOverlap())  ? m_overlapMarkerColor.value() : m_extrusionMarkerColor.value();
         q->fValue += 1000;
      }
   }

   m_marker->ElementChanged();
   gEve->FullRedraw3D(false, true);
}
