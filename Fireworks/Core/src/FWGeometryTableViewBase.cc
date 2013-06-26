#include <iostream>

#include <boost/bind.hpp>
#include <boost/regex.hpp>

#include "Fireworks/Core/interface/FWGeometryTableViewBase.h"
#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/interface/FWGeometryTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/src/FWPopupMenu.cc"
#include "Fireworks/Core/src/FWGeoTopNodeScene.h"
#include "Fireworks/Core/src/FWEveDigitSetScalableMarker.cc"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"


#include "TGFileDialog.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#include "TGStatusBar.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGLPhysicalShape.h"
#include "TGMenu.h"
#include "TGComboBox.h"
// #define PERFTOOL_BROWSER
#include "TGeoShape.h"
#include "TGeoBBox.h"
#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TEveScene.h"
#include "TEveSceneInfo.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGLCamera.h"
#include "TEveSelection.h"
#ifdef PERFTOOL_BROWSER 
#include <google/profiler.h>
#endif

//______________________________________________________________________________
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================

Bool_t FWGeometryTableViewBase::FWViewCombo::HandleButton(Event_t* event) 
{
   if (event->fType == kButtonPress)
   {
      bool map = false;

      FWPopupMenu* m_viewPopup = new FWPopupMenu(0);

      TEveElementList* views = gEve->GetViewers();
      int idx = 0;

      for (TEveElement::List_i it = views->BeginChildren(); it != views->EndChildren(); ++it)
      { 
         TEveViewer* v = ((TEveViewer*)(*it));
         if (strstr( v->GetElementName(), "3D") )
         {     
            bool added = false;          
            m_viewPopup->AddEntry(v->GetElementName(), idx);

            for (TEveElement::List_i eit = v->BeginChildren(); eit != v->EndChildren(); ++eit )
            {
               TEveScene* s = ((TEveSceneInfo*)*eit)->GetScene();
               if (m_el && s->HasChildren() && s->FirstChild() == m_el) {
                  added = true;
                  break;
               }
            }
            map = true;
            if (added)
               m_viewPopup->CheckEntry(idx);
         }
         ++idx;
      }

      if (map) {

         Window_t wdummy;
         Int_t ax,ay;
         gVirtualX->TranslateCoordinates(GetId(),
                                         gClient->GetDefaultRoot()->GetId(),
                                         event->fX, event->fY, //0,0 in local coordinates
                                         ax,ay, //coordinates of screen
                                         wdummy);


         m_viewPopup->PlaceMenu(ax, ay, true,true);
         m_viewPopup->Connect("Activated(Int_t)",
                              "FWGeometryTableViewBase",
                              const_cast<FWGeometryTableViewBase*>(m_tableView),
                              "selectView(Int_t)");
      }
      else
      {
         fwLog(fwlog::kInfo) << "No 3D View added. \n";
      }
   }
   return true;
}

//==============================================================================
//==============================================================================
// workaround to get ESC key event 

namespace {
   class FWGeometryVF : public TGVerticalFrame
   {
   public:
      FWGeometryVF(const TGWindow* p, FWGeometryTableViewBase* tv) :TGVerticalFrame(p), m_tv (tv)
      {
         m_tv = tv;
         gVirtualX->SelectInput(GetId(), kKeyPressMask | kKeyReleaseMask | kExposureMask |
                                kPointerMotionMask | kStructureNotifyMask | kFocusChangeMask |
                                kEnterWindowMask | kLeaveWindowMask);
      }

      virtual ~FWGeometryVF() {};

      virtual Bool_t HandleKey(Event_t *event)
      {
         if (event->fCode == (UInt_t) gVirtualX->KeysymToKeycode(kKey_Escape)) {
            m_tv->getTableManager()->cancelEditor(true);
         }
         return TGCompositeFrame::HandleKey(event);
      }

      FWGeometryTableViewBase* m_tv;
   };

   class  FWTranspEntry : public TGTextEntry
   {
   public:
      FWTranspEntry(const TGWindow* p, FWGeometryTableViewBase* tv) :TGTextEntry(p), m_tv (tv){}
      virtual ~FWTranspEntry() {}

      virtual Bool_t HandleKey(Event_t *event)
      {
         if (event->fCode == (UInt_t) gVirtualX->KeysymToKeycode(kKey_Escape)) {
            m_tv->getTableManager()->cancelEditor(true);
         }
         return TGTextEntry::HandleKey(event);
      }
      FWGeometryTableViewBase* m_tv;
   };
}
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
//==============================================================================
FWGeometryTableViewBase::FWGeometryTableViewBase(TEveWindowSlot* iParent,FWViewType::EType type, FWColorManager* colMng )
   : FWViewBase(type),
     m_topNodeIdx(this, "TopNodeIndex", -1l, 0, 1e7),
     m_autoExpand(this,"ExpandList:", 1l, 0l, 100l),
     m_enableHighlight(this,"EnableHighlight", true),
     m_parentTransparencyFactor(this, "ParentTransparencyFactor", 1l, 0l, 100l),
     m_leafTransparencyFactor(this, "LeafTransparencyFactor", 1l, 0l, 100l),
m_minParentTransparency(this, "MinParentTransparency", type == FWViewType::kOverlapTable ? 0l : 90l, 0l, 100l),
     m_minLeafTransparency(this, "MinLeafTransparency", 0l, 0l, 100l),
     m_colorManager(colMng),
     m_colorPopup(0),
     m_eveWindow(0),
     m_frame(0),
     m_viewBox(0),
     m_viewersConfig(0),
     m_enableRedraw(true),
     m_marker(0),
     m_eveTopNode(0),
     m_eveScene(0),
     m_tableRowIndexForColorPopup(-1)
{
   m_eveWindow = iParent->MakeFrame(0);
   TGCompositeFrame* xf = m_eveWindow->GetGUICompositeFrame();

   m_frame = new FWGeometryVF(xf, this);

   xf->AddFrame(m_frame, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));  
   
   m_parentTransparencyFactor.changed_.connect(boost::bind(&FWGeometryTableViewBase::refreshTable3D,this));
   m_leafTransparencyFactor.changed_.connect(boost::bind(&FWGeometryTableViewBase::refreshTable3D,this));
   m_minParentTransparency.changed_.connect(boost::bind(&FWGeometryTableViewBase::refreshTable3D,this));
   m_minLeafTransparency.changed_.connect(boost::bind(&FWGeometryTableViewBase::refreshTable3D,this));
 
}

void FWGeometryTableViewBase::postConst()
{
   m_tableWidget = new FWTableWidget(getTableManager(), m_frame); 
   m_frame->AddFrame(m_tableWidget,new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,2,2,0,0));
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);
   m_tableWidget->Connect("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",
                          "FWGeometryTableViewBase",this,
                          "cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)");
   m_tableWidget->disableGrowInWidth();
   //   resetSetters();

   
   FWTranspEntry *editor = new  FWTranspEntry(m_tableWidget->body(), this);
   editor->SetBackgroundColor(gVirtualX->GetPixel(kYellow-7));
   editor->SetFrameDrawn(false);
   editor->Connect("ReturnPressed()",  "FWGeometryTableViewBase",this,"transparencyChanged()");
   getTableManager()->setCellValueEditor(editor);

   m_frame->MapSubwindows();
   editor->UnmapWindow();
   m_frame->Layout();
   m_eveWindow->GetGUICompositeFrame()->Layout();
   m_frame->MapWindow();
}
//______________________________________________________________________________

FWGeometryTableViewBase::~FWGeometryTableViewBase()
{
   // take out composite frame and delete it directly (zwithout the timeout)
   TGCompositeFrame *frame = m_eveWindow->GetGUICompositeFrame();
   frame->RemoveFrame( m_frame );
   delete m_frame;



   m_eveWindow->DestroyWindowAndSlot();
   delete getTableManager();
}


namespace {
TEveScene* getMarkerScene(TEveViewer* v)
{
  TEveElement* si = v->FindChild(Form("SI - EventScene %s", v->GetElementName()));
  if(si) 
    return ((TEveSceneInfo*)(si))->GetScene();
  else
    return 0;
}
}
//==============================================================================


void FWGeometryTableViewBase::cdNode(int idx)
{
   std::string p;
   getTableManager()->getNodePath(idx, p);
   setPath(idx, p);
}

void FWGeometryTableViewBase::cdTop()
{
   std::string path = "/" ;
   path += getTableManager()->refEntries().at(0).name();
   setPath(-1, path ); 
}

void FWGeometryTableViewBase::cdUp()
{   
   if (getTopNodeIdx() != -1)
   {
      int pIdx = getTableManager()->refEntries()[getTopNodeIdx()].m_parent;
      std::string p;
      getTableManager()->getNodePath(pIdx, p);
      setPath(pIdx, p);
   }
}

void FWGeometryTableViewBase::setPath(int parentIdx, std::string&)
{
   m_eveTopNode->clearSelection();

   // printf("set path %d \n", parentIdx);
   m_topNodeIdx.set(parentIdx);
   // getTableManager()->refEntries().at(getTopNodeIdx()).setBitVal(FWGeometryTableManagerBase::kVisNodeSelf,!m_disableTopNode.value() );
   getTableManager()->setLevelOffset(getTableManager()->refEntries().at(getTopNodeIdx()).m_level);
 

   checkExpandLevel();
   refreshTable3D(); 
}

//------------------------------------------------------------------------------

void  FWGeometryTableViewBase::checkExpandLevel()
{
   // check expand state
   int ae = m_autoExpand.value();
   if ( m_topNodeIdx.value() > 0) 
      ae += getTableManager()->refEntries().at(m_topNodeIdx.value()).m_level;

   for (FWGeometryTableManagerBase::Entries_i i = getTableManager()->refEntries().begin(); i !=  getTableManager()->refEntries().end(); ++i)
   {
      if (i->m_level  < ae)
         i->setBit(FWGeometryTableManagerBase::kExpanded);
      else
         i->resetBit(FWGeometryTableManagerBase::kExpanded);
   } 
}

//==============================================================================

void
FWGeometryTableViewBase::populate3DViewsFromConfig()
{
   // post-config 
   if (m_viewersConfig) {
      TEveElementList* viewers = gEve->GetViewers();
      const FWConfiguration::KeyValues* keyVals = m_viewersConfig->keyValues();

      if(0!=keyVals)  
      {
         for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it!= keyVals->end(); ++it) {
    
            TString sname = it->first;
            TEveViewer* v = dynamic_cast<TEveViewer*>(viewers->FindChild(sname.Data()));
            if (!v)
            {
               fwLog(fwlog::kError)  << "FWGeometryTableViewBase::populate3DViewsFromConfig no viewer found " << it->first << std::endl;
               return;
            }
            v->AddScene(m_eveScene);  
            m_viewBox->setElement(m_eveTopNode);
            if (m_marker) getMarkerScene(v)->AddElement(m_marker);

            gEve->FullRedraw3D(false, true);
         }   
      }
   }
}

//==============================================================================

void 
FWGeometryTableViewBase::selectView(int idx)
{
   // callback from sleclect view popup menu

   m_viewBox->setElement(m_eveTopNode);

   TEveElement::List_i it = gEve->GetViewers()->BeginChildren();
   std::advance(it, idx);
   TEveViewer* v = (TEveViewer*)(*it);

   for (TEveElement::List_i eit = v->BeginChildren(); eit != v->EndChildren(); ++eit )
   {
      if ((((TEveSceneInfo*)(*eit))->GetScene()) == m_eveScene)
      {
        v->RemoveElement(*eit);
        if (m_marker) getMarkerScene(v)->RemoveElement(m_marker);
        gEve->Redraw3D();
        return;
      }
   }

   if (m_marker) getMarkerScene(v)->AddElement(m_marker); 
   v->AddScene(m_eveScene);
   gEve->Redraw3D();
}

//==============================================================================

void 
FWGeometryTableViewBase::setColumnSelected(int idx)
{
   // printf("cell clicled top node %p\n", (void*)m_eveTopNode);
   if (gEve->GetSelection()->HasChild( m_eveTopNode))
      gEve->GetSelection()->RemoveElement( m_eveTopNode);

   if (gEve->GetHighlight()->HasChild( m_eveTopNode))
      gEve->GetHighlight()->RemoveElement( m_eveTopNode);

   // reset bits and sets for old selected table entry
   m_eveTopNode->UnSelected();
   m_eveTopNode->UnHighlighted();


   if (m_eveTopNode->selectPhysicalFromTable(idx))
      gEve->GetSelection()->AddElement(m_eveTopNode);

   getTableManager()->refEntry(idx).setBit(FWGeometryTableManagerBase::kSelected);
   getTableManager()->redrawTable();
   gEve->Redraw3D();
}
//______________________________________________________________________________

void 
FWGeometryTableViewBase::cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t x, Int_t y)
{
   int idx = getTableManager()->rowToIndex()[iRow];
   FWGeometryTableManagerBase::NodeInfo& ni = getTableManager()->refEntries()[idx];

   if (iColumn != 2)  getTableManager()->cancelEditor(false);

   bool elementChanged = false;
   if (iButton == kButton1) 
   {
      if (iColumn == 0)
      {
         Window_t wdummy;
         Int_t xLoc,yLoc;
         gVirtualX->TranslateCoordinates(gClient->GetDefaultRoot()->GetId(), m_tableWidget->GetId(),  x, y, xLoc, yLoc, wdummy);

         if (getTableManager()->firstColumnClicked(iRow, xLoc))
            setColumnSelected(idx);
      }
      else if (iColumn == 1)
      { 
         std::vector<Color_t> colors;
         m_colorManager->fillLimitedColors(colors);
      
         if (!m_colorPopup) {
            m_colorPopup = new FWColorPopup(gClient->GetDefaultRoot(), colors.front());
            m_colorPopup->InitContent("", colors);
            m_colorPopup->Connect("ColorSelected(Color_t)","FWGeometryTableViewBase", const_cast<FWGeometryTableViewBase*>(this), "nodeColorChangeRequested(Color_t");
         }
         m_tableRowIndexForColorPopup = idx;
         m_colorPopup->SetName("Selected");
         m_colorPopup->ResetColors(colors, m_colorManager->backgroundColorIndex()==FWColorManager::kBlackIndex);
         m_colorPopup->PlacePopup(x, y, m_colorPopup->GetDefaultWidth(), m_colorPopup->GetDefaultHeight());
         return;
      }
      else if (iColumn == 2)
      {
         // transparency edit
         getTableManager()->showEditor(idx);
      }
      else if (iColumn == 3)
      {
         // vis self
         getTableManager()->setVisibility(ni, !getTableManager()->getVisibility(ni));
         elementChanged = true;
      }
      else if (iColumn == 4)
      { 
         // vis children
         getTableManager()->setVisibilityChld(ni, !getTableManager()->getVisibilityChld(ni));
         elementChanged = true;
      }
      else if (iColumn == 6)
      {
         // used in overlaps for RnrMarker column
         ni.switchBit(BIT(5));
         elementChanged = true;
      }
      else
      {
         setColumnSelected(idx);
      }

      if (elementChanged) {
         refreshTable3D();
         // getTableManager()->dataChanged();
      }
   }
   else if (iColumn == 0)
   {
      setColumnSelected(idx);
      m_eveTopNode->popupMenu(x, y, 0);
   }
}


void FWGeometryTableViewBase::setBackgroundColor()
{
   bool backgroundIsWhite = m_colorManager->backgroundColorIndex()==FWColorManager::kWhiteIndex;
   if(backgroundIsWhite) {
      m_tableWidget->SetBackgroundColor(0xffffff);
      m_tableWidget->SetLineSeparatorColor(0x000000);
   } else {
      m_tableWidget->SetBackgroundColor(0x000000);
      m_tableWidget->SetLineSeparatorColor(0xffffff);
   }
   getTableManager()->setBackgroundToWhite(backgroundIsWhite);
   gClient->NeedRedraw(m_tableWidget);
}

//______________________________________________________________________________

void FWGeometryTableViewBase::nodeColorChangeRequested(Color_t col)
{
   // AMT: need to add virtual   FWGeometryTableView::nodeColorChangeRequested() for volume mode
   
   //   printf("color change %d \n", m_tableRowIndexForColorPopup);
   if (m_tableRowIndexForColorPopup >= 0) {
      FWGeometryTableManagerBase::NodeInfo& ni = getTableManager()->refEntries()[m_tableRowIndexForColorPopup];
      ni.m_color = col;
      ni.m_node->GetVolume()->SetLineColor(col);
      refreshTable3D();
      m_tableRowIndexForColorPopup = -1;
   }
}


//______________________________________________________________________________
void FWGeometryTableViewBase::chosenItem(int menuIdx)
{
   int selectedIdx = m_eveTopNode->getFirstSelectedTableIndex();
   FWGeometryTableManagerBase::NodeInfo& ni = getTableManager()->refEntry(selectedIdx);
   // printf("chosen item %s %d\n", ni.name(), menuIdx);
   
   TGeoVolume *gv = ni.m_node->GetVolume();
   bool resetHome = false;
   if (gv)
   {
      switch (menuIdx)
      {
         case FWGeoTopNode::kVisSelfOff:
            getTableManager()->setVisibility(ni, false);
            refreshTable3D();
            break;
            
         case FWGeoTopNode::kVisChldOff:
            getTableManager()->setDaughtersSelfVisibility(selectedIdx, false);
            refreshTable3D();
            break;
            
         case FWGeoTopNode::kVisChldOn:
            getTableManager()->setDaughtersSelfVisibility(selectedIdx,  true);
            refreshTable3D();
            break;
            
         case FWGeoTopNode::kPrintMaterial:
            gv->InspectMaterial();
            break;
            
         case FWGeoTopNode::kPrintShape:
            gv->InspectShape();
            break;
            
         case FWGeoTopNode::kPrintPath:
         {
            std::string ps;
             getTableManager()->getNodePath(selectedIdx, ps);
            std::cout << ps << std::endl;
            break;
         }  
         case FWGeoTopNode::kSetTopNode:
            cdNode(selectedIdx);
            break;         
            
         case FWGeoTopNode::kSetTopNodeCam:
            cdNode(selectedIdx);
            resetHome = true;
            break;
            
         case FWGeoTopNode::kCamera:
         {
            TGLViewer* v = FWGeoTopNode::s_pickedViewer;
            v->CurrentCamera().SetExternalCenter(true);
            v->CurrentCamera().SetCenterVec(FWGeoTopNode::s_pickedCamera3DCenter.X(), FWGeoTopNode::s_pickedCamera3DCenter.Y(), FWGeoTopNode::s_pickedCamera3DCenter.Z());
            v->SetDrawCameraCenter(true);
           // resetHome = true;
            break;
         }
         default:
            return;
      }
   }
   
   if (resetHome) gEve->FullRedraw3D(true, true);
   
}
//______________________________________________________________________________
void FWGeometryTableViewBase::transparencyChanged()
{
   getTableManager()->applyTransparencyFromEditor();
   refreshTable3D();
}

//______________________________________________________________________________

void FWGeometryTableViewBase::refreshTable3D()
{
   if (m_enableRedraw)
   {
      if (gEve->GetSelection()->HasChild(m_eveTopNode))
         gEve->GetSelection()->RemoveElement(m_eveTopNode);

      if (gEve->GetHighlight()->HasChild(m_eveTopNode))
         gEve->GetHighlight()->RemoveElement(m_eveTopNode);

      m_eveTopNode->m_scene->PadPaint(m_eveTopNode->m_scene->GetPad());
      gEve->Redraw3D(); 

      getTableManager()->redrawTable();
   }
}

//______________________________________________________________________________

void FWGeometryTableViewBase::addTo(FWConfiguration& iTo) const
{
   FWConfigurableParameterizable::addTo(iTo);

   FWConfiguration viewers(1);
   FWConfiguration tempArea;

   for(TEveElement::List_i k = gEve->GetViewers()->BeginChildren(); k!= gEve->GetViewers()->EndChildren(); ++k)
   {
      for (TEveElement::List_i eit = (*k)->BeginChildren(); eit != (*k)->EndChildren(); ++eit )
      {
         TEveScene* s = ((TEveSceneInfo*)*eit)->GetScene();
         if (s->GetGLScene() == m_eveTopNode->m_scene)
         {
            viewers.addKeyValue( (*k)->GetElementName(), tempArea);
            break;
         }
      }
   }

   iTo.addKeyValue("Viewers", viewers, true);
}

//______________________________________________________________________________

void FWGeometryTableViewBase::setTopNodePathFromConfig(const FWConfiguration& iFrom)
{
   int tn;
   const FWConfiguration* value = iFrom.valueForKey( m_topNodeIdx.name() );
   if (!value) return;

   std::istringstream s(value->value());
   s>> tn;
   int lastIdx = getTableManager()->refEntries().size() -1;
   if (tn >= lastIdx) { 
      fwLog(fwlog::kWarning) << Form("Ignoring node path from confugration file -- %s value larger than number of nodes \n", m_topNodeIdx.name().c_str());
      return;
   }
   //   std::cerr << "set top node " << ;
   m_topNodeIdx.set(tn);
}

//______________________________________________________________________________

void FWGeometryTableViewBase::reloadColors()
{
  // printf("relaodColors \n");
   for (FWGeometryTableManagerBase::Entries_i i = getTableManager()->refEntries().begin(); i !=  getTableManager()->refEntries().end(); ++i)
   {
      i->m_color = i->m_node->GetVolume()->GetLineColor();
   }
   
   refreshTable3D();
}


//______________________________________________________________________________

void FWGeometryTableViewBase::populateController(ViewerParameterGUI& gui) const
{
   gui.requestTab("Style").
   separator().
   //addParam(&m_parentTransparencyFactor).
  // addParam(&m_leafTransparencyFactor).
   addParam(&m_minParentTransparency).
   addParam(&m_minLeafTransparency).
   separator();
   TGTextButton* butt = new TGTextButton(gui.getTabContainer(), "ReloadColors");
   gui.getTabContainer()->AddFrame(butt);
   butt->Connect("Clicked()", "FWGeometryTableViewBase", (FWGeometryTableViewBase*)this, "reloadColors()");

}


