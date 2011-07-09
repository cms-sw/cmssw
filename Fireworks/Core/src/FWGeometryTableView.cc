#include <iostream>
#include <boost/bind.hpp>

#include "Fireworks/Core/interface/FWGeometryTableView.h"
#include "Fireworks/Core/interface/FWGeoTopNode.h"
#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/src/FWPopupMenu.cc"
#include "Fireworks/Core/src/FWColorSelect.h"

#include "Fireworks/Core/interface/fwLog.h"

#include "TGFileDialog.h"
#include "TGeoNode.h"
#include "TGeoMatrix.h"
#include "TGStatusBar.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGMenu.h"
#include "TGComboBox.h"
#include "KeySymbols.h"

// #define PERFTOOL_BROWSER
#include "TGeoShape.h"
#include "TGeoBBox.h"
#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TGeoManager.h"
#include "TEveScene.h"
#include "TEveViewer.h"
#include "TGLViewer.h"
#include "TGLCamera.h"
#ifdef PERFTOOL_BROWSER 
#include <google/profiler.h>
#endif

bool geodebug = 0;

enum GeoMenuOptions {
   kSetTopNode,
   kVisOn,
   kVisOff,
   kInspectMaterial,
   kInspectShape,
   kCamera,
   kTableDebug
};

FWGeometryTableView::FWGeometryTableView(TEveWindowSlot* iParent,FWColorManager* colMng, TGeoManager* geoManager )
   : FWViewBase(FWViewType::kGeometryTable),
     m_mode(this, "Mode:", 0l, 0l, 1l),
     m_filter(this,"Materials:",std::string()),
     m_autoExpand(this,"ExpandList:", 1l, 0l, 100l),
     m_visLevel(this,"VisLevel:", 3l, 1l, 100l),
     m_topNodeIdx(this, "TopNodeIndex", -1l, 0, 1e7),
     m_colorManager(colMng),
     m_tableManager(0),
     m_geoManager(geoManager),
     m_eveTopNode(0),
     m_colorPopup(0),
     m_eveWindow(0),
     m_frame(0),
     m_viewBox(0)
{
   m_eveWindow = iParent->MakeFrame(0);
   TGCompositeFrame* xf = m_eveWindow->GetGUICompositeFrame();

   m_frame = new TGVerticalFrame(xf);
   xf->AddFrame(m_frame, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));


   m_mode.addEntry(0, "Node");
   m_mode.addEntry(1, "Volume");
   
   m_tableManager = new FWGeometryTableManager(this);
   m_autoExpand.changed_.connect(boost::bind(&FWGeometryTableView::autoExpandChanged, this));
   m_visLevel.changed_.connect(boost::bind(&FWGeometryTableView::refreshTable3D,this));
   m_mode.changed_.connect(boost::bind(&FWGeometryTableView::refreshTable3D, this));
   m_filter.changed_.connect(boost::bind(&FWGeometryTableView::updateFilter, this));

   // top row
   {
      TGHorizontalFrame* hp =  new TGHorizontalFrame(m_frame);
 
      if (0) { TGTextButton* fileOpen = new TGTextButton (hp, "Open Geometry File");
         hp->AddFrame(fileOpen);
         fileOpen->Connect("Clicked()","FWGeometryTableView",this,"browse()");
      }
      {
         TGTextButton* rb = new TGTextButton (hp, "cdTop");
         hp->AddFrame(rb);
         rb->Connect("Clicked()","FWGeometryTableView",this,"cdTop()");
      } {
         TGTextButton* rb = new TGTextButton (hp, "CdUp");
         hp->AddFrame(rb);
         rb->Connect("Clicked()","FWGeometryTableView",this,"cdUp()");
      }

      {
         // hp->AddFrame(new TGLabel(hp,"Scene3D"),new TGLayoutHints(kLHintsBottom, 4,2, 0, 2));
         m_viewBox = new TGComboBox(hp);
         updateViewers3DList();
         hp->AddFrame( m_viewBox,new TGLayoutHints(kLHintsExpandY|kLHintsExpandX, 4, 2, 0, 0));
         m_viewBox->Connect("Selected(Int_t)", "FWGeometryTableView", this, "selectView(Int_t)");
      }
      m_frame->AddFrame(hp,new TGLayoutHints(kLHintsLeft|kLHintsExpandX, 4, 2, 2, 2));
   }

   m_settersFrame = new TGHorizontalFrame(m_frame);
   m_frame->AddFrame( m_settersFrame, new TGLayoutHints(kLHintsExpandX,4,2,2,2));
   m_settersFrame->SetCleanup(kDeepCleanup);

   m_tableWidget = new FWTableWidget(m_tableManager, m_frame); 
   m_frame->AddFrame(m_tableWidget,new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,2,2,0,0));
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);
   m_tableWidget->Connect("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",
                          "FWGeometryTableView",this,
                          "cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)");
   m_tableWidget->disableGrowInWidth();
   resetSetters();

   if (m_geoManager)
   {
      m_tableManager->loadGeometry();
      cdTop();
      //      populate3DView();
   }

   m_frame->MapSubwindows();
   m_frame->Layout();
   xf->Layout();
   m_frame->MapWindow();
}

FWGeometryTableView::~FWGeometryTableView()
{
  // take out composite frame and delete it directly (without the timeout)
   TGCompositeFrame *frame = m_eveWindow->GetGUICompositeFrame();
   frame->RemoveFrame( m_frame );
   delete m_frame;

   m_eveWindow->DestroyWindowAndSlot();
   delete m_tableManager;
}

//==============================================================================

void
FWGeometryTableView::addTo(FWConfiguration& iTo) const
{
   FWConfigurableParameterizable::addTo(iTo);
}
  
void
FWGeometryTableView::setFrom(const FWConfiguration& iFrom)
{ 
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      (*it)->setFrom(iFrom);

   }     
   resetSetters();
   cdNode(m_topNodeIdx.value());
}

void
FWGeometryTableView::resetSetters()
{
   if (!m_settersFrame->GetList()->IsEmpty())
   {
      m_setters.clear();
      while(!m_settersFrame->GetList()->IsEmpty())
      {
         TGFrameElement *el = (TGFrameElement*) m_settersFrame->GetList()->First();
         m_settersFrame->RemoveFrame(el->fFrame);
      }
   }
   TGCompositeFrame* frame =  m_settersFrame;
   makeSetter(frame, &m_mode);
   makeSetter(frame, &m_filter);
   makeSetter(frame, &m_autoExpand);
   makeSetter(frame, &m_visLevel);
   m_settersFrame->MapSubwindows();
   m_frame->Layout();
}

void
FWGeometryTableView::makeSetter(TGCompositeFrame* frame, FWParameterBase* param) 
{
   boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(param) );
   ptr->attach(param, this);
 
   TGFrame* m_frame = ptr->build(frame, false);
   frame->AddFrame(m_frame, new TGLayoutHints(kLHintsExpandX));

   m_setters.push_back(ptr);
}

//==============================================================================

void
FWGeometryTableView::updateViewers3DList()
{
   m_viewBox->RemoveAll();
   TEveElementList* scenes = gEve->GetScenes();
   int idx = 0;

   for (TEveElement::List_i it = scenes->BeginChildren(); it != scenes->EndChildren(); ++it)
   { 
      TEveScene* s = ((TEveScene*)(*it));
      TString name = s->GetElementName();
      if (name.Contains("3D") && !name.Contains("Geo"))
      {
         m_viewBox->AddEntry(s->GetElementName(), idx);
      }
      ++idx;
   }
}

void 
FWGeometryTableView::selectView(int idx)
{
  
   m_eveTopNode = new FWGeoTopNode(this);
   const char* n = Form("%s level[%d] size[%d] \n",m_geoManager->GetCurrentNode()->GetName(), getVisLevel(), (int)m_tableManager->refEntries().size());                            
   m_eveTopNode->SetElementName(n);

 TEveElement::List_i it = gEve->GetScenes()->BeginChildren();
  std::advance(it, idx);

   TEveScene* s = ((TEveScene*)(*it));
   TString name = s->GetElementName();
   s->AddElement(m_eveTopNode);

   m_eveTopNode->ElementChanged();
   gEve->Redraw3D();
}

//==============================================================================
void 
FWGeometryTableView::cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t x, Int_t y)
{
   m_tableManager->setSelection(iRow, iColumn, iButton);
   FWGeometryTableManager::NodeInfo& ni = m_tableManager->refSelected();



   if (iButton == kButton1) 
   {

      if (iColumn == FWGeometryTableManager::kName)
      {
         m_tableManager->firstColumnClicked(iRow);
         return;
      }
      else if (iColumn == FWGeometryTableManager::kColor)
      { 
         std::vector<Color_t> colors;
         m_colorManager->fillLimitedColors(colors);
      
         if (!m_colorPopup) {
            m_colorPopup = new FWColorPopup(gClient->GetDefaultRoot(), colors.front());
            m_colorPopup->InitContent("", colors);
            m_colorPopup->Connect("ColorSelected(Color_t)","FWGeometryTableView", const_cast<FWGeometryTableView*>(this), "nodeColorChangeRequested(Color_t)");
         }
         m_colorPopup->SetName("Selected");
         m_colorPopup->ResetColors(colors, m_colorManager->backgroundColorIndex()==FWColorManager::kBlackIndex);
         m_colorPopup->PlacePopup(x, y, m_colorPopup->GetDefaultWidth(), m_colorPopup->GetDefaultHeight());
         return;
      }
      else
      {
         bool elementChanged = false;
         if (iColumn ==  FWGeometryTableManager::kVisSelf)
         {
            if (getVolumeMode())
               ni.m_node->GetVolume()->SetVisibility(!ni.isVisible(getVolumeMode()));   
            else
               ni.m_node->SetVisibility(!ni.isVisible(getVolumeMode()));    
            elementChanged = true;
         }
         if (iColumn ==  FWGeometryTableManager::kVisChild)
         {
            if (getVolumeMode())
               ni.m_node->GetVolume()->VisibleDaughters(!ni.isVisDaughters(getVolumeMode()));   
            else
               ni.m_node->VisibleDaughters(!ni.isVisDaughters(getVolumeMode()));  
            elementChanged = true;
         }


         if (elementChanged)
         {
            m_eveTopNode->ElementChanged();
            gEve->RegisterRedraw3D();
         }
      }
        

      m_tableManager->dataChanged();

   }
   else if (iColumn == FWGeometryTableManager::kName)
   {
      FWPopupMenu* m_modelPopup = new FWPopupMenu();
      m_modelPopup->AddEntry("Set As Top Node", kSetTopNode);
      m_modelPopup->AddSeparator();
      m_modelPopup->AddEntry("Rnr Off For All Children", kVisOff);
      m_modelPopup->AddEntry("Rnr On For All Children", kVisOn);
      m_modelPopup->AddSeparator();
      m_modelPopup->AddEntry("Set Camera Center", kCamera);
      m_modelPopup->AddSeparator();
      m_modelPopup->AddEntry("InspectMaterial", kInspectMaterial);
      m_modelPopup->AddEntry("InspectShape", kInspectShape);
      //  m_modelPopup->AddEntry("Table Debug", kTableDebug);

      m_modelPopup->PlaceMenu(x,y,true,true);
      m_modelPopup->Connect("Activated(Int_t)",
                            "FWGeometryTableView",
                            const_cast<FWGeometryTableView*>(this),
                            "chosenItem(Int_t)");
   }
}

void FWGeometryTableView::chosenItem(int x)
{
   FWGeometryTableManager::NodeInfo& ni = m_tableManager->refSelected();
   TGeoVolume* gv = ni.m_node->GetVolume();
   bool visible = true;
   if (gv)
   {
      switch (x) {
         case kSetTopNode:
            cdNode(m_tableManager->m_selectedIdx);
            break;

         case kVisOff:
            visible = false;
         case kVisOn: 
            if (getVolumeMode())
            {
               m_tableManager->setDaughterVolumesVisible(visible);
            }
            else
            {
               for (int d = 0; d < ni.m_node->GetNdaughters(); ++d )
               {
              
                  ni.m_node->GetDaughter(d)->SetVisibility(visible);
                  ni.m_node->GetDaughter(d)->VisibleDaughters(visible);
               }
            }
            refreshTable3D();
            break;

         case kCamera:
         {
            TGeoHMatrix mtx;
            m_tableManager->getNodeMatrix( m_tableManager->refSelected(), mtx);

            static double pnt[3];
            TGeoBBox* bb = static_cast<TGeoBBox*>( m_tableManager->refSelected().m_node->GetVolume()->GetShape());
            const double* origin = bb->GetOrigin();
            mtx.LocalToMaster(origin, pnt);

            TEveElementList* vl = gEve->GetViewers();
            for (TEveElement::List_i it = vl->BeginChildren(); it != vl->EndChildren(); ++it)
            {
               TEveViewer* v = ((TEveViewer*)(*it));
               TString name = v->GetElementName();
               if (name.Contains("3D"))
               {
                  v->GetGLViewer()->SetDrawCameraCenter(true);
                  TGLCamera& cam = v->GetGLViewer()->CurrentCamera();
                  cam.SetExternalCenter(true);
                  cam.SetCenterVec(pnt[0], pnt[1], pnt[2]);
               }
            }

            break;
         }
         case kInspectMaterial:
            gv->InspectMaterial();
            break;
         case kInspectShape:
            gv->InspectShape();
            break;
         case kTableDebug:
            // std::cout << "node name " << ni.name() << "parent " <<m_tableManager->refEntries()[ni.m_parent].name() <<  std::endl;
            // printf("node expanded [%d] imported[%d] children[%d]\n", ni.m_expanded,m_tableManager->nodeImported(m_tableManager->m_selectedIdx) ,  ni.m_node->GetNdaughters());
            m_tableManager->printChildren( m_tableManager->m_selectedIdx);
            break;
      }
   }
}

void FWGeometryTableView::setBackgroundColor()
{
   bool backgroundIsWhite = m_colorManager->backgroundColorIndex()==FWColorManager::kWhiteIndex;
   if(backgroundIsWhite) {
      m_tableWidget->SetBackgroundColor(0xffffff);
      m_tableWidget->SetLineSeparatorColor(0x000000);
   } else {
      m_tableWidget->SetBackgroundColor(0x000000);
      m_tableWidget->SetLineSeparatorColor(0xffffff);
   }
   m_tableManager->setBackgroundToWhite(backgroundIsWhite);
   gClient->NeedRedraw(m_tableWidget);
}

void FWGeometryTableView::nodeColorChangeRequested(Color_t col)
{
   FWGeometryTableManager::NodeInfo& ni = m_tableManager->refSelected();
   ni.m_color = col;
   ni.m_node->GetVolume()->SetLineColor(col);
   refreshTable3D();
}

void
FWGeometryTableView::printTable()
{
   // print all entries
   m_tableManager->printChildren(-1);
}

//______________________________________________________________________________


void FWGeometryTableView::cdNode(int idx)
{
   std::string p;
   m_tableManager->getNodePath(idx, p);
   setPath(idx, p);
}

void FWGeometryTableView::cdTop()
{
   std::string path = "/" ;
   path += m_tableManager->refEntries().at(0).name();
   setPath(-1, path ); 
}

void FWGeometryTableView::cdUp()
{   
   if ( getTopNodeIdx() != -1)
   {
      int pIdx   = m_tableManager->refEntries()[getTopNodeIdx()].m_parent;
      std::string p;
      m_tableManager->getNodePath(pIdx, p);
      setPath(pIdx, p);
   }
}

void FWGeometryTableView::setPath(int parentIdx, std::string& path)
{
   m_topNodeIdx.set(parentIdx);
#ifdef PERFTOOL_BROWSER  
   ProfilerStart(Form("cdPath%d.prof", parentIdx));
#endif

   m_geoManager->cd(path.c_str());
   // TGeoNode* topNode = m_geoManager->GetCurrentNode();
   // printf(" Set Path to [%s], curren node %s \n", path.c_str(), topNode->GetName());

   m_tableManager->topGeoNodeChanged(parentIdx);
   m_tableManager->checkExpandLevel();
   refreshTable3D();
   // printf("END Set Path to [%s], curren node %s \n", m_path.value().c_str(), topNode->GetName()); 


#ifdef PERFTOOL_BROWSER  
   ProfilerStop();
#endif  
   std::string title =  path;
   if (title.size() > 40)
   {
      title = title.substr(title.size() -41, 40);
      size_t del = title.find_first_of('/');
      if (del > 0)
      {
         title = title.substr(del);
      }
      title = "..." + title;
   }
}
//______________________________________________________________________________

void FWGeometryTableView::updateFilter()
{
   m_tableManager->updateFilter();
   refreshTable3D();
}

//______________________________________________________________________________

void FWGeometryTableView::autoExpandChanged()
{
  if (!m_geoManager) return;

   m_tableManager->checkExpandLevel();
   m_tableManager->redrawTable();
}

//______________________________________________________________________________

void FWGeometryTableView::refreshTable3D()
{
   if (!m_geoManager) return;

   m_tableManager->redrawTable();

   if ( m_eveTopNode) {
      m_eveTopNode->ElementChanged();
      gEve->FullRedraw3D(false, true);
   } 
}
