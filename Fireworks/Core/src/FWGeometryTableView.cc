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
#include "TEveSceneInfo.h"
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


class FWViewCombo : public TGTextButton
{
private:
   FWGeometryTableView* m_tableView;
   TEveElement* m_el;

public:
   FWViewCombo(const TGWindow *p, FWGeometryTableView* t): 
      TGTextButton(p, "Select Views", -1, TGButton::GetDefaultGC()(), TGTextButton::GetDefaultFontStruct(), kRaisedFrame | kDoubleBorder  ), m_tableView(t), m_el(0) {}
   virtual ~FWViewCombo() {}
   void setElement(TEveElement* x) {m_el = x;}

   virtual Bool_t  HandleButton(Event_t* event) 
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
               TEveSceneInfo* si = ( TEveSceneInfo*)v->FindChild(Form("SI - EventScene %s",v->GetElementName() ));
               if (m_el) {
                  for (TEveElement::List_i it = m_el->BeginParents(); it != m_el->EndParents(); ++it ){
                     if (*it == si->GetScene()) {
                        added = true;
                        break;
                     }
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
                                 "FWGeometryTableView",
                                 const_cast<FWGeometryTableView*>(m_tableView),
                                 "selectView(Int_t)");
         }
         else
         {
            fwLog(fwlog::kError) << "No 3D View added. \n";
         }
      }
      return true;
   }

};

//==============================================================================
//==============================================================================

FWGeometryTableView::FWGeometryTableView(TEveWindowSlot* iParent,FWColorManager* colMng, TGeoNode* tn, TObjArray* volumes )
   : FWViewBase(FWViewType::kGeometryTable),
     m_mode(this, "Mode:", 0l, 0l, 1l),
     m_filter(this,"Materials:",std::string()),
     m_autoExpand(this,"ExpandList:", 1l, 0l, 100l),
     m_visLevel(this,"VisLevel:", 3l, 1l, 100l),
     m_topNodeIdx(this, "TopNodeIndex", -1l, 0, 1e7),
     m_colorManager(colMng),
     m_tableManager(0),
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
         TGTextButton* rb = new TGTextButton (hp, "CdTop");
         hp->AddFrame(rb, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 0) );
         rb->Connect("Clicked()","FWGeometryTableView",this,"cdTop()");
      } {
         TGTextButton* rb = new TGTextButton (hp, "CdUp");
         hp->AddFrame(rb, new TGLayoutHints(kLHintsNormal, 2, 2, 0, 0));
         rb->Connect("Clicked()","FWGeometryTableView",this,"cdUp()");
      }

      {
         m_viewBox = new FWViewCombo(hp, this);
         hp->AddFrame( m_viewBox,new TGLayoutHints(kLHintsExpandY, 2, 2, 0, 0));
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

   if (tn)
   {
      m_tableManager->loadGeometry(tn, volumes);
      cdTop();
   }

   m_frame->MapSubwindows();
   m_frame->Layout();
   xf->Layout();
   m_frame->MapWindow();
}

FWGeometryTableView::~FWGeometryTableView()
{
  // take out composite frame and delete it directly (zwithout the timeout)
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

   FWConfiguration viewers(1);
   if (m_eveTopNode)
   { 
      for (TEveElement::List_i it = m_eveTopNode->BeginParents(); it != m_eveTopNode->EndParents(); ++it )
      {
         FWConfiguration tempArea;
         TEveScene* scene = dynamic_cast<TEveScene*>(*it);
         std::string n = scene->GetElementName();
         viewers.addKeyValue(n, tempArea);
      }
   }
   iTo.addKeyValue("Viewers", viewers, true);
}
  
void
FWGeometryTableView::setFrom(const FWConfiguration& iFrom)
{ 
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      (*it)->setFrom(iFrom);

   }     


   // views
   const FWConfiguration* controllers = iFrom.valueForKey("Viewers");
   if (controllers) {
      TEveElementList* scenes = gEve->GetScenes();
      const FWConfiguration::KeyValues* keyVals = controllers->keyValues();
      if(0!=keyVals) 
      {
         for(FWConfiguration::KeyValuesIt it = keyVals->begin(); it!= keyVals->end(); ++it) {
    
            TString sname = it->first;
            // printf("%d scene elements %s\n",  scenes->NumChildren(), sname.Data());
            TEveElement* s = scenes->FindChild(sname);
            if (s)
            {
                std::cout << sname.Data() << std::endl;   
               if (!m_eveTopNode) {
                  m_eveTopNode = new FWGeoTopNode(this);
                  m_eveTopNode->IncDenyDestroy();
                  m_viewBox->setElement(m_eveTopNode);
               }
               s->AddElement(m_eveTopNode);
            }
         }   
      }
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
FWGeometryTableView::selectView(int idx)
{
   TEveElement::List_i it = gEve->GetViewers()->BeginChildren();
   std::advance(it, idx);
   TEveViewer* v = (TEveViewer*)(*it);
   TEveSceneInfo* si = (TEveSceneInfo*)v->FindChild(Form("SI - EventScene %s",v->GetElementName()));

   bool added = false;
   if (!m_eveTopNode) {
      m_eveTopNode = new FWGeoTopNode(this);
      m_eveTopNode->IncDenyDestroy();
      m_viewBox->setElement(m_eveTopNode);
   }
   else
   {
      for (TEveElement::List_i it = m_eveTopNode->BeginParents(); it != m_eveTopNode->EndParents(); ++it ){
         if (*it == si->GetScene()) {
            added = true;
            break;
         }
      }
   }
   printf("add node %s \n", si->GetElementName());

   if (added)
      si->GetScene()->RemoveElement(m_eveTopNode);
   else
      si->GetScene()->AddElement(m_eveTopNode);

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
      FWPopupMenu* m_nodePopup = new FWPopupMenu();
      m_nodePopup->AddEntry("Set As Top Node", kSetTopNode);
      m_nodePopup->AddSeparator();
      m_nodePopup->AddEntry("Rnr Off For All Children", kVisOff);
      m_nodePopup->AddEntry("Rnr On For All Children", kVisOn);
      m_nodePopup->AddSeparator();
      m_nodePopup->AddEntry("Set Camera Center", kCamera);
      m_nodePopup->AddSeparator();
      m_nodePopup->AddEntry("InspectMaterial", kInspectMaterial);
      m_nodePopup->AddEntry("InspectShape", kInspectShape);
      //  m_nodePopup->AddEntry("Table Debug", kTableDebug);

      m_nodePopup->PlaceMenu(x,y,true,true);
      m_nodePopup->Connect("Activated(Int_t)",
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

   //  m_geoManager->cd(path.c_str());

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
   m_tableManager->checkExpandLevel();
   m_tableManager->redrawTable();
}

//______________________________________________________________________________

void FWGeometryTableView::refreshTable3D()
{
   m_tableManager->redrawTable();

   if ( m_eveTopNode) {
      m_eveTopNode->ElementChanged();
      gEve->FullRedraw3D(false, true);
   } 
}
