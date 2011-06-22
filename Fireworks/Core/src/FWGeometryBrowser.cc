

#include <boost/bind.hpp>

#include "Fireworks/Core/interface/FWGeometryBrowser.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"

#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "Fireworks/Core/interface/fwLog.h"

#include "Fireworks/Core/src/FWPopupMenu.cc"
#include "Fireworks/Core/src/FWColorSelect.h"

#include "TFile.h"
#include "TGFileDialog.h"
#include "TGeoNode.h"
#include "TGStatusBar.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGMenu.h"
#include "KeySymbols.h"

#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TGeoManager.h"
#include "TEveScene.h"

bool geodebug = 0;

enum GeoMenuOptions {
   kSetTopNode,
   kInspectMaterial,
   kInspectShape
};

#include <iostream>
FWGeometryBrowser::FWGeometryBrowser(FWGUIManager *guiManager, FWColorManager *colorManager)
   : TGMainFrame(gClient->GetRoot(), 600, 500),
     m_mode(this, "Mode:", 1l, 0l, 1l),
     m_filter(this,"Materials:",std::string()),
     m_autoExpand(this,"AutoExpand:", 3l, 0l, 1000l),
     m_visLevel(this,"VisLevel:", 3l, 0l, 100l),
     m_maxDaughters(this,"MaxChildren:", 999l, 0l, 1000l), // debug
     m_guiManager(guiManager),
     m_colorManager(colorManager),
     m_tableManager(0),
     m_geometryFile(0),
     m_settersFrame(0),
     m_geoManager(0),
     m_eveTopNode(0),
     m_colorPopup(0),
     m_path("/cms:World_1")
{
   m_mode.addEntry(0, "Node");
   m_mode.addEntry(1, "Volume");
   
   m_tableManager = new FWGeometryTableManager(this);

   m_mode.changed_.connect(boost::bind(&FWGeometryTableManager::updateMode,m_tableManager));
   m_autoExpand.changed_.connect(boost::bind(&FWGeometryTableManager::updateAutoExpand,m_tableManager));
   m_maxDaughters.changed_.connect(boost::bind(&FWGeometryTableManager::updateAutoExpand,m_tableManager));
   m_filter.changed_.connect(boost::bind(&FWGeometryTableManager::updateFilter,m_tableManager));
   m_maxDaughters.changed_.connect(boost::bind(&FWGeometryTableManager::updateAutoExpand,m_tableManager));
 

   TGHorizontalFrame* hp =  new TGHorizontalFrame(this);
   AddFrame(hp,new TGLayoutHints(kLHintsLeft, 4, 2, 2, 2));
 
   TGTextButton* fileOpen = new TGTextButton (hp, "Open Geometry File");
   hp->AddFrame(fileOpen);
   fileOpen->Connect("Clicked()","FWGeometryBrowser",this,"browse()");

   {
      TGTextButton* rb = new TGTextButton (hp, "cdTop");
      hp->AddFrame(rb);
      rb->Connect("Clicked()","FWGeometryBrowser",this,"cdTop()");
   } {
      TGTextButton* rb = new TGTextButton (hp, "CdUp");
      hp->AddFrame(rb);
      rb->Connect("Clicked()","FWGeometryBrowser",this,"cdUp()");
   }

   m_settersFrame = new TGHorizontalFrame(this);
   this->AddFrame( m_settersFrame);
   m_settersFrame->SetCleanup(kDeepCleanup);


   m_tableWidget = new FWTableWidget(m_tableManager, this); 
   AddFrame(m_tableWidget,new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,2,2,2,2));
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);
   m_tableWidget->Connect("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",
                          "FWGeometryBrowser",this,
                          "cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)");
   m_tableWidget->disableGrowInWidth();
   resetSetters();
   backgroundChanged();

   m_statBar = new TGStatusBar(this, this->GetWidth(), 12);
   m_statBar->SetText("No simulation geomtery loaded.");
   AddFrame(m_statBar, new TGLayoutHints(kLHintsExpandX));

   SetWindowName("Geometry Browser");
   this->Connect("CloseWindow()","FWGeometryBrowser",this,"windowIsClosing()");
   Layout();
   MapSubwindows();


   m_colorManager->colorsHaveChanged_.connect(boost::bind(&FWGeometryBrowser::backgroundChanged,this));

   m_visLevel.changed_.connect(boost::bind(&FWGeometryBrowser::updateVisLevel,this));

   gVirtualX->SelectInput(GetId(), kKeyPressMask | kKeyReleaseMask | kExposureMask |
                          kPointerMotionMask | kStructureNotifyMask | kFocusChangeMask |
                          kEnterWindowMask | kLeaveWindowMask);

}

FWGeometryBrowser::~FWGeometryBrowser()
{}

void
FWGeometryBrowser::resetSetters()
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
   if (geodebug) makeSetter(frame, &m_maxDaughters);
   m_settersFrame->MapSubwindows();
   Layout();
}

void
FWGeometryBrowser::makeSetter(TGCompositeFrame* frame, FWParameterBase* param) 
{
   boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(param) );
   ptr->attach(param, this);
 
   TGFrame* pframe = ptr->build(frame, false);
   frame->AddFrame(pframe, new TGLayoutHints(kLHintsExpandX));

   m_setters.push_back(ptr);
}
//==============================================================================

void
FWGeometryBrowser::addTo(FWConfiguration& iTo) const
{
   FWConfigurableParameterizable::addTo(iTo);
}
  
void
FWGeometryBrowser::setFrom(const FWConfiguration& iFrom)
{ 
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      
      if (!geodebug && (&m_maxDaughters == (*it)))
          continue;
          
      (*it)->setFrom(iFrom);

   }  
   resetSetters();
}

//==============================================================================
void 
FWGeometryBrowser::cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t x, Int_t y)
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
            m_colorPopup->Connect("ColorSelected(Color_t)","FWGeometryBrowser", const_cast<FWGeometryBrowser*>(this), "nodeColorChangeRequested(Color_t)");
         }
         m_colorPopup->SetName("Selected");
         m_colorPopup->ResetColors(colors, m_colorManager->backgroundColorIndex()==FWColorManager::kBlackIndex);
         // m_colorPopup->SetSelection(id.item()->modelInfo(id.index()).displayProperties().color());
         m_colorPopup->PlacePopup(x, y, m_colorPopup->GetDefaultWidth(), m_colorPopup->GetDefaultHeight());
       
         return;
      }
      else
      {
       
         if (iColumn ==  FWGeometryTableManager::kVisSelf)
         {
            ni.m_node->SetVisibility(!ni.m_node->IsVisible());
            printf("set visiblity %s [%d] \n",ni.name(), ni.m_node->IsVisible() );
         }
         if (iColumn ==  FWGeometryTableManager::kVisChild)
         {
            ni.m_node->VisibleDaughters(!ni.m_node->IsVisDaughters());
            printf("set visiblity daughterts %s [%d] \n",ni.name(), ni.m_node->IsVisDaughters() );
         }
      }
        
      m_eveTopNode->ElementChanged(true, true);
      gEve->RegisterRedraw3D();

      m_tableManager->dataChanged();

   }
   else if (iColumn == FWGeometryTableManager::kName)
   {
      FWPopupMenu* m_modelPopup = new FWPopupMenu();
      m_modelPopup->AddEntry("Set As Top Node", kSetTopNode);
      m_modelPopup->AddEntry("InspectMaterial", kInspectMaterial);
      m_modelPopup->AddEntry("InspectShape", kInspectShape);

      m_modelPopup->PlaceMenu(x,y,true,true);
      m_modelPopup->Connect("Activated(Int_t)",
                            "FWGeometryBrowser",
                            const_cast<FWGeometryBrowser*>(this),
                            "chosenItem(Int_t)");
   }
}

void FWGeometryBrowser::chosenItem(int x)
{
   FWGeometryTableManager::NodeInfo& ni = m_tableManager->refSelected();
   TGeoVolume* gv = ni.m_node->GetVolume();
   if (gv)
   {
      switch (x) {
         case kSetTopNode:
            cdSelected();
            break;
         case kInspectMaterial:
            gv->InspectMaterial();
            break;
         case kInspectShape:
            gv->InspectShape();
            break;
      }
   }
}

void FWGeometryBrowser::backgroundChanged()
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
   fClient->NeedRedraw(m_tableWidget);
   fClient->NeedRedraw(this);

}

void  FWGeometryBrowser::nodeColorChangeRequested(Color_t col)
{
   FWGeometryTableManager::NodeInfo& ni = m_tableManager->refSelected();
   TGeoVolume* gv = ni.m_node->GetVolume();
   gv->SetLineColor(col);
 m_eveTopNode->ElementChanged(true, true);
         gEve->RegisterRedraw3D();

}

bool FWGeometryBrowser::HandleKey(Event_t *event)
{
   if (!fBindList) return kFALSE;

   TIter next(fBindList);
   TGMapKey *m;
   TGFrame  *w = 0;

   while ((m = (TGMapKey *) next())) {
      if (m->fKeyCode == event->fCode) {
         w = (TGFrame *) m->fWindow;
         if (w->HandleKey(event)) return kTRUE;
      }
   }
   return kFALSE;
}

void
FWGeometryBrowser::windowIsClosing()
{
  UnmapWindow();
}

void 
FWGeometryBrowser::readFile()
{
   try {

      if ( ! m_geometryFile )
         throw std::runtime_error("No root file.");
  
      m_geometryFile->ls();
      
      if ( !m_geometryFile->Get("cmsGeo;1"))
         throw std::runtime_error("Can't find TGeoManager object in selected file.");

      m_geoManager = (TGeoManager*) m_geometryFile->Get("cmsGeo;1");
      MapRaised();
      loadGeometry();

   }
   catch (std::runtime_error &e)
   {
      fwLog(fwlog::kError) << "Failed to load simulation geomtery.\n";
      updateStatusBar("Failed to load simulation geomtery from file");
   }
}


void 
FWGeometryBrowser::loadGeometry()
{ 
   // 3D scene
   if (m_eveTopNode) m_eveTopNode->Destroy();


   m_eveTopNode = new TEveGeoTopNode(m_geoManager, m_geoManager->GetCurrentNode());
   m_topGeoNode = m_geoManager->GetCurrentNode();
   TEveElementList* scenes = gEve->GetScenes();
   for (TEveElement::List_i it = scenes->BeginChildren(); it != scenes->EndChildren(); ++it)
   {
      TEveScene* s = ((TEveScene*)(*it));
      TString name = s->GetElementName();
      if (name.Contains("3D"))
      {
         s->AddElement(m_eveTopNode);
         printf("Add top node to %s scene \n", s->GetName());
      }
   }
   gEve->Redraw3D();

   m_tableManager->loadGeometry();
}


void
FWGeometryBrowser::browse()
{
   //std::cout<<"FWGeometryBrowser::browse()"<<std::endl;

   const char* defaultPath = Form("%s/cmsSimGeom-14.root",  gSystem->Getenv( "CMSSW_BASE" ));
   if( !gSystem->AccessPathName(defaultPath))
   {
      m_geometryFile = new TFile( defaultPath, "READ");
   }
   else
   {  
      const char* kRootType[] = {"ROOT files","*.root", 0, 0};
      TGFileInfo fi;
      fi.fFileTypes = kRootType;

      new TGFileDialog(gClient->GetDefaultRoot(), 
                       (TGWindow*) m_guiManager->getMainFrame(), kFDOpen, &fi);

      m_guiManager->updateStatus("loading geometry file...");
      m_geometryFile = new TFile(fi.fFilename, "READ");
   }
   m_guiManager->clearStatus();
   readFile();
}


void FWGeometryBrowser::updateStatusBar(const char* status) {
   m_statBar->SetText(status, 0);
}

void FWGeometryBrowser::updateVisLevel()
{
   if (m_eveTopNode) 
   {
      m_eveTopNode->SetVisLevel(m_visLevel.value());
      m_eveTopNode->ElementChanged();
      gEve->RegisterRedraw3D();
   }
}

//______________________________________________________________________________


void FWGeometryBrowser::cdSelected()
{
   m_tableManager->selectedPath(m_path);
   updatePath();
}

void FWGeometryBrowser::cdTop()
{
   m_path = "/cms:World_1";
   updatePath();
}

void FWGeometryBrowser::cdUp()
{
   if ( m_path != "/cms:World_1")
   {
      size_t del = m_path.find_last_of('/');
      m_path = m_path.substr(0, del);
      updatePath();
   }
}


void  FWGeometryBrowser::updatePath()
{
   m_geoManager->cd(m_path.c_str());
   printf("BEGIN Set Path to [%s], curren node %s \n", m_path.c_str(), m_geoManager->GetCurrentNode()->GetName());
   m_topGeoNode =  m_geoManager->GetCurrentNode();
   loadGeometry(); 
   printf("END Set Path to [%s], curren node %s \n", m_path.c_str(), m_geoManager->GetCurrentNode()->GetName()); 
}
