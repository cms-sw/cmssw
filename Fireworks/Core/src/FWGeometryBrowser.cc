

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
     m_maxDaughters(this,"MaxChildren:", 999l, 0l, 1000l), // debug
     m_guiManager(guiManager),
     m_colorManager(colorManager),
     m_tableManager(0),
     m_geometryFile(0),
     m_settersFrame(0),
     m_geoManager(0),
     m_eveTopNode(0),
     m_colorPopup(0)
{
   m_mode.addEntry(0, "Node");
   m_mode.addEntry(1, "Volume");
   
   m_tableManager = new FWGeometryTableManager(this);


   TGHorizontalFrame* hp =  new TGHorizontalFrame(this);
   AddFrame(hp,new TGLayoutHints(kLHintsLeft, 4, 2, 2, 2));
 
   TGTextButton* fileOpen = new TGTextButton (hp, "Open Geometry File");
   hp->AddFrame(fileOpen);
   fileOpen->Connect("Clicked()","FWGeometryBrowser",this,"browse()");

   TGTextButton* rb = new TGTextButton (hp, "Reset");
   hp->AddFrame(rb);
   rb->Connect("Clicked()","FWGeometryBrowser",this,"reset()");


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
         /*
           if (m_mode.value() == kNode)
           {
           TGeoNode* gn = ni.m_node;
           if (iColumn == FWGeometryTableManager::kVisSelf)
           gn->SetVisibility(!gn->IsVisible());
           else if (iColumn == FWGeometryTableManager::kVisChild)
           gn->VisibleDaughters(!gn->IsVisDaughters());
           }
           else*/
         { 
            if (ni.m_node && ni.m_node->GetVolume() )
            {
               TGeoVolume* gv = ni.m_node->GetVolume();
               if (iColumn ==  FWGeometryTableManager::kVisSelf)
                  gv->SetVisibility(!gv->IsVisible());
               if (iColumn ==  FWGeometryTableManager::kVisChild)
                  gv->VisibleDaughters(!gv->IsVisDaughters());
            }
            else
            {
               if (ni.m_node)
                  fwLog(fwlog::kError) << "Can't find volume for node " <<  ni.name() << std::endl;
               else
                  fwLog(fwlog::kError) << "Can't find node " << std::endl;

            }
         }
        
         m_eveTopNode->ElementChanged(true, true);
         gEve->RegisterRedraw3D();

         m_tableManager->dataChanged();
      }

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
            m_geoManager->SetTopVolume(gv);
            loadGeometry();
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

void FWGeometryBrowser::reset()
{
   m_geoManager->SetTopVolume(m_geoManager->GetMasterVolume());
   loadGeometry();  
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

