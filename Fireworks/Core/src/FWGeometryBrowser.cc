#include <iostream>
#include <boost/bind.hpp>

#include "Fireworks/Core/interface/FWGeometryBrowser.h"
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

#include "TFile.h"
#include "TGFileDialog.h"
#include "TGeoNode.h"
#include "TGStatusBar.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGMenu.h"
#include "KeySymbols.h"

// #define PERFTOOL_BROWSER

#include "TEveManager.h"
#include "TEveGeoNode.h"
#include "TGeoManager.h"
#include "TEveScene.h"
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
   kTableDebug
};

FWGeometryBrowser::FWGeometryBrowser(FWGUIManager *guiManager, FWColorManager *colorManager)
   : TGMainFrame(gClient->GetRoot(), 600, 500),
     m_mode(this, "Mode:", 0l, 0l, 1l),
     m_filter(this,"Materials:",std::string()),
     m_autoExpand(this,"ExpandList:", 1l, 0l, 100l),
     m_visLevel(this,"VisLevel:", 3l, 1l, 100l),
     // m_maxDaughters(this,"MaxChildren:", 4l, 0l, 1000l), // debug
     //     m_path(this, "Path",std::string("")),
     m_topNodeIdx(this, "TopNodeIndex", -1l, 0, 1e7),
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

   m_autoExpand.changed_.connect(boost::bind(&FWGeometryBrowser::autoExpandChanged, this));
   m_visLevel.changed_.connect(boost::bind(&FWGeometryBrowser::refreshTable3D,this));

   m_mode.changed_.connect(boost::bind(&FWGeometryBrowser::refreshTable3D, this));

   // m_maxDaughters.changed_.connect(boost::bind(&FWGeometryBrowser::refreshTable3D, this)); // debug

   m_filter.changed_.connect(boost::bind(&FWGeometryBrowser::updateFilter, this));

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
   if (0){
      TGTextButton* rb = new TGTextButton (hp, "print");
      hp->AddFrame(rb);
      rb->Connect("Clicked()","FWGeometryBrowser",this,"printTable()");
   }
   m_settersFrame = new TGHorizontalFrame(this);
   this->AddFrame( m_settersFrame, new TGLayoutHints(kLHintsExpandX,4,2,2,2));
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
   DontCallClose();
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
   // if (geodebug) makeSetter(frame, &m_maxDaughters);
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
      
      // if (!geodebug && (&m_maxDaughters == (*it)))  continue;
          
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
      m_modelPopup->AddEntry("InspectMaterial", kInspectMaterial);
      m_modelPopup->AddEntry("InspectShape", kInspectShape);
      m_modelPopup->AddEntry("Table Debug", kTableDebug);

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

void FWGeometryBrowser::nodeColorChangeRequested(Color_t col)
{
   FWGeometryTableManager::NodeInfo& ni = m_tableManager->refSelected();
   ni.m_color = col;
   ni.m_node->GetVolume()->SetLineColor(col);
   refreshTable3D();
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

#ifdef PERFTOOL_BROWSER  
      ProfilerStart("load.browser.prof");
#endif  
      m_tableManager->loadGeometry();

#ifdef PERFTOOL_BROWSER  
      ProfilerStop();
#endif 


      m_eveTopNode = new FWGeoTopNode(this);
      const char* n = Form("%s level[%d] size[%d] \n",m_geoManager->GetCurrentNode()->GetName(), getVisLevel(), (int)m_tableManager->refEntries().size());                            
      m_eveTopNode->SetElementName(n);

      TEveElementList* scenes = gEve->GetScenes();
      for (TEveElement::List_i it = scenes->BeginChildren(); it != scenes->EndChildren(); ++it)
      {
         TEveScene* s = ((TEveScene*)(*it));
         TString name = s->GetElementName();
         if (name.Contains("3D") && !name.Contains("Geo"))
            s->AddElement(m_eveTopNode);
      }

      cdNode(m_topNodeIdx.value());
      // refreshTable3D();
      MapRaised();
   }
   catch (std::runtime_error &e)
   {
      fwLog(fwlog::kError) << "Failed to load simulation geomtery.\n";
      updateStatusBar("Failed to load simulation geomtery from file");
   }
}

void
FWGeometryBrowser::printTable()
{
   // print all entries
   m_tableManager->printChildren(-1);
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

//______________________________________________________________________________

void FWGeometryBrowser::updateStatusBar(const char* txt) {
   if (!txt) 
      txt = m_tableManager->getStatusMessage().c_str();

   m_statBar->SetText(txt, 0);
   fClient->NeedRedraw(this);
}

//______________________________________________________________________________


void FWGeometryBrowser::cdNode(int idx)
{
   std::string p;
   m_tableManager->getNodePath(idx, p);
   setPath(idx, p);
}

void FWGeometryBrowser::cdTop()
{
   std::string path = "/" ;
   path += m_tableManager->refEntries().at(0).name();
   setPath(-1, path ); 
}

void FWGeometryBrowser::cdUp()
{   
   if ( getTopNodeIdx() != -1)
   {
      int pIdx   = m_tableManager->refEntries()[getTopNodeIdx()].m_parent;
      std::string p;
      m_tableManager->getNodePath(pIdx, p);
      setPath(pIdx, p);
   }
}

void FWGeometryBrowser::setPath(int parentIdx, std::string& path)
{
   m_topNodeIdx.set(parentIdx);
#ifdef PERFTOOL_BROWSER  
   ProfilerStart(Form("cdPath%d.prof", parentIdx));
#endif

   m_geoManager->cd(path.c_str());
   TGeoNode* topNode = m_geoManager->GetCurrentNode();
   printf(" Set Path to [%s], curren node %s \n", path.c_str(), topNode->GetName());

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
   SetWindowName( Form("GeometryBrowser: %-10s",title.c_str())); 
   updateStatusBar();

}
//______________________________________________________________________________

void FWGeometryBrowser::updateFilter()
{
   m_tableManager->updateFilter();
   refreshTable3D();
   updateStatusBar();
}

//______________________________________________________________________________

void FWGeometryBrowser::autoExpandChanged()
{
  if (!m_geoManager) return;

   m_tableManager->checkExpandLevel();
   m_tableManager->redrawTable();
}

//______________________________________________________________________________

void FWGeometryBrowser::refreshTable3D()
{
   if (!m_geoManager) return;

   m_tableManager->redrawTable();

   if ( m_eveTopNode) {
      //      printf("refresh \n");
      m_eveTopNode->ElementChanged();
      gEve->FullRedraw3D(false, true);
   } 
   updateStatusBar();
}
