#include "Fireworks/Core/interface/FWGeometryBrowser.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"

#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "Fireworks/Core/interface/fwLog.h"


#include "TFile.h"
#include "TGFileDialog.h"
#include "TGeoNode.h"
#include "TGStatusBar.h"
#include "TGButton.h"
#include "TGLabel.h"

bool geodebug = 0;

#include <iostream>
FWGeometryBrowser::FWGeometryBrowser(FWGUIManager *guiManager)
   : TGMainFrame(gClient->GetRoot(), 600, 500),
     m_mode(this, "Mode:", 1l, 0l, 1l),
     m_filter(this,"Materials:",std::string()),
     m_autoExpand(this,"AutoExpand:", 5l, 0l, 1000l),
     m_maxDaughters(this,"MaxChildren:", 999l, 0l, 1000l), // debug
     m_guiManager(guiManager),
     m_tableManager(0),
     m_geometryFile(0),
     m_fileOpen(0),
     m_settersFrame(0)
{
   m_mode.addEntry(0, "Node");
   m_mode.addEntry(1, "Volume");
   
   m_tableManager = new FWGeometryTableManager(this);
 
   TGTextButton* m_fileOpen = new TGTextButton (this, "Open Geometry File");
   this->AddFrame(m_fileOpen,  new TGLayoutHints( kLHintsExpandX , 4, 2, 2, 2));
   m_fileOpen->Connect("Clicked()","FWGeometryBrowser",this,"browse()");


   m_settersFrame = new TGHorizontalFrame(this);
   this->AddFrame( m_settersFrame,new TGLayoutHints(kLHintsExpandX, 2, 2, 2, 2));
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

   m_statBar = new TGStatusBar(this, this->GetWidth(), 12);
   m_statBar->SetText("No simulation geomtery loaded.");
   AddFrame(m_statBar, new TGLayoutHints(kLHintsExpandX));

   SetWindowName("Geometry Browser");
   this->Connect("CloseWindow()","FWGeometryBrowser",this,"windowIsClosing()");
   Layout();
   MapSubwindows();

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
      TGFrameElement *el = (TGFrameElement*) m_settersFrame->GetList()->First();
      m_settersFrame->RemoveFrame(el->fFrame);
   }
   
   TGHorizontalFrame* frame = new TGHorizontalFrame(m_settersFrame);
   m_settersFrame->AddFrame(frame);
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
FWGeometryBrowser::cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t, Int_t)
{
   if (iButton != kButton1)
   {
      // m_tableManager->setSelection(iRow, iColumn, iKeyMod);
      return;   
   }
 
   if (iButton == kButton1 && iColumn == 0)
   {
      m_tableManager->firstColumnClicked(iRow);
   }
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
FWGeometryBrowser::newIndexSelected(int iSelectedRow, int iSelectedColumn)
{
  if (iSelectedRow == -1)
    return;

  m_tableManager->dataChanged();
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

      TGeoManager* m_geoManager = (TGeoManager*) m_geometryFile->Get("cmsGeo;1");
      m_tableManager->loadGeometry(m_geoManager);
      MapRaised();

   }
   catch (std::runtime_error &e)
   {
      fwLog(fwlog::kError) << "Failed to load simulation geomtery.\n";
      updateStatusBar("Failed to load simulation geomtery from file");
   }
}

void
FWGeometryBrowser::browse()
{
   std::cout<<"FWGeometryBrowser::openFile()"<<std::endl;

   if (!geodebug)
   {  
      const char* kRootType[] = {"ROOT files","*.root", 0, 0};
      TGFileInfo fi;
      fi.fFileTypes = kRootType;
 
      m_guiManager->updateStatus("opening geometry file...");

      new TGFileDialog(gClient->GetDefaultRoot(), 
                       (TGWindow*) m_guiManager->getMainFrame(), kFDOpen, &fi);

      m_guiManager->updateStatus("loading geometry file...");
      m_geometryFile = new TFile(fi.fFilename, "READ");
   }
   else
   {
      // AMT developing environment
      m_geometryFile = new TFile("cmsSimGeom-14.root", "READ");
   }
   m_guiManager->clearStatus();

   readFile();
}


void FWGeometryBrowser::updateStatusBar(const char* status) {
   m_statBar->SetText(status, 0);
}
