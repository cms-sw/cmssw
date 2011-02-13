#include "Fireworks/Core/interface/FWGeometryTable.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"

#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWGeometryTableManager.h"
#include "Fireworks/Core/interface/fwLog.h"


#include "TFile.h"
#include "TGFileDialog.h"
#include "TGeoNode.h"
#include "TGComboBox.h"
#include "TGLabel.h"

#include <iostream>
FWGeometryTable::FWGeometryTable(FWGUIManager *guiManager)
   : TGMainFrame(gClient->GetRoot(), 600, 500),
     m_mode(this, "Mode:", 0l, 0l, 1l),
     m_filter(this,"Materials:",std::string()),
     m_autoExpand(this,"AutoExpand:", 2l, 0l, 1000l),
     m_maxDaughters(this,"MaxDaughters:", 999l, 0l, 1000l), // debug
     m_guiManager(guiManager),
     m_tableManager(0),
     m_geometryFile(0),
     m_fileOpen(0),
     m_settersFrame(0)
{
   m_mode.addEntry(0, "Node");
   m_mode.addEntry(1, "Volume");
   
   m_tableManager = new FWGeometryTableManager(this);
 

   // TGCompositeFrame* hf = new TGHorizontalFrame(this);
   //AddFrame(hf, new TGLayoutHints(kLHintsExpandX|kLHintsTop));

   TGTextButton* m_fileOpen = new TGTextButton (this, "Open Geometry File");
   this->AddFrame(m_fileOpen,  new TGLayoutHints( kLHintsExpandX , 2, 2, 2, 2));
   m_fileOpen->Connect("Clicked()","FWGeometryTable",this,"browse()");


   m_settersFrame = new TGHorizontalFrame(this);
   this->AddFrame( m_settersFrame);
   m_settersFrame->SetCleanup(kDeepCleanup);

   m_tableWidget = new FWTableWidget(m_tableManager, this); 
   AddFrame(m_tableWidget,new TGLayoutHints(kLHintsExpandX|kLHintsExpandY|kLHintsBottom,2,2,2,2));
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);
   m_tableWidget->Connect("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",
                          "FWGeometryTable",this,
                          "cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)");
   resetSetters();
   //  openFile();

   SetWindowName("Geometry Browser");
   this->Connect("CloseWindow()","FWGeometryTable",this,"windowIsClosing()");
   MapSubwindows();
}

FWGeometryTable::~FWGeometryTable()
{}

void
FWGeometryTable::resetSetters()
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
   makeSetter(frame, &m_maxDaughters);
   m_settersFrame->MapSubwindows();
   Layout();
}

void
FWGeometryTable::makeSetter(TGCompositeFrame* frame, FWParameterBase* param) 
{
   boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(param) );
   ptr->attach(param, this);
 
   TGFrame* pframe = ptr->build(frame, false);
   frame->AddFrame(pframe, new TGLayoutHints(kLHintsExpandX));
   m_setters.push_back(ptr);
}
//==============================================================================


void
FWGeometryTable::addTo(FWConfiguration& iTo) const
{
   FWConfigurableParameterizable::addTo(iTo);
}
  
void
FWGeometryTable::setFrom(const FWConfiguration& iFrom)
{
   printf("FWGeometryTable::setFrom\n");
   for(const_iterator it =begin(), itEnd = end();
       it != itEnd;
       ++it) {
      (*it)->setFrom(iFrom);      
   }  
   resetSetters();
}

//==============================================================================
void 
FWGeometryTable::cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t, Int_t)
{
   if (iButton != kButton1)
   {
      m_tableManager->setSelection(iRow, iColumn, iKeyMod);
      return;   
   }

   if (iColumn == 0)
   {
      m_tableManager->firstColumnClicked(iRow);
   }
}

bool FWGeometryTable::HandleKey(Event_t *event)
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
FWGeometryTable::windowIsClosing()
{
  UnmapWindow();
}

void
FWGeometryTable::newIndexSelected(int iSelectedRow, int iSelectedColumn)
{
  if (iSelectedRow == -1)
    return;

  m_tableManager->dataChanged();
}

void 
FWGeometryTable::readFile()
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
   }
}

void
FWGeometryTable::browse()
{
   std::cout<<"FWGeometryTable::openFile()"<<std::endl;

   if (1)
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
