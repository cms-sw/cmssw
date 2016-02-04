#include "Fireworks/Core/interface/FWGeometryTable.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"

#include "Fireworks/Core/interface/FWGeometryTableManager.h"


#include "TFile.h"
#include "TGFileDialog.h"
#include "TGeoNode.h"
#include "TGWindow.h"

#include <iostream>

FWGeometryTable::FWGeometryTable(FWGUIManager *guiManager)
  : TGMainFrame(gClient->GetRoot(), 400, 600),
    m_guiManager(guiManager),
    m_geometryTable(new FWGeometryTableManager()),
    m_geometryFile(0),
    m_fileOpen(0),
    m_topNode(0),
    m_topVolume(0),
    m_level(-1)
{
  gVirtualX->SelectInput(GetId(), kKeyPressMask | kKeyReleaseMask | kExposureMask |
                         kPointerMotionMask | kStructureNotifyMask | kFocusChangeMask |
                         kEnterWindowMask | kLeaveWindowMask);
  this->Connect("CloseWindow()","FWGeometryTable",this,"windowIsClosing()");

  FWDialogBuilder builder(this);
  builder.indent(4)
    .spaceDown(10)
    //.addTextButton("Open geometry file", &m_fileOpen) 
    .addLabel("Filter:").floatLeft(4).expand(false, false)
    .addTextEntry("", &m_search).expand(true, false)
    .spaceDown(10)
    .addTable(m_geometryTable, &m_tableWidget).expand(true, true);

  openFile();
    
  m_tableWidget->SetBackgroundColor(0xffffff);
  m_tableWidget->SetLineSeparatorColor(0x000000);
  m_tableWidget->SetHeaderBackgroundColor(0xececec);
  m_tableWidget->Connect("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",
                         "FWGeometryTable",this,
                         "cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)");

  MapSubwindows();
  Layout();
}

FWGeometryTable::~FWGeometryTable()
{}

void 
FWGeometryTable::cellClicked(Int_t iRow, Int_t iColumn, Int_t iButton, Int_t iKeyMod, Int_t, Int_t)
{
  if (iButton != kButton1)
    return;   

  m_geometryTable->setExpanded(iRow);
  m_geometryTable->setSelection(iRow, iColumn, iKeyMod);
}

void
FWGeometryTable::windowIsClosing()
{
  UnmapWindow();
  DontCallClose();
}

void
FWGeometryTable::newIndexSelected(int iSelectedRow, int iSelectedColumn)
{
  if (iSelectedRow == -1)
    return;

  m_geometryTable->dataChanged();
}

void 
FWGeometryTable::handleNode(const TGeoNode* node)
{
  for ( size_t d = 0, de = node->GetNdaughters(); d != de; ++d )
  {
    handleNode(node->GetDaughter(d));
  }
}

void 
FWGeometryTable::readFile()
{
   if ( ! m_geometryFile )
   {
      std::cout<<"FWGeometryTable::readFile() no geometry file!"<< std::endl;
      return;
   }
  
   m_geometryFile->ls();
      
   TGeoManager* m_geoManager = (TGeoManager*) m_geometryFile->Get("cmsGeo;1");

   /*
     m_topVolume = m_geoManager->GetTopVolume();
     m_topVolume->Print();

     m_topNode   = m_geoManager->GetTopNode();
     m_topNode->Print();

     for ( size_t n = 0, 
     ne = m_topVolume->GetNode(0)->GetNdaughters();
     n != ne; ++n )
     {
     m_topVolume->GetNode(0)->GetDaughter(n)->Print();
     }
   */

   m_geometryTable->fillNodeInfo(m_geoManager);
}

void
FWGeometryTable::openFile()
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
