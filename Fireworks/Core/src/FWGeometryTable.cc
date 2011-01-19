#include "Fireworks/Core/interface/FWGeometryTable.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"

#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/src/FWTabularWidget.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/TableWidget/interface/FWTextTreeCellRenderer.h"
#include "Fireworks/TableWidget/interface/GlobalContexts.h"

#include "TFile.h"
#include "TGFileDialog.h"
#include "TGWindow.h"
#include "TGeoManager.h"

#include <iostream>

struct NodeInfo
{
  NodeInfo()
    {}
  
  std::string name;
  std::string title;
};


class FWGeometryTableManager : public FWTableManagerBase
{
public:
  FWGeometryTableManager()
    : m_selectedRow(-1)
    { 
      m_renderer.setGraphicsContext(&fireworks::boldGC());
      m_daughterRenderer.setGraphicsContext(&fireworks::italicGC());
      reset();
    }

  virtual void implSort(int, bool)
    {
      recalculateVisibility();
    }

   virtual int unsortedRowNumber(int unsorted) const
    {
      return unsorted;
    }

   virtual int numberOfRows() const 
    {
      return m_row_to_index.size();
    }

   virtual int numberOfColumns() const 
    {
      return 2;
    }
   
  virtual std::vector<std::string> getTitles() const 
    {
      std::vector<std::string> returnValue;
      returnValue.reserve(numberOfColumns());
      returnValue.push_back("Name");
      returnValue.push_back("Title");
      return returnValue;
    }
  
  virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const
    {
      std::cout<<"m_row_to_index.size(), iSortedRowNumber: "
               << m_row_to_index.size() <<" "<< iSortedRowNumber <<std::endl;

      if (static_cast<int>(m_row_to_index.size()) <= iSortedRowNumber)
      {
        m_renderer.setData(std::string("Ack!"), false);
        return &m_renderer;
      }       

      FWTextTreeCellRenderer* renderer = &m_renderer;
      int unsortedRow =  m_row_to_index[iSortedRowNumber];
      const NodeInfo& data = m_nodeInfo[unsortedRow];

      std::string name;
      std::string title;

      name = data.name;
      title = data.title;

      std::cout<<"name, title: "<< name <<"  "<< title <<std::endl;

      if (iCol == 0)
        renderer->setData(name, false);
      else if (iCol == 1)
        renderer->setData(title, false);
      else
         renderer->setData(std::string(), false);


      renderer->setIndentation(0);
      return renderer;
   }

  void setExpanded(int row)
    {
      if ( row == -1 )
        return;
      
      recalculateVisibility();
      dataChanged();
      visualPropertiesChanged();
    }
  
   void setSelection (int row, int column, int mask) 
    {
      if(mask == 4) 
      {
        if( row == m_selectedRow) 
        {
          row = -1;
        }
      }
      changeSelection(row, column);
    }

  virtual const std::string title() const 
    {
      return "Geometry";
    }

   int selectedRow() const 
    {
      return m_selectedRow;
    }

   int selectedColumn() const 
    {
      return m_selectedColumn;
    }
 
   virtual bool rowIsSelected(int row) const 
    {
      return m_selectedRow == row;
    }

  void reset() 
    {
      changeSelection(-1, -1);
      recalculateVisibility();
      dataChanged();
      visualPropertiesChanged();
    }

  void recalculateVisibility()
    {
      m_row_to_index.clear();
        
      for ( size_t i = 0, e = m_nodeInfo.size(); i != e; ++i )
          m_row_to_index.push_back(i);
    } 

  std::vector<int> &rowToIndex() { return m_row_to_index; }
  sigc::signal<void,int,int> indexSelected_;

  void fillNodeInfo(TGeoManager* geoManager)
    {
      std::cout<<"fillNodeInfo in"<<std::endl;
     
      TGeoVolume* topVolume = geoManager->GetTopVolume();

      NodeInfo topNodeInfo;
      topNodeInfo.name = geoManager->GetTopNode()->GetName();
      topNodeInfo.title = geoManager->GetTopNode()->GetTitle();
      m_nodeInfo.push_back(topNodeInfo);

      for ( size_t n = 0, 
                  ne = topVolume->GetNode(0)->GetNdaughters();
            n != ne; ++n )
      {
        NodeInfo nodeInfo;
        nodeInfo.name = topVolume->GetNode(0)->GetDaughter(n)->GetName();
        nodeInfo.title = topVolume->GetNode(0)->GetDaughter(n)->GetTitle();
        m_nodeInfo.push_back(nodeInfo);
      }

      /*
      TObjArray* listOfNodes = geoManager->GetListOfNodes();
      TGeoNode* node;

      for ( int i = 0; i <= listOfNodes->GetLast(); ++i )
      {
        if ( (node = (TGeoNode*)listOfNodes->At(i)) )
        {
          node->Print();
        }
      }
      */
      
      std::cout<<"fillNodeInfo out"<<std::endl;
    }
  

private:
  void changeSelection(int iRow, int iColumn)
    {      
      if (iRow == m_selectedRow && iColumn == m_selectedColumn)
        return;
      
      m_selectedRow = iRow;
      m_selectedColumn = iColumn;

      indexSelected_(iRow, iColumn);
      visualPropertiesChanged();
    }    

  std::vector<int>  m_row_to_index;
  int               m_selectedRow;
  int               m_selectedColumn;
  
  std::vector<NodeInfo> m_nodeInfo;

  mutable FWTextTreeCellRenderer m_renderer;         
  mutable FWTextTreeCellRenderer m_daughterRenderer;  
};

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
  std::cout<<"FWGeometryTable::FWGeometryTable(FWGUIManager *guiManager) in"<<std::endl;

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

  
  
  //m_fileOpen->Connect("Clicked()", "FWGeometryTable", this, "openFile()");
  //m_fileOpen->SetEnabled(true);
 
  MapSubwindows();
  Layout();

  std::cout<<"FWGeometryTable::FWGeometryTable(FWGUIManager *guiManager) out"<<std::endl;
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
    std::cout<<"FWGeometryTable::readFile() no geometry file!"<<std::endl;
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
  //m_tableWidget->body()->DoRedraw();
}

void
FWGeometryTable::openFile()
{
  std::cout<<"FWGeometryTable::openFile()"<<std::endl;
  
  const char* kRootType[] = {"ROOT files","*.root", 0, 0};
  TGFileInfo fi;
  fi.fFileTypes = kRootType;
 
  m_guiManager->updateStatus("opening geometry file...");

  new TGFileDialog(gClient->GetDefaultRoot(), 
                   (TGWindow*) m_guiManager->getMainFrame(), kFDOpen, &fi);

  m_guiManager->updateStatus("loading geometry file...");
  
  if ( fi.fFilename ) 
    m_geometryFile = new TFile(fi.fFilename, "READ");

  m_guiManager->clearStatus();

  readFile();
}
