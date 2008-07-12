// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGUIEventDataAdder
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 13 09:58:53 EDT 2008
// $Id: FWGUIEventDataAdder.cc,v 1.6 2008/07/11 01:15:22 dmytro Exp $
//

// system include files
#include <iostream>
#include <sigc++/signal.h>
#include <boost/bind.hpp>
#include "TGFrame.h"
#include "TGTextEntry.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TClass.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

// user include files
#include "Fireworks/Core/src/FWGUIEventDataAdder.h"
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/src/LightTableWidget.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/FWLite/interface/Event.h"
//
// constants, enums and typedefs
//
static const std::string& dataForColumn( const FWGUIEventDataAdder::Data& iData, int iCol)
{
   switch (iCol) {
      case 0:
         return iData.purpose_;
         break;
      case 3:
         return iData.type_;
         break;
      case 1:
         return iData.moduleLabel_;
         break;
      case 2:
         return iData.productInstanceLabel_;
         break;
      case 4:
         return iData.processName_;
         break;
      default:
         break;
   }
   static const std::string s_blank;
   return s_blank;
}

static const unsigned int kNColumns = 5;
class DataAdderTableManager : public LightTableManager {
public:
   DataAdderTableManager(const std::vector<FWGUIEventDataAdder::Data>* iData):
   m_data(iData), m_selectedRow(-1) { dataChanged();}
   
   virtual int NumberOfRows() const { return m_data->size();}
   virtual int NumberOfCols() const { return kNColumns;}
   virtual void Sort(int col, bool sortOrder);
   virtual std::vector<std::string> GetTitles(int col) {
      std::vector<std::string> returnValue;
      returnValue.reserve(kNColumns);
      returnValue.push_back("Purpose");
      returnValue.push_back("Module Label");
      returnValue.push_back("Product Instance Label");
      returnValue.push_back("C++ Class");
      returnValue.push_back("Process Name");
      return returnValue;
   }
   virtual void FillCells(int rowStart, int colStart, 
                          int rowEnd, int colEnd, std::vector<std::string>& oToFill)
   {
      oToFill.clear();
      assert(rowStart <=rowEnd);
      assert(rowEnd <= static_cast<int>(m_data->size()));
      assert(m_data->size() == m_row_to_index.size());
      assert(colStart <= colEnd);
      assert(colEnd <= static_cast<int>(kNColumns));
      oToFill.reserve((rowEnd-rowStart)*(colEnd-colStart));
      for(int row= rowStart; row != rowEnd; ++row) {
         const FWGUIEventDataAdder::Data& data = (*m_data)[m_row_to_index[row]];
         for(int col=colStart; col!=colEnd; ++col) {
            oToFill.push_back(dataForColumn(data,col));
         }
      }
   }
   
   void Selection (int row, int mask) {
      if(mask == 4) {
         if( row == m_selectedRow) {
            row = -1;
         }
      }
      changeSelection(row);
   }

   virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame) {return 0;}
   virtual void UpdateRowCell(int row, TGFrame *rowCell) {}
   virtual const std::string title() const { return "Viewable Collections"; }
   
   int selectedRow() const {
      return m_selectedRow;
   }
   //virtual void sort (int col, bool reset = false);
   virtual bool rowIsSelected(int row) const {
      return m_selectedRow == row;
   }
   
   void dataChanged() {
      changeSelection(-1);
      m_row_to_index.clear();
      m_row_to_index.reserve(m_data->size());
      for(unsigned int i =0; i < m_data->size(); ++i) {
         m_row_to_index.push_back(i);
      }
   }
   sigc::signal<void,int> indexSelected_;
private:
   void changeSelection(int iRow) {
      if(iRow != m_selectedRow) {
         m_selectedRow=iRow;
         if(-1 == iRow) {
            indexSelected_(-1);
         } else {
            indexSelected_(m_row_to_index[iRow]);
         }
      }
   }
   const std::vector<FWGUIEventDataAdder::Data>* m_data;
   std::vector<int> m_row_to_index;
   int m_selectedRow;
};

namespace {
   template <typename TMap>
   void doSort(int col, 
               const std::vector<FWGUIEventDataAdder::Data>& iData,
               TMap& iOrdered, 
               std::vector<int>& oRowToIndex,
               int& ioSelectedRow)
   {
      int selectedIndex = -1;
      if(ioSelectedRow != -1) {
         selectedIndex = oRowToIndex[ioSelectedRow];
      }
      unsigned int index=0;
      for(std::vector<FWGUIEventDataAdder::Data>::const_iterator it = iData.begin(),
          itEnd = iData.end();
          it!=itEnd;
          ++it,++index) {
         iOrdered.insert(std::make_pair(dataForColumn(*it,col),index));
      }
      unsigned int row = 0;
      for(typename TMap::iterator it = iOrdered.begin(),
             itEnd = iOrdered.end();
          it != itEnd;
          ++it,++row) {
         if(it->second == selectedIndex) {
               ioSelectedRow = row;
         }
         oRowToIndex[row]=it->second;
      }
   }
}

void 
DataAdderTableManager::Sort(int col, bool sortOrder)
{
   if(sortOrder) {
      std::multimap<std::string,int> ordered;
      doSort(col,*m_data, ordered, m_row_to_index,m_selectedRow);
   }else {
      std::multimap<std::string,int,std::greater<std::string> > ordered;
      doSort(col,*m_data, ordered, m_row_to_index,m_selectedRow);
   }
}


//
// static data member definitions
//

//
// constructors and destructor
//
static void addToFrame(TGVerticalFrame* iParent, const char* iName, TGTextEntry*& oSet)
{
   TGCompositeFrame* hf = new TGHorizontalFrame(iParent);
   hf->AddFrame(new TGLabel(hf,iName),new TGLayoutHints(kLHintsLeft|kLHintsCenterY,2,2,2,2));
   oSet = new TGTextEntry(hf,"");
   hf->AddFrame(oSet,new TGLayoutHints(kLHintsExpandX|kLHintsCenterY));
   iParent->AddFrame(hf);
}

FWGUIEventDataAdder::FWGUIEventDataAdder(
                                         UInt_t iWidth,UInt_t iHeight, 
                                         FWEventItemsManager* iManager,
                                         TGFrame* iParent,
                                         const fwlite::Event* iEvent,
                                         const TFile* iFile,
const std::set<std::pair<std::string,std::string> >& iTypeAndPurpose):
m_manager(iManager),
m_presentEvent(0),
m_parentFrame(iParent),
m_typeAndPurpose(iTypeAndPurpose)
{
   createWindow();
   update(iFile,iEvent);
}

// FWGUIEventDataAdder::FWGUIEventDataAdder(const FWGUIEventDataAdder& rhs)
// {
//    // do actual copying here;
// }

FWGUIEventDataAdder::~FWGUIEventDataAdder()
{
}

//
// assignment operators
//
// const FWGUIEventDataAdder& FWGUIEventDataAdder::operator=(const FWGUIEventDataAdder& rhs)
// {
//   //An exception safe implementation is
//   FWGUIEventDataAdder temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWGUIEventDataAdder::addNewItem()
{
   TClass* theClass = TClass::GetClass(m_type->GetText());
   if(0==theClass) {
      return;
   }
   const std::string moduleLabel = m_moduleLabel->GetText();
   if(moduleLabel.empty()) {
      return;
   }
   
   const std::string name = m_name->GetText();
   if(name.empty()) {
      return;
   }
   
   if ( m_manager->find( name ) ) {
      std::cout << "Event item " << name << 
	" is already registered. Please use another name" << std::endl;
      return;
   }
   
   FWPhysicsObjectDesc desc(name, theClass, m_purpose->GetText(),
                            FWDisplayProperties(),
                            moduleLabel,
                            m_productInstanceLabel->GetText(),
                            m_processName->GetText());
   m_manager->add( desc);
   if (m_frame) m_frame->CloseWindow();
}

void 
FWGUIEventDataAdder::show()
{
   // Map main frame 
   if(0==m_frame) {
      createWindow();
      m_tableWidget->Reinit();
   }
   m_frame->MapWindow(); 
}

void
FWGUIEventDataAdder::windowIsClosing()
{
   // m_frame->Cleanup();
   // delete m_frame;
   m_frame=0;
   // delete m_tableWidget;
   m_tableWidget=0;
   delete m_tableManager;
   m_tableManager=0;
}


void
FWGUIEventDataAdder::createWindow()
{
   m_frame = new TGTransientFrame(gClient->GetDefaultRoot(),m_parentFrame,600,400);
   // m_frame->MapWindow(); 
   //m_frame->SetCleanup(kDeepCleanup);
   m_frame->Connect("CloseWindow()","FWGUIEventDataAdder",this,"windowIsClosing()");
   TGVerticalFrame* vf = new TGVerticalFrame(m_frame);
   m_frame->AddFrame(vf, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,10,10,10,10));
   
   addToFrame(vf, "Name:", m_name);
   addToFrame(vf, "Purpose:", m_purpose);
   addToFrame(vf,"C++ type:",m_type);
   addToFrame(vf,"Module label:",m_moduleLabel);
   addToFrame(vf,"Product instance label:",m_productInstanceLabel);
   addToFrame(vf,"Process name:",m_processName);
   
   m_tableManager= new DataAdderTableManager(&m_useableData);
   m_tableManager->indexSelected_.connect(boost::bind(&FWGUIEventDataAdder::newIndexSelected,this,_1));
   m_tableWidget = new LightTableWidget(vf,m_tableManager,600,400);
   // vf->AddFrame(m_tableWidget, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   
   m_apply = new TGTextButton(vf,"Add Data");
   vf->AddFrame(m_apply, new TGLayoutHints(kLHintsBottom|kLHintsCenterX));
   m_apply->Connect("Clicked()","FWGUIEventDataAdder",this,"addNewItem()");
   
   // Set a name to the main frame 
   m_frame->SetWindowName("Show Additional Event Data"); 
   
   // Map all subwindows of main frame 
   m_frame->MapSubwindows(); 
   
   // Initialize the layout algorithm 
   m_frame->Layout();    

   // Initialize the layout algorithm 
   m_frame->Resize(m_frame->GetDefaultSize()); 
   
}

void
FWGUIEventDataAdder::update(const TFile* iFile, const fwlite::Event* iEvent)
{
   if(m_presentEvent != iEvent) {
      m_presentEvent=iEvent;
      assert(0!=iFile);
      fillData(iFile);
   }
}

void 
FWGUIEventDataAdder::fillData(const TFile* iFile)
{
   m_useableData.clear();
   if(0!=m_presentEvent) {
      static const std::string s_blank;
      const std::vector<edm::BranchDescription>& branches = 
      m_presentEvent->getBranchDescriptions();
      Data d;
      
      //I'm not going to modify TFile but I need to see what it is holding
      TTree* eventsTree = dynamic_cast<TTree*>(const_cast<TFile*>(iFile)->Get("Events"));
      assert(0!=eventsTree);
      
      std::set<std::string> branchNamesInFile;
      TIter nextBranch(eventsTree->GetListOfBranches());
      while(TBranch* branch = static_cast<TBranch*>(nextBranch())) {
         branchNamesInFile.insert(branch->GetName());
      }
      
      
      for(std::vector<edm::BranchDescription>::const_iterator itBranch =
          branches.begin(), itEnd=branches.end();
          itBranch != itEnd;
          ++itBranch) {
         if(itBranch->present() && 
            branchNamesInFile.end() != branchNamesInFile.find(itBranch->branchName())){
            std::set<std::pair<std::string,std::string> >::iterator itTP =
            m_typeAndPurpose.upper_bound(std::make_pair(itBranch->fullClassName(),s_blank));
            //std::cout <<"try to find match "<<itBranch->fullClassName()<<std::endl;
            while(itTP != m_typeAndPurpose.end() &&
                  itTP->first == itBranch->fullClassName()) {
               d.purpose_ = itTP->second;
               d.type_ = itBranch->fullClassName();
               d.moduleLabel_ = itBranch->moduleLabel();
               d.productInstanceLabel_ = itBranch->productInstanceName();
               d.processName_ = itBranch->processName();
               m_useableData.push_back(d);
               /*
               std::cout <<d.purpose_<<" "<<d.type_<<" "
            <<d.moduleLabel_<<" "
               <<d.productInstanceLabel_<<" "
               <<d.processName_<<std::endl;
                */
               ++itTP;
            }
         }
      }      
      m_tableManager->dataChanged();
      m_tableWidget->Reinit();
   }
}

void 
FWGUIEventDataAdder::newIndexSelected(int iSelectedIndex)
{
   if(-1 != iSelectedIndex) {
      m_purpose->SetText(m_useableData[iSelectedIndex].purpose_.c_str());
      m_type->SetText(m_useableData[iSelectedIndex].type_.c_str());
      m_moduleLabel->SetText(m_useableData[iSelectedIndex].moduleLabel_.c_str());
      m_productInstanceLabel->SetText(m_useableData[iSelectedIndex].productInstanceLabel_.c_str());
      m_processName->SetText(m_useableData[iSelectedIndex].processName_.c_str());
      std::set<int> selectedRows;
      selectedRows.insert(m_tableManager->selectedRow());
      m_tableWidget->SelectRows(selectedRows);
   }
}

//
// const member functions
//

//
// static member functions
//
