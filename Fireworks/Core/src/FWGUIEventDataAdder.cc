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
// $Id: FWGUIEventDataAdder.cc,v 1.20 2009/06/06 21:28:58 chrjones Exp $
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
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/FWLite/interface/Event.h"

//Had to hide this type from Cint
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"
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
class DataAdderTableManager : public FWTableManagerBase {
public:
   DataAdderTableManager(const std::vector<FWGUIEventDataAdder::Data>* iData) :
      m_data(iData), m_selectedRow(-1) {
      reset();
   }

   virtual int numberOfRows() const {
      return m_data->size();
   }
   virtual int numberOfColumns() const {
      return kNColumns;
   }
   
   virtual int unsortedRowNumber(int iSortedRowNumber) const {
      return m_row_to_index[iSortedRowNumber];
   }

   virtual void implSort(int col, bool sortOrder);
   virtual std::vector<std::string> getTitles() const {
      std::vector<std::string> returnValue;
      returnValue.reserve(kNColumns);
      returnValue.push_back("Purpose");
      returnValue.push_back("Module Label");
      returnValue.push_back("Product Instance Label");
      returnValue.push_back("C++ Class");
      returnValue.push_back("Process Name");
      return returnValue;
   }
   
   virtual FWTableCellRendererBase* cellRenderer(int iSortedRowNumber, int iCol) const
   {
      
      if(static_cast<int>(m_row_to_index.size())>iSortedRowNumber) {
         int unsortedRow =  m_row_to_index[iSortedRowNumber];
         const FWGUIEventDataAdder::Data& data = (*m_data)[unsortedRow];

         m_renderer.setData(dataForColumn(data,iCol),m_selectedRow==unsortedRow);
      } else {
         m_renderer.setData(std::string(),false);
      }
      return &m_renderer;
   }

   void setSelection (int row, int mask) {
      if(mask == 4) {
         if( row == m_selectedRow) {
            row = -1;
         }
      }
      changeSelection(row);
   }

   virtual const std::string title() const {
      return "Viewable Collections";
   }

   int selectedRow() const {
      return m_selectedRow;
   }
   //virtual void sort (int col, bool reset = false);
   virtual bool rowIsSelected(int row) const {
      return m_selectedRow == row;
   }

   void reset() {
      changeSelection(-1);
      m_row_to_index.clear();
      m_row_to_index.reserve(m_data->size());
      for(unsigned int i =0; i < m_data->size(); ++i) {
         m_row_to_index.push_back(i);
      }
      dataChanged();
   }
   sigc::signal<void,int> indexSelected_;
private:
   void changeSelection(int iRow) {
      if(iRow != m_selectedRow) {
         m_selectedRow=iRow;
         if(-1 == iRow) {
            indexSelected_(-1);
         } else {
            indexSelected_(iRow);
         }
         visualPropertiesChanged();
      }
   }
   const std::vector<FWGUIEventDataAdder::Data>* m_data;
   std::vector<int> m_row_to_index;
   int m_selectedRow;
   mutable FWTextTableCellRenderer m_renderer;
};

namespace {
   template <typename TMap>
   void doSort(int col,
               const std::vector<FWGUIEventDataAdder::Data>& iData,
               TMap& iOrdered,
               std::vector<int>& oRowToIndex)
   {
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
         oRowToIndex[row]=it->second;
      }
   }
}

void
DataAdderTableManager::implSort(int col, bool sortOrder)
{
   if(sortOrder) {
      std::multimap<std::string,int> ordered;
      doSort(col,*m_data, ordered, m_row_to_index);
   } else {
      std::multimap<std::string,int,std::greater<std::string> > ordered;
      doSort(col,*m_data, ordered, m_row_to_index);
   }
}


//
// static data member definitions
//

//
// constructors and destructor
//
static TGLayoutHints* addToFrame(TGVerticalFrame* iParent, const char* iName, TGTextEntry*& oSet,
                                 unsigned int& oLabelWidth)
{
   TGLayoutHints* returnValue = new TGLayoutHints(kLHintsLeft|kLHintsCenterY,2,2,2,2);
   TGCompositeFrame* hf = new TGHorizontalFrame(iParent);
   TGLabel* label = new TGLabel(hf,iName);
   oLabelWidth= label->GetWidth();
   hf->AddFrame(label, returnValue);
   oSet = new TGTextEntry(hf,"");
   hf->AddFrame(oSet,new TGLayoutHints(kLHintsExpandX|kLHintsCenterY));
   iParent->AddFrame(hf, new TGLayoutHints(kLHintsExpandX));
   return returnValue;
}

FWGUIEventDataAdder::FWGUIEventDataAdder(
   UInt_t iWidth,UInt_t iHeight,
   FWEventItemsManager* iManager,
   TGFrame* iParent,
   const fwlite::Event* iEvent,
   const TFile* iFile,
   const FWTypeToRepresentations& iTypeAndReps) :
   m_manager(iManager),
   m_presentEvent(0),
   m_parentFrame(iParent),
   m_typeAndReps( new FWTypeToRepresentations(iTypeAndReps))
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
   delete m_typeAndReps;
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
   TClass* theClass = TClass::GetClass(m_type.c_str());
   if(0==theClass) {
      return;
   }
   const std::string moduleLabel = m_moduleLabel;
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

   int largest = -1;
   if(m_manager->begin() != m_manager->end()) {
      if ( *(m_manager->begin()) )
         largest = (*(m_manager->begin()))->layer();
   }
   for(FWEventItemsManager::const_iterator it = m_manager->begin(),
                                           itEnd = m_manager->end();
       it!=itEnd;
       ++it) {
      if((*it) && largest < (*it)->layer()) {
         largest = (*it)->layer();
      }
   }
   ++largest;
   std::string processName = m_processName;
   if(m_doNotUseProcessName->IsOn()) {
      processName="";
   }
   FWPhysicsObjectDesc desc(name, theClass, m_purpose,
                            FWDisplayProperties(),
                            moduleLabel,
                            m_productInstanceLabel,
                            processName,
                            "",
                            largest);
   m_manager->add( desc);
   if (m_frame) m_frame->UnmapWindow();
}

void
FWGUIEventDataAdder::show()
{
   // Map main frame
   if(0==m_frame) {
      createWindow();
      //m_tableWidget->Reinit();
   }
   m_frame->MapWindow();
}

void
FWGUIEventDataAdder::windowIsClosing()
{
   m_name->SetText("");
   m_purpose.clear();
   m_type.clear();
   m_moduleLabel.clear();
   m_processName.clear();
   m_productInstanceLabel.clear();
   m_apply->SetEnabled(false);
   
   m_frame->UnmapWindow();
   m_frame->DontCallClose();
   /*
      // m_frame->Cleanup();
      // delete m_frame;
      m_frame=0;
      // delete m_tableWidget;
      m_tableWidget=0;
      delete m_tableManager;
      m_tableManager=0;
    */
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

   unsigned int maxWidth = 0;
   std::vector<TGLayoutHints*> hints(1);
   std::vector<unsigned int> widths(1);
   assert(1==hints.size());
   unsigned int index = 0;
   hints[index]=addToFrame(vf, "Name:", m_name,widths[index]);
   if(widths[index] > maxWidth) {maxWidth = widths[index];}
   ++index;
   m_doNotUseProcessName= new TGCheckButton(vf,"Do not use Process Name and instead only get this data from the most recent Process",1);
   m_doNotUseProcessName->SetState(kButtonDown);
   vf->AddFrame(m_doNotUseProcessName);
   /*
   hints[index]=addToFrame(vf, "Purpose:", m_purpose,widths[index]);
   if(widths[index] > maxWidth) {maxWidth = widths[index];}
   ++index;
   hints[index]=addToFrame(vf,"C++ type:",m_type,widths[index]);
   if(widths[index] > maxWidth) {maxWidth = widths[index];}
   ++index;
   hints[index]=addToFrame(vf,"Module label:",m_moduleLabel,widths[index]);
   if(widths[index] > maxWidth) {maxWidth = widths[index];}
   ++index;
   hints[index]=addToFrame(vf,"Product instance label:",m_productInstanceLabel,widths[index]);
   if(widths[index] > maxWidth) {maxWidth = widths[index];}
   ++index;
   hints[index]=addToFrame(vf,"Process name:",m_processName,widths[index]);
   if(widths[index] > maxWidth) {maxWidth = widths[index];}
    */
   std::vector<unsigned int>::iterator itW = widths.begin();
   for(std::vector<TGLayoutHints*>::iterator itH = hints.begin(), itEnd = hints.end();
       itH != itEnd;
       ++itH,++itW) {
      (*itH)->SetPadLeft(maxWidth - *itW);
   }

   TGLabel* label = new TGLabel(vf,"Viewable Collections");
   vf->AddFrame(label,new TGLayoutHints(kLHintsNormal,0,0,10));
   m_tableManager= new DataAdderTableManager(&m_useableData);
   m_tableManager->indexSelected_.connect(boost::bind(&FWGUIEventDataAdder::newIndexSelected,this,_1));
   m_tableWidget = new FWTableWidget(m_tableManager,vf);
   m_tableWidget->Resize(200,200);
   vf->AddFrame(m_tableWidget, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   m_tableWidget->Connect("rowClicked(Int_t,Int_t,Int_t)","FWGUIEventDataAdder",this,"rowClicked(Int_t,Int_t,Int_t)");

   m_apply = new TGTextButton(vf,"Add Data");
   vf->AddFrame(m_apply, new TGLayoutHints(kLHintsBottom|kLHintsCenterX));
   m_apply->Connect("Clicked()","FWGUIEventDataAdder",this,"addNewItem()");
   m_apply->SetEnabled(false);

   // Set a name to the main frame
   m_frame->SetWindowName("Add Collection");

   // Map all subwindows of main frame
   m_frame->MapSubwindows();

   // Initialize the layout algorithm
   m_frame->Layout();

   // Initialize the layout algorithm
   //m_frame->Resize(m_frame->GetDefaultSize());

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


      std::set<std::string> purposes;
      for(std::vector<edm::BranchDescription>::const_iterator itBranch =
             branches.begin(), itEnd=branches.end();
          itBranch != itEnd;
          ++itBranch) {
         if(itBranch->present() &&
            branchNamesInFile.end() != branchNamesInFile.find(itBranch->branchName())){
            const std::vector<FWRepresentationInfo>& infos = m_typeAndReps->representationsForType(itBranch->fullClassName());

            //std::cout <<"try to find match "<<itBranch->fullClassName()<<std::endl;
            //the infos list can contain multiple items with the same purpose so we will just find
            // the unique ones
            purposes.clear();
            for(std::vector<FWRepresentationInfo>::const_iterator itInfo = infos.begin(),
                                                                  itInfoEnd = infos.end();
                itInfo != itInfoEnd;
                ++itInfo) {
               purposes.insert(itInfo->purpose());
            }
            for(std::set<std::string>::const_iterator itPurpose = purposes.begin(),
                                                      itEnd = purposes.end();
                itPurpose != itEnd;
                ++itPurpose) {
               d.purpose_ = *itPurpose;
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
            }
         }
      }
      m_tableManager->reset();
      m_tableManager->sort(0,true);
      //m_tableWidget->dataChanged();
      //m_tableWidget->Reinit();
   }
}

void
FWGUIEventDataAdder::newIndexSelected(int iSelectedIndex)
{
   if(-1 != iSelectedIndex) {
      m_purpose =m_useableData[iSelectedIndex].purpose_;
      m_type = m_useableData[iSelectedIndex].type_;
      std::string oldModuleLabel = m_moduleLabel;
      m_moduleLabel = m_useableData[iSelectedIndex].moduleLabel_;
      m_productInstanceLabel = m_useableData[iSelectedIndex].productInstanceLabel_;
      m_processName = m_useableData[iSelectedIndex].processName_;
      
      if(strlen(m_name->GetText())==0 || oldModuleLabel == m_name->GetText()) {
         m_name->SetText(m_moduleLabel.c_str());
      }
      m_apply->SetEnabled(true);
      //std::set<int> selectedRows;
      //selectedRows.insert(m_tableManager->selectedRow());
      //m_tableWidget->SelectRows(selectedRows);
   }
}

void 
FWGUIEventDataAdder::rowClicked(Int_t iRow,Int_t iButton,Int_t iKeyMod)
{
   if(iButton==kButton1) {
      m_tableManager->setSelection(iRow,iKeyMod);
   }
}

//
// const member functions
//

//
// static member functions
//
