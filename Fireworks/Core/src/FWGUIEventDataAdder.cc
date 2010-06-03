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
// $Id: FWGUIEventDataAdder.cc,v 1.38 2010/06/03 19:27:59 chrjones Exp $
//

// system include files
#include <iostream>
#include <sigc++/signal.h>
#include <boost/bind.hpp>
#include <boost/regex.hpp>
#include <algorithm>
#include "TGFrame.h"
#include "TGTextEntry.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TGMsgBox.h"
#include "TClass.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

// user include files
#include "Fireworks/Core/src/FWGUIEventDataAdder.h"
#include "Fireworks/Core/interface/FWPhysicsObjectDesc.h"
#include "Fireworks/Core/interface/FWEventItemsManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWItemAccessorFactory.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/Core/interface/fwLog.h"

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
      case 4:
         return iData.type_;
         break;
      case 1:
         return iData.moduleLabel_;
         break;
      case 2:
         return iData.productInstanceLabel_;
         break;
      case 3:
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
      returnValue.push_back("Process Name");
      returnValue.push_back("C++ Class");
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
   update(iFile, iEvent);
}

// FWGUIEventDataAdder::FWGUIEventDataAdder(const FWGUIEventDataAdder& rhs)
// {
//    // do actual copying here;
// }

FWGUIEventDataAdder::~FWGUIEventDataAdder()
{
   delete m_typeAndReps;

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
      TString msg("Event item '");
      msg += name;
      msg += "' is already registered. Please use another name.";
      fwLog(fwlog::kWarning) << msg.Data() << std::endl;
      new TGMsgBox(gClient->GetDefaultRoot(), m_frame,
                   "Error - Name conflict", msg, kMBIconExclamation, kMBOk);
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
   if(m_doNotUseProcessName->IsOn() && m_doNotUseProcessName->IsEnabled()) {
      processName="";
   }
   FWPhysicsObjectDesc desc(name, theClass, m_purpose,
                            FWDisplayProperties::defaultProperties,
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
   }
   m_frame->MapWindow();
}

void
FWGUIEventDataAdder::windowIsClosing()
{
   m_name->SetText("");
   m_search->SetText("");
   m_purpose.clear();
   m_type.clear();
   m_moduleLabel.clear();
   m_processName.clear();
   m_productInstanceLabel.clear();
   m_apply->SetEnabled(false);
   
   m_frame->UnmapWindow();
   m_frame->DontCallClose();
}

void
FWGUIEventDataAdder::createWindow()
{
   m_frame = new TGTransientFrame(gClient->GetDefaultRoot(),m_parentFrame,600,400);
   m_frame->Connect("CloseWindow()","FWGUIEventDataAdder",this,"windowIsClosing()");

   TGVerticalFrame* vf = new TGVerticalFrame(m_frame);
   m_frame->AddFrame(vf, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY,10,10,10,10));

   unsigned int maxWidth = 0;
   std::vector<TGLayoutHints*> hints(1);
   std::vector<unsigned int> widths(1);
   assert(1==hints.size());

   unsigned int index = 0;

   hints[index] = addToFrame(vf, "", m_search, widths[index]);
   
   if ( widths[index] > maxWidth ) 
   {
     maxWidth = widths[index];
   }

   ++index;
    
   TGHorizontalFrame* searchButtonFrame = new TGHorizontalFrame(vf);
   vf->AddFrame(searchButtonFrame, new TGLayoutHints(kLHintsTop | kLHintsCenterX));

   TGTextButton* searchButton = new TGTextButton(searchButtonFrame, "Search");
   searchButtonFrame->AddFrame(searchButton, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 0));
   searchButton->Connect("Clicked()", "FWGUIEventDataAdder", this, "searchData()");
  
   std::vector<unsigned int>::iterator itW = widths.begin();
   
   for ( std::vector<TGLayoutHints*>::iterator itH = hints.begin(), itEnd = hints.end();
         itH != itEnd; ++itH,++itW ) 
   {
     (*itH)->SetPadLeft(maxWidth - *itW);
   }

   TGLabel* label = new TGLabel(vf,"Viewable Collections");
   vf->AddFrame(label,new TGLayoutHints(kLHintsNormal,0,0,10));

   m_tableManager= new DataAdderTableManager(&m_useableData);
   m_tableManager->indexSelected_.connect(boost::bind(&FWGUIEventDataAdder::newIndexSelected,this,_1));

   m_tableWidget = new FWTableWidget(m_tableManager,vf);
   m_tableWidget->Resize(200,200);

   // tpm: my modifications
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);

   vf->AddFrame(m_tableWidget, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   m_tableWidget->Connect("rowClicked(Int_t,Int_t,Int_t,Int_t,Int_t)","FWGUIEventDataAdder",this,"rowClicked(Int_t,Int_t,Int_t,Int_t,Int_t)");

   hints[index] = addToFrame(vf, "", m_name, widths[index]);
   
   if ( widths[index] > maxWidth ) 
   {
     maxWidth = widths[index];
   }

   ++index;

   m_doNotUseProcessName= new TGCheckButton(vf,"Do not use Process Name and instead only get this data from the most recent Process",1);
   m_doNotUseProcessName->SetState(kButtonDown);
   vf->AddFrame(m_doNotUseProcessName);

   TGHorizontalFrame* dataButtonFrame = new TGHorizontalFrame(vf);
   vf->AddFrame(dataButtonFrame,new TGLayoutHints(kLHintsBottom|kLHintsCenterX));
  
   TGTextButton* cancelButton = new TGTextButton(dataButtonFrame,"Cancel");
   cancelButton->Connect("Clicked()","FWGUIEventDataAdder",this, "windowIsClosing()");
   dataButtonFrame->AddFrame(cancelButton, new TGLayoutHints(kLHintsLeft,0,5,5,0));
   
   m_apply = new TGTextButton(dataButtonFrame,"Add Data");
   dataButtonFrame->AddFrame(m_apply, new TGLayoutHints(kLHintsRight,5,0,5,0));
   m_apply->Connect("Clicked()","FWGUIEventDataAdder",this,"addNewItem()");
   m_apply->SetEnabled(false);
   
   // Set a name to the main frame
   m_frame->SetWindowName("Add Collection");

   // Map all subwindows of main frame
   m_frame->MapSubwindows();

   // Initialize the layout algorithm
   m_frame->Layout();
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
FWGUIEventDataAdder::searchData()
{
  boost::regex re;
 
  try
  {  
    re.assign(m_search->GetText(), boost::regex_constants::icase);  
  }

  catch (boost::regex_error& e)
  {
    fwLog(fwlog::kInfo)<< m_search->GetText() <<" is not a valid regular expression."<<std::endl;
    fwLog(fwlog::kInfo)<< e.what() <<std::endl;
    return;
  }
  
  for ( std::vector<Data>::const_iterator di = m_useableData.begin(),
                                       diEnd = m_useableData.end();
        di != diEnd; ++di )
  {
    if ( boost::regex_search((*di).purpose_, re) ||       
         boost::regex_search((*di).moduleLabel_, re) ||
         boost::regex_search((*di).productInstanceLabel_, re) ||
         boost::regex_search((*di).type_, re) ||
         boost::regex_search((*di).processName_, re) )
    {
      std::cout<< (*di).purpose_ <<" "
               << (*di).type_ <<" "
               << (*di).moduleLabel_ <<" "
               << (*di).productInstanceLabel_ <<" "
               << (*di).processName_ <<std::endl;
    }
  }

  //m_tableManager->reset();
  //m_tableManager->sort(0, true);
}

/** This method inspects the opened TFile @a iFile and for each branch containing 
    products for which we can either build a TCollectionProxy or for which
    we have a specialized accessor, it registers it as a viewable item.
 */
void
FWGUIEventDataAdder::fillData(const TFile* iFile)
{
   m_useableData.clear();
   
   if (!m_presentEvent)
      return;

   const std::vector<std::string>& history = m_presentEvent->getProcessHistory();

   // Turns out, in the online system we do sometimes gets files without any  
   // history, this really should be investigated
   if (0 == history.size())
      fwLog(fwlog::kWarning) << "WARNING: the file '"
         << iFile->GetName() << "' contains no processing history"
            " and therefore should have no accessible data";
   
   std::copy(history.rbegin(),history.rend(),
             std::back_inserter(m_processNamesInFile));
   
   static const std::string s_blank;
   const std::vector<edm::BranchDescription>& descriptions =
      m_presentEvent->getBranchDescriptions();
   Data d;
   
   //I'm not going to modify TFile but I need to see what it is holding
   TTree* eventsTree = dynamic_cast<TTree*>(const_cast<TFile*>(iFile)->Get("Events"));
   assert(eventsTree);
   
   std::set<std::string> branchNamesInFile;
   TIter nextBranch(eventsTree->GetListOfBranches());
   while(TBranch* branch = static_cast<TBranch*>(nextBranch()))
      branchNamesInFile.insert(branch->GetName());
   
   typedef std::set<std::string> Purposes;
   Purposes purposes;
   std::string classType;
   
   for(size_t bi = 0, be = descriptions.size(); bi != be; ++bi) 
   {
      const edm::BranchDescription &desc = descriptions[bi];
      
      if(desc.present() &&
         branchNamesInFile.end() != branchNamesInFile.find(desc.branchName()))
      {
         const std::vector<FWRepresentationInfo>& infos 
            = m_typeAndReps->representationsForType(desc.fullClassName());
   
         /*
         //std::cout <<"try to find match "<<itBranch->fullClassName()<<std::endl;
         //For each view we need to find the non-sub-part builder whose proximity is smallest and 
         // then register only that purpose
         //NOTE: for now, we will ignore the view and only look for the closest proximity
         unsigned int minProx = ~(0U);
         for (size_t ii = 0, ei = infos.size(); ii != ei; ++ii) {
            if (!infos[ii].representsSubPart() && minProx > infos[ii].proximity()) {
               minProx = infos[ii].proximity();
            }
         }
          */
         
         //the infos list can contain multiple items with the same purpose so we will just find
         // the unique ones
         purposes.clear();
         for (size_t ii = 0, ei = infos.size(); ii != ei; ++ii) {
           /* if(!infos[ii].representsSubPart() && minProx != infos[ii].proximity()) {
               continue;
            } */
            purposes.insert(infos[ii].purpose());
         }
   
         if (purposes.empty())
            purposes.insert("Table");
         
         for (Purposes::const_iterator itPurpose = purposes.begin(),
                                      itEnd = purposes.end();
              itPurpose != itEnd;
              ++itPurpose) 
         {
            // Determine whether or not the class can be iterated
            // either by using a TVirtualCollectionProxy (of the class 
            // itself or on one of its members), or by using a 
            // FWItemAccessor plugin.
            TClass* theClass = TClass::GetClass(desc.fullClassName().c_str());
            
            if (!theClass)
               continue;

            if (!theClass->GetTypeInfo())
               continue;
            
            // This is pretty much the same thing that happens 
            if (!FWItemAccessorFactory::classAccessedAsCollection(theClass))
	         {
		         fwLog(fwlog::kDebug) << theClass->GetName() 
                          << " will not be displayed in table." << std::endl;
		         continue;
	         }
            d.type_ = desc.fullClassName();
            d.purpose_ = *itPurpose;
            d.moduleLabel_ = desc.moduleLabel();
            d.productInstanceLabel_ = desc.productInstanceName();
            d.processName_ = desc.processName();
            m_useableData.push_back(d);
	         fwLog(fwlog::kDebug) << "Add collection will display " << d.type_ 
                                 << " " << d.moduleLabel_ 
                                 << " " << d.productInstanceLabel_
                                 << " " << d.processName_ << std::endl;
         }
      }
   }
   m_tableManager->reset();
   m_tableManager->sort(0, true);
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
      
      //Check to see if this is the last process, if it is then we can let the user decide
      // to not use the process name when doing the lookup.  This makes a saved configuration
      // more robust.  However, if they choose a collection not from the last process then we need the
      // process name in order to correctly get the data they want
      bool isMostRecentProcess =true;
      int index = 0;
      for(std::vector<Data>::iterator it = m_useableData.begin(), itEnd = m_useableData.end();
          it != itEnd && isMostRecentProcess;
          ++it,++index) {
         if(index == iSelectedIndex) {continue;}
         if(it->moduleLabel_ == m_moduleLabel &&
            it->purpose_ == m_purpose &&
            it->type_ == m_type &&
            it->productInstanceLabel_ == m_productInstanceLabel) {
            //see if this process is newer than the data requested
            for(std::vector<std::string>::iterator itHist = m_processNamesInFile.begin(),itHistEnd = m_processNamesInFile.end();
                itHist != itHistEnd;
                ++itHist) {
               if (m_processName == *itHist) {
                  break;
               }
               if(it->processName_ == *itHist) {
                  isMostRecentProcess = false;
                  break;
               }
            }
         }
      }
      if(isMostRecentProcess) {
         if(!m_doNotUseProcessName->IsEnabled()) {
            m_doNotUseProcessName->SetEnabled(true);
         }
      } else {
         //NOTE: must remember state before we get here because 'enable' and 'on' are mutually
         // exlcusive :(
         m_doNotUseProcessName->SetEnabled(false);
      }
   }
}

void 
FWGUIEventDataAdder::rowClicked(Int_t iRow,Int_t iButton,Int_t iKeyMod,Int_t,Int_t)
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
