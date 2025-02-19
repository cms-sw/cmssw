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
// $Id: FWGUIEventDataAdder.cc,v 1.51 2011/03/17 10:55:41 amraktad Exp $
//

// system include files
#include <iostream>
#include <sigc++/signal.h>
#include <boost/bind.hpp>
#include <algorithm>
#include <cctype>
#include <string>

#include "TGFrame.h"
#include "TGTextEntry.h"
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
#include "Fireworks/Core/interface/FWJobMetadataManager.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTextTableCellRenderer.h"
#include "Fireworks/Core/interface/fwLog.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"
//
// constants, enums and typedefs
//
static const std::string& dataForColumn( const FWJobMetadataManager::Data& iData, int iCol)
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
   DataAdderTableManager(FWJobMetadataManager *manager):
      m_manager(manager), m_selectedRow(-1), m_filter() 
   {
      reset();
   }

   virtual int numberOfRows() const {
      return m_row_to_index.size();
   }
   virtual int numberOfColumns() const {
      return kNColumns;
   }
   
   /** Updates the table using the passed @a filter.
       Notice that in this case we reset the sorting and show results by those
       best matching the filter.
     */
   virtual void sortWithFilter(const char *filter)
   {
      m_filter = filter;
      sort(-1, sortOrder());
      dataChanged();
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
         const FWJobMetadataManager::Data& data = (m_manager->usableData())[unsortedRow];

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
      m_row_to_index.reserve(m_manager->usableData().size());
      for(unsigned int i =0; i < m_manager->usableData().size(); ++i) {
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
   FWJobMetadataManager* m_manager;
   std::vector<int>      m_row_to_index;
   int                   m_selectedRow;
   std::string           m_filter;
   mutable FWTextTableCellRenderer m_renderer;
};

namespace {
void strip(std::string &source, const char *str)
   {
      std::string remove(str);
      while(true)
      {
         size_t found = source.find(remove);
         if (found == std::string::npos)
            break;
         source.erase(found, remove.size());
      }
   }

   /** Helper classes to handle sorting and filtering.
   
       The idea is that we sort things so that:
   
       - An item that matches is always less than one that does not.
       - If two items both match they are sorted according to the sorting 
         criteria.
       - If two items both do not match, the are always sorted (so that we
         do not waste time in sorting non matching items).
      
       Then we tell the table that the size of the available data is only
       the size of the matching items.
       
       Notice that the matching here does not work with regular expressions 
       but it matches in the case the filter string is found anywhere, 
       regardless of the case.
     */
   class SortAndFilter
   {
   public:
      SortAndFilter(const char *filter, int column, bool order, 
                    const std::vector<FWJobMetadataManager::Data> &data)
         : m_filter(filter),
           m_column(column),
           m_order(order),
           m_data(data)
         {
            simplify(m_filter);
            m_weights.resize(data.size());
            
            // Calculate whether or not all the entries match the given filter.
            // This is done only once, since it's invariant under permutations
            // of the data.
            for (size_t i = 0, e = m_weights.size(); i != e; ++i)
               m_weights[i] = matchesFilter(m_data[i]);
         }

      /** Makes @a str lowercase and eliminates bits we dont want to take
          into account in while searching.
        */
      static void simplify(std::string &str)
      {
         std::transform(str.begin(), str.end(), str.begin(), tolower);
         strip(str, "std::");
         strip(str, "edm::");
         strip(str, "vector<");
         strip(str, "clonepolicy");
         strip(str, "ownvector");
         strip(str, "rangemap<");
         strip(str, "strictweakordering<");
         strip(str, "sortedcollection<");
         strip(str, "reco::");
         strip(str, "edmnew::");
      }
      
      unsigned int matches(const std::string &str) const
         {
            std::string up(str);
            simplify(up);
            const char *begin = up.c_str();
            
            // If the filter is empty, we consider anything as matching 
            // (i.e. it will not loop).
            // If the filter is not empty but the string to be matched is, we 
            // consider it as if it was not matching.
            if ((!m_filter.empty()) && str.empty())
               return 0;
            
            // There are two level of matching. "Full string" and 
            // "All characters". "Full string" matches return an higher weight 
            // and therefore should appear on top.
            if (strstr(begin, m_filter.c_str()))
               return 2;

            return 0;
         }
      
      /** If any of the columns (including "Purpose"!!) matches, we consider
          the row valid.
          
          @return the best score obtained when matching strings.
        */
      unsigned int matchesFilter(const FWJobMetadataManager::Data &data) const
         {
            std::vector<unsigned int> scores;
            scores.reserve(10);
            scores.push_back(matches(data.purpose_));
            scores.push_back(matches(data.type_));
            scores.push_back(matches(data.moduleLabel_));
            scores.push_back(matches(data.productInstanceLabel_));
            scores.push_back(matches(data.processName_));
            std::sort(scores.begin(), scores.end());
            return scores.back();
         }
      
      /** Have a look at the class description to see the rationale behind 
          this.
        */
      bool operator()(const int &aIndex, const int &bIndex)
         {
            // In case no column is selected, we sort by relevance of the 
            // filter.
            if (m_column == -1)
               return m_weights[aIndex] >= m_weights[bIndex];

            const FWJobMetadataManager::Data &a = m_data[aIndex];
            const FWJobMetadataManager::Data &b = m_data[bIndex];

            if (m_order)
               return dataForColumn(a, m_column) < dataForColumn(b, m_column);
            else
               return dataForColumn(a, m_column) > dataForColumn(b, m_column);
         }
   private:
      std::string m_filter;
      int         m_column;
      bool        m_order;

      const std::vector<FWJobMetadataManager::Data> &m_data;
      std::vector<unsigned int>                      m_weights;
   };

   void doSort(int column,
               const char *filter,
               bool descentSort,
               const std::vector<FWJobMetadataManager::Data>& iData,
               std::vector<int>& oRowToIndex)
   {
      std::vector<int> ordered;
      ordered.reserve(iData.size());
      
      for (size_t i = 0, e = iData.size(); i != e; ++i)
         ordered.push_back(i);
      
      SortAndFilter sorter(filter, column, descentSort, iData);
      // GE: Using std::sort does not work for some reason... Bah...
      std::stable_sort(ordered.begin(), ordered.end(), sorter);

      oRowToIndex.clear();
      oRowToIndex.reserve(ordered.size());
      // Only keep track of the rows that match.
      for (size_t i = 0, e = ordered.size(); i != e; ++i)
         if (sorter.matchesFilter(iData[ordered[i]]) != 0)
            oRowToIndex.push_back(ordered[i]);
   }
}

void
DataAdderTableManager::implSort(int column, bool sortOrder)
{
   doSort(column, m_filter.c_str(), sortOrder, m_manager->usableData(), m_row_to_index);
}

//
// static data member definitions
//

//
// constructors and destructor
//

FWGUIEventDataAdder::FWGUIEventDataAdder(
   UInt_t iWidth,UInt_t iHeight,
   FWEventItemsManager* iManager,
   TGFrame* iParent,
   FWJobMetadataManager *iMetadataManager) 
   :
      m_manager(iManager),
      m_metadataManager(iMetadataManager),
      m_parentFrame(iParent)
{
   m_metadataManager->metadataChanged_.connect(boost::bind(&FWGUIEventDataAdder::metadataUpdatedSlot, this));
   createWindow();
}

// FWGUIEventDataAdder::FWGUIEventDataAdder(const FWGUIEventDataAdder& rhs)
// {
//    // do actual copying here;
// }

FWGUIEventDataAdder::~FWGUIEventDataAdder()
{
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
   m_manager->add(desc);
}

void
FWGUIEventDataAdder::addNewItemAndClose()
{
   addNewItem();
   windowIsClosing();
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
   m_applyAndClose->SetEnabled(false);
   
   m_frame->UnmapWindow();
   m_frame->DontCallClose();
}

void
FWGUIEventDataAdder::updateFilterString(const char *str)
{
   m_tableManager->sortWithFilter(str);
   m_tableManager->dataChanged();
}

void
FWGUIEventDataAdder::createWindow()
{
   m_tableManager = new DataAdderTableManager(m_metadataManager);
   m_tableManager->indexSelected_.connect(boost::bind(&FWGUIEventDataAdder::newIndexSelected,this,_1));

   m_frame = new TGTransientFrame(gClient->GetDefaultRoot(),m_parentFrame,600,400);
   m_frame->Connect("CloseWindow()","FWGUIEventDataAdder",this,"windowIsClosing()");

   FWDialogBuilder builder(m_frame);
   TGTextButton *cancelButton, *resetButton;
   
   builder.indent(10)
          .spaceDown(15)
          .addLabel("Search:", 0).expand(false).spaceUp(4).floatLeft(4)
          .addTextEntry("", &m_search).spaceUp(0)
          .frameSpaceDown(10)
          .addLabel("Viewable Collections", 8)
          .frameSpaceDown(5)
          .addTable(m_tableManager, &m_tableWidget).expand(true, true)
          .addLabel("Name:", 0).expand(false).spaceUp(4).floatLeft(4)
          .addTextEntry("", &m_name).spaceUp(0).floatLeft(4)
          .addTextButton(" Reset  ", &resetButton).expand(false, false)
          .frameSpaceUpDown(5)
          .addCheckbox("Do not use Process Name and "
                       "instead only get this data "
                       "from the most recent Process",
                       &m_doNotUseProcessName)
          .frameSpaceDown(15)
          .hSpacer().floatLeft(0)
          .addTextButton(" Close  ", &cancelButton).expand(false).floatLeft(4)
          .addTextButton(" Add Data  ", &m_apply).expand(false).floatLeft(4)
          .addTextButton(" Add Data && Close  ", &m_applyAndClose).expand(false)
                                .spaceRight(25).spaceDown(15);

   m_search->Connect("TextChanged(const char *)", "FWGUIEventDataAdder", 
                     this, "updateFilterString(const char *)");
   m_search->SetEnabled(true);
   m_tableWidget->SetBackgroundColor(0xffffff);
   m_tableWidget->SetLineSeparatorColor(0x000000);
   m_tableWidget->SetHeaderBackgroundColor(0xececec);
   m_tableWidget->Connect("rowClicked(Int_t,Int_t,Int_t,Int_t,Int_t)",
                          "FWGUIEventDataAdder",this,
                          "rowClicked(Int_t,Int_t,Int_t,Int_t,Int_t)");

   m_tableWidget->disableGrowInWidth();
   m_name->SetState(true);
   resetButton->SetEnabled(true);
   resetButton->Connect("Clicked()", "FWGUIEventDataAdder", this, "resetNameEntry()");
   m_doNotUseProcessName->SetState(kButtonDown);
   cancelButton->Connect("Clicked()","FWGUIEventDataAdder", this, "windowIsClosing()");
   cancelButton->SetEnabled(true);
   m_apply->Connect("Clicked()", "FWGUIEventDataAdder", this, "addNewItem()");
   m_applyAndClose->Connect("Clicked()", "FWGUIEventDataAdder", this, "addNewItemAndClose()");
   
   m_frame->SetWindowName("Add Collection");
   m_frame->MapSubwindows();
   m_frame->Layout();
}

/** Slot for the metadataChanged signal of the FWJobMetadataManager.
    Whenever metadata changes, for whatever reason (e.g. a new file), we
    need to update the table.
  */
void
FWGUIEventDataAdder::metadataUpdatedSlot(void)
{
   m_tableManager->reset();
   m_tableManager->sort(0, true);
}

void
FWGUIEventDataAdder::resetNameEntry()
{
   m_name->SetText(m_apply->IsEnabled() ? m_moduleLabel.c_str() : "", kFALSE);
}

void
FWGUIEventDataAdder::newIndexSelected(int iSelectedIndex)
{
   if(-1 != iSelectedIndex) {
      std::vector<FWJobMetadataManager::Data> &metadata = m_metadataManager->usableData();
      m_purpose = metadata[iSelectedIndex].purpose_;
      m_type = metadata[iSelectedIndex].type_;
      std::string oldModuleLabel = m_moduleLabel;
      m_moduleLabel = metadata[iSelectedIndex].moduleLabel_;
      m_productInstanceLabel = metadata[iSelectedIndex].productInstanceLabel_;
      m_processName = metadata[iSelectedIndex].processName_;
      
      if(strlen(m_name->GetText())==0 || oldModuleLabel == m_name->GetText()) {
         m_name->SetText(m_moduleLabel.c_str());
      }
      m_apply->SetEnabled(true);
      m_applyAndClose->SetEnabled(true);

      //Check to see if this is the last process, if it is then we can let the user decide
      // to not use the process name when doing the lookup.  This makes a saved configuration
      // more robust.  However, if they choose a collection not from the last process then we need the
      // process name in order to correctly get the data they want
      bool isMostRecentProcess =true;
      int index = 0;
      for(std::vector<FWJobMetadataManager::Data>::iterator it = metadata.begin(), itEnd = metadata.end();
          it != itEnd && isMostRecentProcess;
          ++it,++index) {
         if(index == iSelectedIndex) {continue;}
         if(it->moduleLabel_ == m_moduleLabel &&
            it->purpose_ == m_purpose &&
            it->type_ == m_type &&
            it->productInstanceLabel_ == m_productInstanceLabel) {
            //see if this process is newer than the data requested
            
            for(size_t pni = 0, pne = m_metadataManager->processNamesInJob().size();
                pni != pne; ++pni) 
            {
               const std::string &processName = m_metadataManager->processNamesInJob()[pni];
               if (m_processName == processName)
                  break;

               if(it->processName_ == processName) 
               {
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
