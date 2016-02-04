#include <sstream>
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/interface/FWCheckedTextTableCellRenderer.h"

class FWCheckedTextTableCellRenderer;

class FWTestTableManager : public FWTableManagerBase {
   public:
      FWTestTableManager(const std::vector<std::string>& iColumns,
                         const std::vector<std::string>& iData /*must be a multiple of columns */);
      virtual  int numberOfRows() const ;
      virtual  int numberOfColumns() const ;
      virtual int unsortedRowNumber(int iSortedRowNumber) const;
      virtual void implSort(int col, bool sortOrder) ; // sortOrder=true means desc order
      virtual std::vector<std::string> getTitles() const;
      virtual FWTableCellRendererBase* cellRenderer(int iRow, int iCol) const;
      
      void selectRow(int);
      //responds to a row being selected 
      void rowClicked(Int_t row, Int_t btn, Int_t keyMod);
      
      //responds to a checkbox
      void checkBoxClicked();
      
      virtual bool hasRowHeaders() const ;
      virtual FWTableCellRendererBase* rowHeader(int iRow) const ;
      
   private:
      std::vector<std::string> m_columns;
      std::vector<std::string> m_content;
      std::vector<unsigned int> m_sortOrder;
      std::vector<bool> m_rowChecked;
      
      FWTextTableCellRenderer* m_renderer;
      FWCheckedTextTableCellRenderer* m_rowHeaderRenderer;
      
      int m_selectedRow;
      mutable int m_lastRequestedRow;
};




FWTestTableManager::FWTestTableManager(const std::vector<std::string>& iColumns,
                                       const std::vector<std::string>& iData):
m_columns(iColumns),
m_content(iData),
m_rowChecked(m_content.size(),false),
m_renderer( new FWTextTableCellRenderer),
m_rowHeaderRenderer( new FWCheckedTextTableCellRenderer),
m_selectedRow(-1),
m_lastRequestedRow(-1)
{
   assert(0== m_content.size()%m_columns.size());
   unsigned int nRows = numberOfRows();
   m_sortOrder.reserve(nRows);
   for(unsigned int i = 0; i < nRows;++i){
      m_sortOrder.push_back(i); 
   }
   m_rowHeaderRenderer->Connect("checkBoxClicked()","FWTestTableManager",this,"checkBoxClicked()");
}

int 
FWTestTableManager::numberOfRows() const
{
   return m_content.size()/m_columns.size();
}

int 
FWTestTableManager::numberOfColumns() const
{
   return m_columns.size();
}

void 
FWTestTableManager::implSort(int col, bool sortOrder) 
{
   if(col >-1 && col < m_columns.size()) {
      std::vector<std::pair<std::string,unsigned int> > toSort;
      int nRows = numberOfRows();
      toSort.reserve(nRows);
      
      for(int row=0; row< nRows; ++row) {
         toSort.push_back( std::make_pair<std::string, unsigned int>(m_content[col+row*numberOfColumns()],row));
      }
      if(sortOrder){
         std::sort(toSort.begin(),toSort.end(),std::greater<std::pair<std::string,unsigned int> >());
      } else {
         std::sort(toSort.begin(),toSort.end(),std::less<std::pair<std::string,unsigned int> >());
      }
      std::vector<unsigned int>::iterator itSort = m_sortOrder.begin();
      for(std::vector<std::pair<std::string,unsigned int> >::iterator it = toSort.begin(), itEnd=toSort.end();
      it != itEnd;
      ++it,++itSort) {
         *itSort = it->second;
      }
   }
}

int 
FWTestTableManager::unsortedRowNumber(int iSortedRowNumber) const
{
   return m_sortOrder[iSortedRowNumber];
}

std::vector<std::string> 
FWTestTableManager::getTitles() const
{
   return std::vector<std::string>(m_columns.begin(),m_columns.end());
}

FWTableCellRendererBase* 
FWTestTableManager::cellRenderer(int iRow, int iCol) const
{
   int rowBeforeSort = m_sortOrder[iRow];
   m_renderer->setData( *(m_content.begin()+rowBeforeSort*m_columns.size()+iCol),
                          rowBeforeSort==m_selectedRow);
   return m_renderer;
}

void 
FWTestTableManager::selectRow(int iRow)
{
   if(iRow != m_selectedRow) {
      m_selectedRow = iRow;
      visualPropertiesChanged();
   }
}

void 
FWTestTableManager::rowClicked(Int_t row, Int_t btn, Int_t keyMod)
{
   if(btn==kButton1) {
      if(row==m_selectedRow) {
         if(keyMod & kKeyShiftMask) {
            selectRow(-1);
         }
      } else {
         selectRow(row);
      }
   }  
}

bool 
FWTestTableManager::hasRowHeaders() const 
{
   return true; 
   //return false;
}
FWTableCellRendererBase* 
FWTestTableManager::rowHeader(int iRow) const
{
   int rowBeforeSort = m_sortOrder[iRow];
   m_lastRequestedRow=rowBeforeSort;
   
   std::ostringstream s;
   s<<rowBeforeSort;
   m_rowHeaderRenderer->setData(s.str(),rowBeforeSort==m_selectedRow);
   m_rowHeaderRenderer->setChecked(m_rowChecked[rowBeforeSort]);
   return m_rowHeaderRenderer;
}

void
FWTestTableManager::checkBoxClicked()
{
   if(-1 != m_lastRequestedRow) {
      m_rowChecked[m_lastRequestedRow] = ! m_rowChecked[m_lastRequestedRow];
      visualPropertiesChanged();
   }
}


void tableWidgetTest()
{
   int width=440;
   int height=400;
   // Create a main frame 
   TGMainFrame *fMain = new TGMainFrame(gClient->GetRoot(),width,height); 

   TGCompositeFrame *tFrame  = new TGCompositeFrame(fMain, width, height);
   TGLayoutHints *tFrameHints = new TGLayoutHints(kLHintsTop|kLHintsLeft|kLHintsExpandX|kLHintsExpandY);
   fMain->AddFrame(tFrame,tFrameHints);
   
    std::vector<std::string> columns;
   columns.push_back("one long title");
   columns.push_back("two");
   columns.push_back("another long one");
   columns.push_back("four");
   std::vector<std::string> data;
   for(int iRow =1; iRow < 1000; ++iRow) {
      for(int iCol =1; iCol < 5;++iCol) {
         std::stringstream s;
         s<<iRow<<" "<<iCol;
         data.push_back(s.str());
      }
   }

   FWTestTableManager* tableM = new FWTestTableManager(columns,data);
   FWTableWidget* table = new FWTableWidget(tableM, tFrame);
   table->Connect("rowClicked(Int_t,Int_t,Int_t)","FWTestTableManager",tableM,"rowClicked(Int_t,Int_t,Int_t)");
   table->sort(0,true);
   tFrame->AddFrame(table,tFrameHints);
   
   fMain->SetWindowName("Header Test");
   fMain->MapSubwindows();
   fMain->Layout();
   fMain->MapWindow();
   
}
