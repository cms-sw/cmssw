#include "TableWidget.h"

// TableWidget, a la Excel.
// Copyright 2008 Cornell University, Ithaca, NY 14853. All rights reserved.
// Author: Valentin Kuznetsov 07/05/2008 (dd/mm/yyyy)

// Destructor
TableWidget::~TableWidget() {
   std::vector<TGTableLayoutHints*>::iterator i;
   std::vector<TGTextEntry*>::iterator j;
   std::vector<TGFrame*>::iterator k;
   for(i=m_tCellHintsVector.begin(); i!=m_tCellHintsVector.end(); ++i) {
       // dereference the iterator to get a pointer then delete the object the pointer points at
       delete *i;
   }
   for(j=m_tCellEntryVector.begin(); j!=m_tCellEntryVector.end(); ++j) {
       delete *j;
   }

   for(i=m_tRowHintsVector.begin(); i!=m_tRowHintsVector.end(); ++i) {
       delete *i;
   }

// I'm not sure should I handle RowEntryVector, since it's entried will be created outside
// of TableWidget/TableManager
//   for(k=m_tRowEntryVector.begin(); k!=m_tRowEntryVector.end(); ++k) {
//       delete *k;
//   }

   for(i=m_tTitleHintsVector.begin(); i!=m_tTitleHintsVector.end(); ++i) {
       delete *i;
   }
   for(j=m_tTitleEntryVector.begin(); j!=m_tTitleEntryVector.end(); ++j) {
       delete *j;
   }

   delete m_tFrameHints;
   delete m_tLayout;
   delete m_tFrame;
   delete m_hSlider;
   delete m_vSlider;
   delete m_vFrameHints;
   delete m_hFrameHints;
   delete m_hFrame;
   m_mainFrame->DeleteWindow();  // delete fMain
}
// Constructor
TableWidget::TableWidget(TGCompositeFrame *frame,TableManager* tableMgr) 
   : m_mainFrame(frame), m_tm(tableMgr)
{
   Init();
}
void
TableWidget::Init(int tabRows,
                     int tabCols,
                     int iRow,
                     int iCol,
                     int cellWidth,
                     int cellHeight,
                     int scrollWidth,
                     int scrollHeight)
{
   // Init table attributes
   m_order=true;
   m_iRow=iRow;
   m_iCol=iCol;
   m_cellWidth=cellWidth;
   m_cellHeight=cellHeight;
   m_scrollWidth=scrollWidth;
   m_scrollHeight=scrollHeight;
   m_tabRows=tabRows;
   m_tabCols=tabCols;
   m_minRowSelected=-1;
   m_maxRowSelected=-1;

   std::cout<<"Created table widget with nRows="<<m_tabRows<<" nCols="<<m_tabCols<<std::endl;
   // we used fixed size for cell width/height. So total table size should be calculated on a fly.
   m_tabWidth=m_cellWidth*(m_tabCols+1); // 1 for additional row id column
   m_tabHeight=m_cellHeight*(m_tabRows+1); // 1 for additional title row
   gClient->GetColorByName("white", m_cellColor);
   gClient->GetColorByName("#dce6ba", m_highlightColor);
   gClient->GetColorByName("#b6bddb", m_highlightTitleColor);
   gClient->GetColorByName("#d8e6e6", m_titleColor);
   gClient->GetColorByName("white", m_textWhiteColor);
   gClient->GetColorByName("black", m_textBlackColor);
   gClient->GetColorByName("#7185e6", m_selectColor);

   // Create composite frame which will hold table and vertical scroll, aligned horizontally
   m_hFrame  = new TGCompositeFrame(m_mainFrame, m_tabWidth+m_scrollWidth, m_tabHeight+m_scrollHeight, kHorizontalFrame);
   m_FrameHints = new TGLayoutHints(kLHintsTop|kLHintsLeft|kLHintsExpandX|kLHintsExpandY);
   m_mainFrame->AddFrame(m_hFrame,m_FrameHints);

   // Create frame to hold table
   m_tFrame = new TGCompositeFrame(m_hFrame, m_cellWidth*m_tabCols, m_cellHeight*m_tabRows);
   // We add 1 row for titles and 1 columns for row's id.
   m_tLayout = new TGTableLayout(m_tFrame,m_tabRows+1,m_tabCols+1,kTRUE,0,0);
   m_tFrame->SetLayoutManager(m_tLayout);
   m_tFrameHints = new TGLayoutHints(kLHintsTop|kLHintsLeft|kLHintsExpandX|kLHintsExpandY);
   m_hFrame->AddFrame(m_tFrame,m_tFrameHints);
   InitTableCells();
   UpdateTableTitle(0);
   UpdateTableRows(0);
   UpdateTableCells(0,0);

   // vertical slider
   m_vSlider = new TGVScrollBar(m_hFrame,2,m_tabHeight-m_cellHeight); // 2 is default width
   m_vSlider->SetRange(m_scrollWidth*m_tm->NumberOfRows(),m_scrollWidth);
   m_vSlider->SetSmallIncrement(m_scrollWidth);
   m_vSlider->Connect("PositionChanged(Int_t)","TableWidget",this,"OnVScroll(Int_t)");
   m_vFrameHints = new TGLayoutHints(kLHintsTop|kLHintsRight|kLHintsExpandY,0,0,m_cellHeight,0);
   m_hFrame->AddFrame(m_vSlider,m_vFrameHints);

   // horizontal slider can be attached to main composite frame
   m_hSlider = new TGHScrollBar(m_mainFrame,m_tabWidth-m_cellWidth);
   m_hSlider->SetRange(m_scrollWidth*m_tm->NumberOfCols(),m_scrollWidth);
   m_hSlider->SetSmallIncrement(m_scrollWidth);
   m_hSlider->Connect("PositionChanged(Int_t)","TableWidget",this,"OnHScroll(Int_t)");
   m_hFrameHints = new TGLayoutHints(kLHintsBottom|kLHintsRight|kLHintsExpandX,m_cellWidth,m_scrollWidth);
   m_mainFrame->AddFrame(m_hSlider,m_hFrameHints);

} 
void
TableWidget::UpdateTableTitle(int iCol)
{
   // remember which col is requested to be shown first, used in sorting.
   m_iCol=iCol;
   std::vector<std::string> titleVector=m_tm->GetTitles(iCol);
   std::vector<std::string>::iterator iter=titleVector.begin();
   for(int col = 0; col < m_tabCols; ++col,++iter) {
       int i=col+1; // first cell is always empty
       m_tTitleEntryVector[i]->Clear();
       m_tTitleEntryVector[i]->SetBackgroundColor(m_titleColor);
       if (iter < titleVector.end())
	    m_tTitleEntryVector[i]->SetText((*iter).c_str());
   }
}
void
TableWidget::UpdateTableRows(int iRow)
{
   // remember which row is requested to be shown first, used in sorting.
   m_iRow=iRow;
   for(int row = 0; row < m_tabRows; ++row) {
//       m_tRowEntryVector[row]->Clear();
//       m_tRowEntryVector[row]->SetText(m_tm->PrintRowNumber(iRow+row).c_str());
       m_tm->UpdateRowCell(iRow+row,m_tRowEntryVector[row]);
   }
}
void
TableWidget::UpdateTableCells(int iRow, int iCol)
{
   std::vector<std::string> tableVector(m_tabRows*m_tabCols);
   std::vector<std::string>::iterator iter;
   m_tm->FillCells(iRow,iCol,iRow+m_tabRows,iCol+m_tabCols,tableVector);
   int i=0;
   for(iter=tableVector.begin();iter!=tableVector.end();++iter, ++i) {
       m_tCellEntryVector[i]->Clear();
       m_tCellEntryVector[i]->SetBackgroundColor(m_cellColor);
       m_tCellEntryVector[i]->SetText((*iter).c_str());
   }
   // clear the rest of the table
   for (int row = 0; row < m_tabRows; ++row)
	for (int col = 0; col < m_tabCols; ++col) {
	     if (row * m_tabCols + col < i)
		  continue;
	     m_tCellEntryVector[row * m_tabCols + col]->Clear();
	     m_tCellEntryVector[row * m_tabCols + col]->SetBackgroundColor(
		  m_cellColor);
	}
}
void
TableWidget::InitTableCells()
{ 
   // Add rows/cols
   Int_t id=0;
   TGTextEntry *cell;
   TGFrame *rowCell;
   for(int row = 0; row < m_tabRows+1; ++row) {
       for(int col = 0; col < m_tabCols+1; ++col) {

           // First row is a title row
           if (!row) {
               if(col) {
                  cell = new TGTextEntry("title",m_tFrame);
               } else {
                  cell = new TGTextEntry("",m_tFrame);
               }
	       cell->ChangeOptions(0);
               cell->Resize(m_cellWidth,m_cellHeight);
               cell->SetBackgroundColor(m_titleColor);
               cell->SetAlignment(kTextRight);
               cell->Connect("ProcessedEvent(Event_t*)","TableWidget",this,"OnTitleClick(Event_t*)");
               TGTableLayoutHints* tloh = 
                   new TGTableLayoutHints(col,col+1,row,row+1, 
                                          kLHintsCenterX|kLHintsCenterY|
                                          kLHintsExpandX|kLHintsExpandY|
                                          kLHintsShrinkX|kLHintsShrinkY|
                                          kLHintsFillX|kLHintsFillY);
               m_tFrame->AddFrame(cell,tloh);
               m_tTitleEntryVector.push_back(cell);
               m_tTitleHintsVector.push_back(tloh);
               m_tTitleIDVector.push_back(cell->GetId());
           }
           // First col is a row number
           if (!col && row) {
//               cell = new TGTextEntry(m_tm->PrintRowNumber(row),m_tFrame);
               rowCell = m_tm->GetRowCell(row,m_tFrame);
	       rowCell->ChangeOptions(0);
               rowCell->Resize(m_cellWidth,m_cellHeight);
               rowCell->Connect("ProcessedEvent(Event_t*)","TableWidget",this,"OnRowClick(Event_t*)");
               TGTableLayoutHints* tloh = 
                   new TGTableLayoutHints(col,col+1,row,row+1, 
                                          kLHintsCenterX|kLHintsCenterY|
                                          kLHintsExpandX|kLHintsExpandY|
                                          kLHintsShrinkX|kLHintsShrinkY|
                                          kLHintsFillX|kLHintsFillY);
               m_tFrame->AddFrame(rowCell,tloh);
               m_tRowEntryVector.push_back(rowCell);
               m_tRowHintsVector.push_back(tloh);
               m_tRowIDVector.push_back(rowCell->GetId());
           }
           if (row && col) {
               cell = new TGTextEntry("",m_tFrame);
	       cell->ChangeOptions(0);
               cell->SetToolTipText("This is a table cell");
               cell->Resize(m_cellWidth,m_cellHeight);
               cell->SetAlignment(kTextRight);
               cell->Connect("ProcessedEvent(Event_t*)","TableWidget",this,"OnCellClick(Event_t*)");
//               cell->Connect("DoubleClicked()","TableWidget",this,"OnCellDoubleClick()");
//               cell->Connect("HandleDoubleClick(Event_t*)","TableWidget",this,"HandleDoubleClick(Event*)");
               TGTableLayoutHints* tloh = 
                   new TGTableLayoutHints(col,col+1,row,row+1, 
                                          kLHintsCenterX|kLHintsCenterY|
                                          kLHintsExpandX|kLHintsExpandY|
                                          kLHintsShrinkX|kLHintsShrinkY|
                                          kLHintsFillX|kLHintsFillY);
               m_tFrame->AddFrame(cell,tloh);
               m_tCellEntryVector.push_back(cell);
               m_tCellHintsVector.push_back(tloh);
               m_tCellIDVector.push_back(cell->GetId());
           }
           ++id;
       }
   }        
   m_tFrame->Layout();
} 
void
TableWidget::OnHScroll(Int_t range)
{
   // NOTE: the horizontal scroll needs to update UpdateTableTitle and UpdateTableCells
   int startCol=range/m_scrollWidth;
   UpdateTableTitle(startCol);
   UpdateTableCells(m_iRow,startCol);
}

void
TableWidget::OnVScroll(Int_t range)
{
   // NOTE: the vertical scroll needs to update UpdatteTableRows and UpdateTableCells
   int startRow=range/m_scrollWidth;
   UpdateTableRows(startRow);
   UpdateTableCells(startRow,m_iCol);
}

void
TableWidget::OnTitleClick(Event_t *event)
{
   // Handle title click.
   if (event->fType != kButtonPress )
      return;
   UInt_t titleID = event->fWindow;
//   std::cout<<"m_iRow="<<m_iRow<<" m_iCol="<<m_iCol<<std::endl;
   // Find index position of title ID in title vector
   std::vector<std::string> titleVector=m_tm->GetTitles(m_iCol);
   std::vector<UInt_t>::const_iterator iter;
   int id=0, titleIdx=-1;
   for(iter=m_tTitleIDVector.begin();iter!=m_tTitleIDVector.end();++iter) {
       // NOTE: remember first cell in title row is empty, and tableManager holds only titles
       // so I'll skip the case when id=0 and use id-1 to retrieve titles from t_tm.
       if((*iter)==titleID && id) {
          titleIdx=m_iCol+id-1; // 1 counts for first empty cell and m_iCol used to know where to start
          std::string s=m_tTitleEntryVector[id]->GetText();
          std::string::size_type loc = s.find( "(desc)", 0 );
          s=titleVector[id-1];
          if(loc!=std::string::npos) {
             s+=" (asc)";
          } else {
             s+=" (desc)";
          }
          m_tTitleEntryVector[id]->Clear();
          m_tTitleEntryVector[id]->SetText(s.c_str());
          m_tTitleEntryVector[id]->SetBackgroundColor(m_highlightTitleColor);
       } else {
	    if (id > 0 && (unsigned int)id < titleVector.size()) {
               std::string s=titleVector[id-1];
               m_tTitleEntryVector[id]->Clear();
               m_tTitleEntryVector[id]->SetText(s.c_str());
               m_tTitleEntryVector[id]->SetBackgroundColor(m_titleColor);
           }
       }
       ++id;
   }
   // Now reorder table cells and remember ordering
   m_tm->Sort(titleIdx,m_order);
   this->UpdateTableCells(m_iRow,m_iCol);
   m_order=!m_order;
   // Update row-number column
   int rowIdx=0, offset=0;
   if (m_order) {
       for(int row = m_tm->NumberOfRows()-1; row >=m_tm->NumberOfRows()-m_tabRows; --row) {
           m_tm->UpdateRowCell(row,m_tRowEntryVector[rowIdx]);
           ++rowIdx;
       }
   } else {
       for(int row = 0; row < m_tabRows; ++row) {
           if(!m_iRow && !row) offset=1;
           m_tm->UpdateRowCell(m_iRow+row+offset,m_tRowEntryVector[rowIdx]);
           ++rowIdx;
       }
   }
   
}
void TableWidget::OnRowClick(Event_t *event)
{
   // Handle row click.
   if (event->fType != kButtonPress ) {
      return;
   }
}
//void TableWidget::OnRowsSelect(Event_t *event) {
//    Handle row selection
//   if (event->fType != kButtonPress ) {
//      return;
//   }
//}
void TableWidget::OnCellClick(Event_t *event)
{
   // Handle title click.
//   if (event->fType != kButtonPress ) {
//      return;
//   }
   UInt_t cellID = event->fWindow;
//   std::cout<<"OnCellClick, cellId="<<cellID<<" event type="<<event->fType<<" state="<<event->fState<<std::endl;
   
   // Find row to highlight
   int id=0, rowId=-1;
   for(int row = 0; row < m_tabRows; ++row) {
       for(int col = 0; col < m_tabCols; ++col) {
//           std::cout << "("
//                     <<row
//                     <<","
//                     <<col
//                     <<")-cell, id="
//                     <<m_tCellIDVector[id]
//                     <<std::endl;
           if(m_tCellIDVector[id]==cellID) {
              rowId=row;
              break;
           }
           ++id;
       }
       if(rowId!=-1) break;
   }

   if (event->fType==kButtonPress) {
       SelectRow(rowId,event->fState,m_selectColor);
   } else if(event->fType==kEnterNotify) {
       HighlightRow(rowId);
   } else if(event->fType==kLeaveNotify) {
       HighlightRow(rowId,m_cellColor);
   }
}
void TableWidget::OnCellDoubleClick()
{
   std::cout<<"OnCellDoubleClick"<<std::endl;
}
//void TableWidget::HandleDoubleClick(Event_t *event)
//{
//   std::cout<<"HandleDoubleClick"<<std::endl;
//}
void TableWidget::HighlightRow(Int_t rowId, Pixel_t hColor) 
{
   int id=0;
   for(int row = 0; row < m_tabRows; ++row) {
       for(int col = 0; col < m_tabCols; ++col) {
          Pixel_t bColor = m_tCellEntryVector[id]->GetBackground();
          if (row==rowId) {
              if (hColor) {
                 m_tCellEntryVector[id]->SetBackgroundColor(hColor);
              } else {
                 m_tCellEntryVector[id]->SetBackgroundColor(m_highlightColor);
              }
              m_tCellEntryVector[id]->SetForegroundColor(m_textBlackColor);
              if (bColor==m_selectColor) {
                  m_tCellEntryVector[id]->SetBackgroundColor(m_selectColor);
                  m_tCellEntryVector[id]->SetForegroundColor(m_textWhiteColor);
              }
          } else {
              if (bColor==m_selectColor) {
                  m_tCellEntryVector[id]->SetBackgroundColor(m_selectColor);
                  m_tCellEntryVector[id]->SetForegroundColor(m_textWhiteColor);
              } else {
                  m_tCellEntryVector[id]->SetBackgroundColor(m_cellColor);
                  m_tCellEntryVector[id]->SetForegroundColor(m_textBlackColor);
              }
          }
          ++id;
       }
   }
}
void TableWidget::SelectRow(Int_t rowId, Mask_t mask, Pixel_t sColor) 
{
   // Emit SIGNAL
   ULong_t args[3];
   args[0] = (ULong_t)rowId;
   args[1] = (ULong_t)mask;
   args[2] = (ULong_t)sColor;
   Emit("SelectRow(Int_t,Mask_t,Pixel_t)",args);
//   std::cout<<"select row="<<rowId<<" mask="<<mask<<" color="<<sColor<<std::endl;
   // Be OS independent
   Mask_t altKey=kKeyMod2Mask;
   // Handle request
   if(!sColor) sColor=m_selectColor;
   Pixel_t cellColor;
   int id=0, selMinRow=-1, selMaxRow=-1;
   for(int row = 0; row < m_tabRows; ++row) {
       for(int col = 0; col < m_tabCols; ++col) {
           if (m_tCellEntryVector[id]->GetBackground()==m_selectColor) {
              if (selMinRow==-1) selMinRow=row;
              if (selMinRow>row) selMinRow=row;
              if (selMaxRow==-1) selMaxRow=row;
              if (selMaxRow<row) selMaxRow=row;
           }
           cellColor=m_tCellEntryVector[id]->GetBackground();
           if (row==rowId) {
               if (mask!=altKey) {
                   m_tCellEntryVector[id]->SetBackgroundColor(sColor);
                   m_tCellEntryVector[id]->SetForegroundColor(m_textWhiteColor);
               } else {
                   // Toggle cell
                   if (cellColor==m_selectColor) {
                       m_tCellEntryVector[id]->SetBackgroundColor(m_cellColor);
                       m_tCellEntryVector[id]->SetForegroundColor(m_textBlackColor);
                   } else {
                       m_tCellEntryVector[id]->SetBackgroundColor(sColor);
                       m_tCellEntryVector[id]->SetForegroundColor(m_textWhiteColor);
                   }
               }
           } else if (mask!=altKey) {
               m_tCellEntryVector[id]->SetBackgroundColor(m_cellColor);
               m_tCellEntryVector[id]->SetForegroundColor(m_textBlackColor);
           }
           ++id;
       }
   }
   m_minRowSelected=selMinRow;
   m_maxRowSelected=selMaxRow;
   // select all rows between selRow and newly selected one.
   int minRow=-1, maxRow=-1;
   if (selMinRow>-1 && mask==kKeyShiftMask) {
      minRow = selMinRow>rowId ? rowId : selMinRow ;
      maxRow = selMaxRow<rowId ? rowId : selMaxRow ;
      m_minRowSelected=minRow;
      m_maxRowSelected=maxRow;
//      std::cout<<"minRow="<<minRow<<" maxRow="<<maxRow<<std::endl;
      id=0;
      for(int row = 0; row < m_tabRows; ++row) {
          for(int col = 0; col < m_tabCols; ++col) {
              if (row>=minRow && row<=maxRow) {
                  m_tCellEntryVector[id]->SetBackgroundColor(sColor);
                  m_tCellEntryVector[id]->SetForegroundColor(m_textWhiteColor);
              } else {
                  m_tCellEntryVector[id]->SetBackgroundColor(m_cellColor);
                  m_tCellEntryVector[id]->SetForegroundColor(m_textBlackColor);
              }
              ++id;
          }
      }
   }
}

