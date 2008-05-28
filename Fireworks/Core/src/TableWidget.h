// -*- C++ -*-
#ifndef Fireworks_Core_TableWidget_h
#define Fireworks_Core_TableWidget_h

// TableWidget, a la Excel.
// Copyright 2008 Cornell University, Ithaca, NY 14853. All rights reserved.
// Author: Valentin Kuznetsov 07/05/2008 (dd/mm/yyyy)
 
#include <vector>
#include <string>
#include <iostream>
#include <sstream>

#include <TROOT.h>
#include <TApplication.h>
#include <TVirtualX.h>
#include <TVirtualPadEditor.h>
#include <TGClient.h>
#include <TGFrame.h>
#include <TGIcon.h>
#include <TGLabel.h>
#include <TGButton.h>
#include <TGTextEntry.h>
#include <TGNumberEntry.h>
#include <TGSlider.h>
#include <TGScrollBar.h>

#include <TGTableLayout.h>
#include <RQ_OBJECT.h> 
 
//enum TableWidgetSortOrder {
//      m_asc = 0,
//      m_desc= 1
//};

// Abstract class interface which is used by TableWidget to access data for a table.
class TableManager {
   public:
     TableManager(void) {} 
     virtual ~TableManager() {}
//      virtual const std::string GetTitle(int col) = 0; 
      virtual int NumberOfRows() const = 0;
      virtual int NumberOfCols() const = 0;
      virtual void Sort(int col, bool sortOrder) = 0; // sortOrder=true means desc order
      virtual std::vector<std::string> GetTitles(int col) = 0;
      virtual void FillCells(int rowStart, int colStart, 
                             int rowEnd, int colEnd, std::vector<std::string>& oToFill) = 0;
      virtual TGFrame* GetRowCell(int row, TGFrame *parentFrame) = 0;
      virtual void UpdateRowCell(int row, TGFrame *rowCell) = 0;
     virtual const std::string title() const { return "Table"; }
};

class TableWidget { 
   RQ_OBJECT("TableWidget") 

   public: 
      TableWidget(TGCompositeFrame *p, TableManager* tm);
      virtual ~TableWidget(); 
      void Init(int tabRows=5,     // number of shown rows
                int tabCols=19,     // number of shown cols
                int iRow=0,        // first shown row
                int iCol=0,        // first shown column
                int cellWidth=100, 
                int cellHeight=25, 
                int scrollWidth=18,
                int scrollHeight=10);
      void InitTableCells();
      void UpdateTableTitle(int iCol);
      void UpdateTableRows(int iRow);
      void UpdateTableCells(int iRow, int iCol);
      void HighlightRow(Int_t rowId, Pixel_t hColor=0);
//      void SelectRow(Int_t rowId, Pixel_t hColor, Mask_t mask=NULL);
      void SelectRow(Int_t rowId, Mask_t mask=NULL, Pixel_t hColor=NULL);

      void OnTitleClick(Event_t *event);
      void OnRowClick(Event_t *event);
      void OnCellClick(Event_t *event);
      void OnCellDoubleClick();
//      void HandleDoubleClick(Event_t *event);
      void OnHScroll(Int_t range);
      void OnVScroll(Int_t range);

      // New APIs
//      SetHighlight();
//      SetColumns(const std::vector<std::string>& oColVector);
//      UpdateData();

   private:
//      TableWidget(const TableWidget& rhs); // stop default
//      const TableWidget& operator=( const TableWidget& ); // stop default

      TGCompositeFrame  *m_mainFrame; 
      TableManager      *m_tm;

      TGCompositeFrame  *m_tFrame, *m_hFrame;
      TGLayoutHints     *m_FrameHints, *m_tFrameHints, *m_hFrameHints, *m_vFrameHints;
      TGTableLayout     *m_tLayout;
      TGHScrollBar      *m_hSlider;
      TGVScrollBar      *m_vSlider;

      // vectors which hold entries of TGCompositeFrame for individual table components
      TGTextEntry		*m_tNameEntry;
      std::vector<TGTextEntry*> m_tTitleEntryVector;
      std::vector<TGFrame*>     m_tRowEntryVector;
      std::vector<TGTextEntry*> m_tCellEntryVector;

      // TGLayoutHints for table components
      TGTableLayoutHints	       *m_tNameHints;
      std::vector<TGTableLayoutHints*> m_tTitleHintsVector;
      std::vector<TGTableLayoutHints*> m_tRowHintsVector;
      std::vector<TGTableLayoutHints*> m_tCellHintsVector;

      // Holders of table cell's internal ROOT IDs.
      std::vector<UInt_t> m_tTitleIDVector;
      std::vector<UInt_t> m_tRowIDVector;
      std::vector<UInt_t> m_tCellIDVector;

      bool m_order;        // define orderering of table columns, only up or down
      int  m_iRow, m_iCol; // init values of shown row/col's.
      int  m_cellWidth;    // table cell width
      int  m_cellHeight;   // table cell height
      int  m_tabWidth;     // table width
      int  m_tabHeight;    // table height
      int  m_scrollWidth;  // width of the table scroll
      int  m_scrollHeight; // height of the table scroll
      int  m_tabRows;      // number of rows in a table
      int  m_tabCols;      // number of columns in a table
      int  m_minRowSelected;
      int  m_maxRowSelected;

      // colors
      Pixel_t m_titleColor,m_highlightTitleColor,m_highlightColor,m_cellColor,m_selectColor,m_textBlackColor,m_textWhiteColor;
};             

#endif
