// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWTableWidget
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:45:42 EST 2009
// $Id: FWTableWidget.cc,v 1.1 2009/02/03 20:33:04 chrjones Exp $
//

// system include files
#include "TGScrollBar.h"
#include "TGTableLayout.h"

// user include files
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/src/FWTabularWidget.h"
#include "Fireworks/TableWidget/src/FWAdapterHeaderTableManager.h"
#include "Fireworks/TableWidget/src/FWAdapterRowHeaderTableManager.h"


//
// constants, enums and typedefs
//
static const UInt_t kRowOptions = kLHintsExpandX|kLHintsFillX|kLHintsShrinkX;
static const UInt_t kColOptions = kLHintsExpandY|kLHintsFillY|kLHintsShrinkY;

//
// static data member definitions
//

//
// constructors and destructor
//
FWTableWidget::FWTableWidget(FWTableManagerBase* iManager,const TGWindow* p):
TGCompositeFrame(p),
   m_bodyTable(iManager),
   m_headerTable(new FWAdapterHeaderTableManager(iManager)),
   m_rowHeaderTable(iManager->hasRowHeaders()?new FWAdapterRowHeaderTableManager(iManager): static_cast<FWTableManagerBase*>(0)),
   m_showingVSlider(true),
   m_showingHSlider(true),
   m_sortedColumn(-1),
   m_descendingSort(true)
{
   SetLayoutManager( new TGTableLayout(this,3,3) );
   
   m_header = new FWTabularWidget(m_headerTable,this);
   AddFrame(m_header, new TGTableLayoutHints(1,2,0,1,kLHintsTop|kLHintsLeft|kRowOptions));	
   m_header->Connect("buttonReleased(Int_t,Int_t,Int_t,Int_t)","FWTableWidget",this,"buttonReleasedInHeader(Int_t,Int_t,Int_t,Int_t)");
	
   m_body = new FWTabularWidget(iManager,this);
   AddFrame(m_body, new TGTableLayoutHints(1,2,1,2,kLHintsTop|kLHintsLeft|kRowOptions|kColOptions));
   m_body->Connect("buttonReleased(Int_t,Int_t,Int_t,Int_t)","FWTableWidget",this,"buttonReleasedInBody(Int_t,Int_t,Int_t,Int_t)");

   //set sizes
   std::vector<unsigned int> columnWidths = m_header->widthOfTextInColumns();
   std::vector<unsigned int> bodyColumns = m_body->widthOfTextInColumns();
   for(std::vector<unsigned int>::iterator it = columnWidths.begin(), itEnd=columnWidths.end(), itBody=bodyColumns.begin();
       it != itEnd;
       ++it,++itBody) {
      if(*itBody > *it) {
         *it = *itBody;
      }
   }
   m_header->setWidthOfTextInColumns(columnWidths);
   m_body->setWidthOfTextInColumns(columnWidths);
   m_rowHeader=0;
   if(m_rowHeaderTable) {
      m_rowHeader = new FWTabularWidget(m_rowHeaderTable,this);
   	AddFrame(m_rowHeader, new TGTableLayoutHints(0,1,1,2,kLHintsTop|kLHintsLeft|kColOptions));
   	m_rowHeader->Connect("buttonReleased(Int_t,Int_t,Int_t,Int_t)","FWTableWidget",this,"buttonReleasedInBody(Int_t,Int_t,Int_t,Int_t)");
      m_rowHeader->setWidthOfTextInColumns(m_rowHeader->widthOfTextInColumns());
   }

   m_hSlider = new TGHScrollBar(this);
   AddFrame(m_hSlider, new TGTableLayoutHints(1,2,2,3,kRowOptions));
   m_vSlider = new TGVScrollBar(this);
   AddFrame(m_vSlider, new TGTableLayoutHints(2,3,1,2,kColOptions));
   MapSubwindows();
   Layout();
   //HideFrame(m_hSlider);
   //HideFrame(m_vSlider);
   m_hSlider->Associate(this);
   m_vSlider->Associate(this);
   
   m_hSlider->SetEditDisabled(kEditDisable | kEditDisableGrab | kEditDisableBtnEnable);
   m_vSlider->SetEditDisabled(kEditDisable | kEditDisableGrab | kEditDisableBtnEnable);
   m_bodyTable->Connect("dataChanged()","FWTableWidget",this,"dataChanged()");
}

// FWTableWidget::FWTableWidget(const FWTableWidget& rhs)
// {
//    // do actual copying here;
// }

FWTableWidget::~FWTableWidget()
{
}

//
// assignment operators
//
// const FWTableWidget& FWTableWidget::operator=(const FWTableWidget& rhs)
// {
//   //An exception safe implementation is
//   FWTableWidget temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWTableWidget::sort(UInt_t iColumn, bool iDescendingSort)
{
   m_headerTable->sort(iColumn,iDescendingSort);
   m_bodyTable->sort(iColumn,iDescendingSort);
   m_sortedColumn =iColumn;
   m_descendingSort=iDescendingSort;
   
   //fClient->NeedRedraw(m_header);
   //fClient->NeedRedraw(m_body);
}

void 
FWTableWidget::Resize(UInt_t w, UInt_t h)
{
   handleResize(w,h);
   TGCompositeFrame::Resize(w,h);
}

void 
FWTableWidget::handleResize(UInt_t w, UInt_t h)
{
   //std::cout <<"Resize"<<std::endl;
   bool redoLayout=false;
   TGDimension def = m_body->GetDefaultSize();
   UInt_t fullWidth = def.fWidth+m_vSlider->GetWidth();
   if(m_rowHeader) {
      TGDimension def = m_rowHeader->GetDefaultSize();
      fullWidth+=def.fWidth;
   }
   if(w < fullWidth) {
      if(!m_showingHSlider) {
         ShowFrame(m_hSlider);
         redoLayout=true;
         m_showingHSlider=true;
      }
      m_hSlider->SetRange(fullWidth,w);
   } else {
      if(m_showingHSlider) {
         HideFrame(m_hSlider);
         m_hSlider->SetPosition(0);
         m_showingHSlider = false;
         redoLayout=true;
      }
   }

   UInt_t fullHeight = def.fHeight+m_header->GetHeight()+m_hSlider->GetHeight();
   if(h < fullHeight) {
      if(!m_showingVSlider) {
         ShowFrame(m_vSlider);
         m_showingVSlider=true;
         redoLayout=true;
      }
      m_vSlider->SetRange(fullHeight,h);
   } else {
      if(m_showingVSlider) {
         HideFrame(m_vSlider);
         m_vSlider->SetPosition(0);
         m_showingVSlider = false;
         redoLayout=true;
      }
   }
   if(redoLayout) {
      Layout();
   }
}

void 
FWTableWidget::SetSize(const TGDimension &s)
{
   //std::cout <<"SetSize"<<std::endl;
   TGCompositeFrame::SetSize(s);
}

void    
FWTableWidget::MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   //std::cout <<"MoveResize"<<std::endl;
   if(w != GetWidth() || h != GetHeight()) {
      handleResize(w,h);
   }
   TGCompositeFrame::MoveResize(x,y,w,h);
}

Bool_t 
FWTableWidget::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Handle message generated by the canvas scrollbars.

   switch (GET_MSG(msg)) {
      case kC_HSCROLL:
         switch (GET_SUBMSG(msg)) {
            case kSB_SLIDERTRACK:
            case kSB_SLIDERPOS:
               m_body->setHorizontalOffset(parm1);
               m_header->setHorizontalOffset(parm1);
               break;
         }
         break;

      case kC_VSCROLL:
         switch (GET_SUBMSG(msg)) {
            case kSB_SLIDERTRACK:
            case kSB_SLIDERPOS:
               m_body->setVerticalOffset(parm1);
               if(m_rowHeader) {
                  m_rowHeader->setVerticalOffset(parm1);
               }
               break;
         }
         break;

      default:
         break;
   }
   return kTRUE;
}

void 
FWTableWidget::buttonReleasedInHeader(Int_t row, Int_t column, Int_t btn, Int_t keyMod)
{
   if(btn != kButton1 && btn != kButton3) {return;}
   if(m_sortedColumn==column) {
      sort(column, !m_descendingSort);
   } else {
      sort(column,true);
   }
}

void 
FWTableWidget::buttonReleasedInBody(Int_t row, Int_t column, Int_t btn, Int_t keyMod)
{
   if(btn == kButton5){
      //should scroll down
      if(m_vSlider) {
         Int_t p = m_vSlider->GetPosition();
         Int_t mx = m_vSlider->GetRange();
         p+=m_vSlider->GetSmallIncrement();
         if(p>mx){ p=mx;}
         m_vSlider->SetPosition(p);
      }
      return;
   }
   if(btn == kButton4){
      //should scroll up
      if(m_vSlider) {
         Int_t p = m_vSlider->GetPosition();
         p -=m_vSlider->GetSmallIncrement();
         if(0>p) {p=0;}
         m_vSlider->SetPosition(p);
      }
      return;
   }
   if(btn != kButton1 && btn != kButton3) {return;}
   if(row>=-1 and row < m_bodyTable->numberOfRows()) {
      rowClicked(m_bodyTable->unsortedRowNumber(row),btn,keyMod);
   }
}

void 
FWTableWidget::rowClicked(Int_t row, Int_t btn, Int_t keyMod)
{
   keyMod = (keyMod&(kKeyShiftMask|kKeyControlMask));
   //std::cout <<"rowClicked "<<row<<" "<<btn<<" "<<keyMod<<std::endl;
   Long_t args[3];
   args[0]=(Long_t)row;
   args[1]=(Long_t)btn;
   args[2]=(Long_t)keyMod;
   Emit("rowClicked(Int_t,Int_t,Int_t)",args);      
}

void 
FWTableWidget::dataChanged()
{
   //set sizes
   std::vector<unsigned int> columnWidths = m_header->widthOfTextInColumns();
   std::vector<unsigned int> bodyColumns = m_body->widthOfTextInColumns();
   for(std::vector<unsigned int>::iterator it = columnWidths.begin(), itEnd=columnWidths.end(), itBody=bodyColumns.begin();
       it != itEnd;
       ++it,++itBody) {
      if(*itBody > *it) {
         *it = *itBody;
      }
   }
   m_header->setWidthOfTextInColumns(columnWidths);
   m_body->setWidthOfTextInColumns(columnWidths);
}

//
// const member functions
//

//
// static member functions
//

ClassImp(FWTableWidget)
