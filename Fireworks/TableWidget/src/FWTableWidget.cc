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
// $Id: FWTableWidget.cc,v 1.25 2012/02/22 00:15:44 amraktad Exp $
//

// system include files
#include <iostream>
#include "TGScrollBar.h"
#include "TGTableLayout.h"
#include "TGResourcePool.h"


// user include files
#include "Fireworks/TableWidget/interface/FWTableWidget.h"
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/src/FWTabularWidget.h"
#include "Fireworks/TableWidget/src/FWAdapterHeaderTableManager.h"
#include "Fireworks/TableWidget/src/FWAdapterRowHeaderTableManager.h"

static const unsigned long kWidgetColor = 0x2f2f2f;

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
   m_headerTable(iManager->hasLabelHeaders()?new FWAdapterHeaderTableManager(iManager): static_cast<FWTableManagerBase*>(0)),
   m_rowHeaderTable(iManager->hasRowHeaders()?new FWAdapterRowHeaderTableManager(iManager): static_cast<FWTableManagerBase*>(0)),
   m_header(0),
   m_rowHeader(0),
   m_showingVSlider(true),
   m_showingHSlider(true),
   m_sortedColumn(-1),
   m_descendingSort(true),
   m_forceLayout(false),
   m_headerBackground(0),
   m_headerForeground(0),
   m_lineSeparator(0)
{
   SetLayoutManager( new TGTableLayout(this,3,3) );
   
   if(0!=m_headerTable) {
      m_header = new FWTabularWidget(m_headerTable,this);
      AddFrame(m_header, new TGTableLayoutHints(1,2,0,1,kLHintsTop|kLHintsLeft|kRowOptions));	
      if (m_bodyTable->cellDataIsSortable()) m_header->Connect("buttonReleased(Int_t,Int_t,Event_t*,Int_t,Int_t)","FWTableWidget",this,"buttonReleasedInHeader(Int_t,Int_t,Event_t*,Int_t,Int_t)");
   }
   m_body = new FWTabularWidget(iManager,this,GetWhiteGC()());
   //m_body->SetBackgroundColor(kWidgetColor);
   AddFrame(m_body, new TGTableLayoutHints(1,2,1,2,kLHintsTop|kLHintsLeft|kRowOptions|kColOptions));
   m_body->Connect("buttonReleased(Int_t,Int_t,Event_t*,Int_t,Int_t)","FWTableWidget",this,"buttonReleasedInBody(Int_t,Int_t,Event_t*,Int_t,Int_t)");

   //set sizes
   std::vector<unsigned int> columnWidths = m_body->widthOfTextInColumns();
   if(0!=m_header) {
      std::vector<unsigned int> headerWidths = m_header->widthOfTextInColumns();
      for(std::vector<unsigned int>::iterator it = columnWidths.begin(), itEnd=columnWidths.end(), itHeader=headerWidths.begin();
          it != itEnd;
          ++it,++itHeader) {
         if(*itHeader > *it) {
            *it = *itHeader;
         }
      }
   }
   if(0!=m_header) {
      m_header->setWidthOfTextInColumns(columnWidths);
   }
   m_body->setWidthOfTextInColumns(columnWidths);
   if(m_rowHeaderTable) {
      m_rowHeader = new FWTabularWidget(m_rowHeaderTable,this, GetWhiteGC()());
      //m_rowHeader->SetBackgroundColor(kWidgetColor);

      AddFrame(m_rowHeader, new TGTableLayoutHints(0,1,1,2,kLHintsTop|kLHintsLeft|kColOptions));
      m_rowHeader->Connect("buttonReleased(Int_t,Int_t,Event_t*,Int_t,Int_t)","FWTableWidget",this,"buttonReleasedInBody(Int_t,Int_t,Event_t*,Int_t,Int_t)");
      m_rowHeader->Connect("buttonReleased(Int_t,Int_t,Event_t*,Int_t,Int_t)","FWTableWidget",this,"buttonReleasedInRowHeader(Int_t,Int_t,Event_t*,Int_t,Int_t)");
      m_rowHeader->setWidthOfTextInColumns(m_rowHeader->widthOfTextInColumns());
   }

   m_hSlider = new TGHScrollBar(this);
   AddFrame(m_hSlider, new TGTableLayoutHints(1,2,2,3,kRowOptions));
   m_hSlider->Connect("ProcessedEvent(Event_t*)", "FWTableWidget", this, "childrenEvent(Event_t *)");
   m_vSlider = new TGVScrollBar(this);
   m_vSlider->SetSmallIncrement(12);
   AddFrame(m_vSlider, new TGTableLayoutHints(2,3,1,2,kColOptions));
   m_vSlider->Connect("ProcessedEvent(Event_t*)", "FWTableWidget", this, "childrenEvent(Event_t *)");
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
   if(0!=m_headerBackground) {
      gClient->GetResourcePool()->GetGCPool()->FreeGC(m_headerBackground->GetGC());
   }
   if(0!= m_headerForeground) {
      gClient->GetResourcePool()->GetGCPool()->FreeGC(m_headerForeground->GetGC());
   }
   
   if(0!= m_lineSeparator) {
      gClient->GetResourcePool()->GetGCPool()->FreeGC(m_lineSeparator->GetGC());
   }
      
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
   if(0!=m_headerTable) {
      m_headerTable->sort(iColumn,iDescendingSort);
   }
   m_bodyTable->sort(iColumn,iDescendingSort);
   m_sortedColumn =iColumn;
   m_descendingSort=iDescendingSort;
   
   //fClient->NeedRedraw(m_header);
   //fClient->NeedRedraw(m_body);
}

void 
FWTableWidget::SetBackgroundColor(Pixel_t iColor)
{
   TGFrame::SetBackgroundColor(iColor);
   if(m_rowHeaderTable) {
      m_rowHeader->SetBackgroundColor(iColor);
      fClient->NeedRedraw(m_rowHeader);
   }
   if(m_header) {
      m_header->SetBackgroundColor(iColor);
      fClient->NeedRedraw(m_header);
   }
   m_body->SetBackgroundColor(iColor);
   fClient->NeedRedraw(m_body);
   fClient->NeedRedraw(this);
}

void 
FWTableWidget::SetHeaderBackgroundColor(Pixel_t iColor)
{
   if(0==m_headerBackground) {
      GCValues_t t = *(gClient->GetResourcePool()->GetFrameGC()->GetAttributes());
      m_headerBackground = gClient->GetResourcePool()->GetGCPool()->GetGC(&t,kTRUE);
   }
   m_headerBackground->SetForeground(iColor);
   if(0!=m_header) {
      m_header->setBackgroundAreaContext((*m_headerBackground)());
   }
}
void 
FWTableWidget::SetHeaderForegroundColor(Pixel_t iColor)
{
   if(0==m_headerForeground) {
      GCValues_t t = *(gClient->GetResourcePool()->GetFrameGC()->GetAttributes());
      m_headerForeground = gClient->GetResourcePool()->GetGCPool()->GetGC(&t,kTRUE);
   }
   m_headerForeground->SetForeground(iColor);
   if(0!=m_header) {
      m_header->setLineContext((*m_headerForeground)());
   }
}

void 
FWTableWidget::SetLineSeparatorColor(Pixel_t iColor)
{
   if(0==m_lineSeparator) {
      GCValues_t t = *(gClient->GetResourcePool()->GetFrameGC()->GetAttributes());
      m_lineSeparator = gClient->GetResourcePool()->GetGCPool()->GetGC(&t,kTRUE);
   }
   m_lineSeparator->SetForeground(iColor);
   m_body->setLineContext( (*m_lineSeparator)());
   if(m_rowHeader) {
      m_rowHeader->setLineContext( (*m_lineSeparator)() );
   }
}


void 
FWTableWidget::Resize(UInt_t w, UInt_t h)
{
   handleResize(w,h);
   TGCompositeFrame::Resize(w,h);
}

bool 
FWTableWidget::handleResize(UInt_t w, UInt_t h)
{
   //std::cout <<"Resize"<<std::endl;
   bool redoLayout=false;

   TGDimension def  = m_body->GetDefaultSize();
   UInt_t fullWidth = def.fWidth;
   if(m_rowHeader) {
      fullWidth += m_rowHeader->GetDefaultSize().fWidth;
   }

   UInt_t headerHeight = 0;
   if(m_header) {
      headerHeight = m_header->GetHeight();
   }
   UInt_t fullHeight = def.fHeight + headerHeight;

   UInt_t sBarWidth  = (h < fullHeight) ? m_vSlider->GetWidth()  : 0;
   UInt_t sBarHeight = (w < fullWidth)  ? m_hSlider->GetHeight() : 0;
   if (sBarWidth == 0 && sBarHeight > 0 && h < fullHeight + sBarHeight)
      sBarWidth = m_vSlider->GetWidth();
   else if (sBarHeight == 0 && sBarWidth > 0 && h < fullWidth + sBarWidth)
      sBarHeight = m_hSlider->GetHeight();
   fullWidth  += sBarWidth;
   fullHeight += sBarHeight;

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

   return redoLayout;
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
               if(m_header) {
                  m_header->setHorizontalOffset(parm1);
               }
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
FWTableWidget::buttonReleasedInHeader(Int_t row, Int_t column, Event_t* event,Int_t,Int_t)
{
   Int_t btn = event->fCode;
   Int_t keyMod = event->fState;
   //Int_t keyMod = event->fState;
   if (btn == kButton1 || btn == kButton3) {
	if(m_sortedColumn==column) {
	     sort(column, !m_descendingSort);
	} else {
	     sort(column,true);
	}
   }
   columnClicked(column, btn, keyMod);
}

void 
FWTableWidget::buttonReleasedInBody(Int_t row, Int_t column, Event_t* event,Int_t iRelX,Int_t iRelY)
{
   Int_t btn = event->fCode;
   Int_t keyMod = event->fState;
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
      Int_t globalX,globalY;
      Window_t childdum;
      gVirtualX->TranslateCoordinates(m_body->GetId(),
                                      gClient->GetDefaultRoot()->GetId(),
                                      event->fX,event->fY,globalX,globalY,childdum);
      cellClicked(m_bodyTable->unsortedRowNumber(row), column, btn, keyMod, globalX, globalY);
      rowClicked(m_bodyTable->unsortedRowNumber(row), btn,keyMod,globalX,globalY);
   }
}

void
FWTableWidget::cellClicked(Int_t row, Int_t column, Int_t btn, Int_t keyMod, Int_t iGlobalX, Int_t iGlobalY)
{
   keyMod = (keyMod&(kKeyShiftMask|kKeyControlMask));
   //std::cout <<"rowClicked "<<row<<" "<<btn<<" "<<keyMod<<std::endl;
   Long_t args[6];
   args[0]=(Long_t)row;
   args[1]=(Long_t)column;
   args[2]=(Long_t)btn;
   args[3]=(Long_t)keyMod;
   args[4]=(Long_t)iGlobalX;
   args[5]=(Long_t)iGlobalY;
   Emit("cellClicked(Int_t,Int_t,Int_t,Int_t,Int_t,Int_t)",args);
}

void
FWTableWidget::childrenEvent(Event_t *)
{
   Clicked();
}

void
FWTableWidget::Clicked()
{
   Emit("Clicked()");
}

void 
FWTableWidget::rowClicked(Int_t row, Int_t btn, Int_t keyMod, Int_t iGlobalX, Int_t iGlobalY)
{
   keyMod = (keyMod&(kKeyShiftMask|kKeyControlMask));
   //std::cout <<"rowClicked "<<row<<" "<<btn<<" "<<keyMod<<std::endl;
   Long_t args[5];
   args[0]=(Long_t)row;
   args[1]=(Long_t)btn;
   args[2]=(Long_t)keyMod;
   args[3]=(Long_t)iGlobalX;
   args[4]=(Long_t)iGlobalY;
   Emit("rowClicked(Int_t,Int_t,Int_t,Int_t,Int_t)",args);      
}

void 
FWTableWidget::columnClicked(Int_t column, Int_t btn, Int_t keyMod)
{
   keyMod = (keyMod&(kKeyShiftMask|kKeyControlMask));
   //std::cout <<"rowClicked "<<row<<" "<<btn<<" "<<keyMod<<std::endl;
   Long_t args[3];
   args[0]=(Long_t)column;
   args[1]=(Long_t)btn;
   args[2]=(Long_t)keyMod;
   Emit("columnClicked(Int_t,Int_t,Int_t)",args);      
}

void 
FWTableWidget::dataChanged()
{
   bool needs_layout = m_forceLayout; m_forceLayout = false;

   m_body->dataChanged();
   if(m_rowHeader) {
      m_rowHeader->dataChanged();
      m_rowHeader->setWidthOfTextInColumns(m_rowHeader->widthOfTextInColumns());
   }
   //set sizes
   std::vector<unsigned int> columnWidths = m_body->widthOfTextInColumns();
   if(m_header) {
      // reset header back to its internal max rather than the last width
      m_header->dataChanged();	
      std::vector<unsigned int> headerWidths = m_header->widthOfTextInColumns();
      for(std::vector<unsigned int>::iterator it = columnWidths.begin(), itEnd=columnWidths.end(), itHeader=headerWidths.begin();
          it != itEnd;
          ++it,++itHeader) {
         if(*itHeader > *it) {
            *it = *itHeader;
         }
      }
      m_header->setWidthOfTextInColumns(columnWidths);
   } 
   m_body->setWidthOfTextInColumns(columnWidths);

   //this updates sliders to match our new data
   bool layoutDoneByhandleResize = handleResize(GetWidth(), GetHeight());
   if (needs_layout && ! layoutDoneByhandleResize)
   {
      Layout();
   }
   gClient->NeedRedraw(m_body);
   if (m_header) gClient->NeedRedraw(m_header);
   if (m_rowHeader) gClient->NeedRedraw(m_rowHeader);
   
}

void 
FWTableWidget::buttonPressedInRowHeader(Int_t row, Int_t column, Event_t* event, Int_t relX, Int_t relY)
{
   Int_t btn = event->fCode;
   if(btn != kButton1 && btn != kButton3) {return;}
   m_bodyTable->buttonReleasedInRowHeader(row, event, relX, relY);
}
void 
FWTableWidget::buttonReleasedInRowHeader(Int_t row, Int_t column, Event_t* event, Int_t relX, Int_t relY)
{
   Int_t btn = event->fCode;
   if(btn != kButton1 && btn != kButton3) {return;}
   m_bodyTable->buttonReleasedInRowHeader(row, event, relX, relY);
}

//
// const member functions
//
TGDimension 
FWTableWidget::GetDefaultSize() const
{
   TGDimension returnValue;
   if(m_header){
      returnValue.fHeight += m_header->GetDefaultHeight();
   }
   if(m_rowHeader) {
      returnValue.fWidth += m_rowHeader->GetDefaultWidth();
   }
   returnValue = returnValue + m_body->GetDefaultSize();
   returnValue.fHeight += m_hSlider->GetDefaultHeight();
   returnValue.fWidth += m_vSlider->GetDefaultWidth();
   
   return returnValue;
}

void
FWTableWidget::disableGrowInWidth()
{
   m_body->disableGrowInWidth();
   if (m_header) m_header->disableGrowInWidth();
   if (m_rowHeader) m_rowHeader->disableGrowInWidth();
}

void
FWTableWidget::DoRedraw()
{
   // override virtual TGFrame::DoRedraw() to prevent call of gVirtualX->ClearArea();
}
//
// static member functions
//

ClassImp(FWTableWidget)
