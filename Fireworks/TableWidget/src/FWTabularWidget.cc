// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWTabularWidget
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:45:21 EST 2009
// $Id: FWTabularWidget.cc,v 1.16 2011/03/07 13:13:51 amraktad Exp $
//

// system include files
#include <cassert>
#include <limits.h>
#include <iostream>
#include "TGResourcePool.h"

// user include files
#include "Fireworks/TableWidget/src/FWTabularWidget.h"
#include "Fireworks/TableWidget/interface/FWTableManagerBase.h"
#include "Fireworks/TableWidget/interface/FWTableCellRendererBase.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

const int FWTabularWidget::kTextBuffer = 2;
const int FWTabularWidget::kSeperatorWidth = 1;

//
// constructors and destructor
//
FWTabularWidget::FWTabularWidget(FWTableManagerBase* iTable, const TGWindow* p, GContext_t context):
TGFrame(p),
m_table(iTable),
m_widthOfTextInColumns(m_table->numberOfColumns(),static_cast<unsigned int>(0)),
m_vOffset(0),
m_hOffset(0),
m_normGC(context),
m_backgroundGC(ULONG_MAX),
m_growInWidth(true)
{

   m_textHeight = iTable->cellHeight();
   m_widthOfTextInColumns = m_table->maxWidthForColumns();
      
   m_tableWidth=(kTextBuffer+kTextBuffer+kSeperatorWidth)*(m_widthOfTextInColumns.size())+kSeperatorWidth;
   for(std::vector<unsigned int>::const_iterator it = m_widthOfTextInColumns.begin(), itEnd = m_widthOfTextInColumns.end();
   it!=itEnd;
   ++it){
      m_tableWidth +=*it;
   }
   Resize();
   
   gVirtualX->GrabButton(fId,kAnyButton, kAnyModifier, kButtonPressMask|kButtonReleaseMask,kNone,kNone);
   m_table->Connect("visualPropertiesChanged()","FWTabularWidget",this,"needToRedraw()");
}

// FWTabularWidget::FWTabularWidget(const FWTabularWidget& rhs)
// {
//    // do actual copying here;
// }

FWTabularWidget::~FWTabularWidget()
{
   m_table->Disconnect("visualPropertiesChanged()", this, "needToRedraw()");
}

//
// assignment operators
//
// const FWTabularWidget& FWTabularWidget::operator=(const FWTabularWidget& rhs)
// {
//   //An exception safe implementation is
//   FWTabularWidget temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWTabularWidget::dataChanged()
{
   m_textHeight = m_table->cellHeight();
   setWidthOfTextInColumns(m_table->maxWidthForColumns());
}

void 
FWTabularWidget::needToRedraw()
{
   fClient->NeedRedraw(this);
}


void 
FWTabularWidget::setWidthOfTextInColumns(const std::vector<unsigned int>& iNew)
{
   assert(iNew.size() == static_cast<unsigned int>(m_table->numberOfColumns()));

   m_widthOfTextInColumns=iNew;
   if (m_growInWidth)
   {
      // with of columns grow to prevent resizing/flickering on next event
      m_widthOfTextInColumnsMax.resize(iNew.size());
      std::vector<unsigned int>::iterator k =  m_widthOfTextInColumnsMax.begin();
      for(std::vector<unsigned int>::iterator it = m_widthOfTextInColumns.begin(); it != m_widthOfTextInColumns.end(); ++it, ++k)
      {
         if ( *it < *k ) 
            *it = *k;
         else
            *k = *it;
      }
   }

   m_tableWidth=0;
   for(std::vector<unsigned int>::const_iterator it = m_widthOfTextInColumns.begin(), itEnd = m_widthOfTextInColumns.end();
       it!=itEnd;
       ++it){
      m_tableWidth +=*it+kTextBuffer+kTextBuffer+kSeperatorWidth;
   }
   m_tableWidth +=kSeperatorWidth;
}

void 
FWTabularWidget::setVerticalOffset(UInt_t iV)
{
   if(iV != m_vOffset) {
      m_vOffset = iV;
      fClient->NeedRedraw(this);
   }
}
void 
FWTabularWidget::setHorizontalOffset(UInt_t iH)
{
   if(iH != m_hOffset){
      m_hOffset = iH;
      fClient->NeedRedraw(this);
   }
}

Bool_t 
FWTabularWidget::HandleButton(Event_t *event)
{
   if (event->fType==kButtonPress) {
      Int_t row,col,relX,relY;
      translateToRowColumn(event->fX, event->fY, row, col,relX,relY);
      //std::cout <<"Press: "<<relX<<" "<<relY<<" "<<row<<" "<<col<<" "<<m_table->numberOfRows()<<" "<<m_table->numberOfColumns()<<std::endl;
      if (row >= 0 && row < m_table->numberOfRows() && col >= 0 && col < m_table->numberOfColumns()) {
         FWTableCellRendererBase* renderer = m_table->cellRenderer(row,col);
         if (renderer) { 
            renderer->buttonEvent(event,relX,relY);
         }
         buttonPressed(row,col,event,relX,relY);
      }
      return true;
   }
   if (event->fType==kButtonRelease) {
      Int_t row,col,relX,relY;
      translateToRowColumn(event->fX, event->fY, row, col,relX, relY);
      //std::cout <<"Release: "<<relX<<" "<<relY<<" "<<row<<" "<<col<<" "<<m_table->numberOfRows()<<" "<<m_table->numberOfColumns()<<std::endl;
      if (row >= 0 && row < m_table->numberOfRows() && col >= 0 && col < m_table->numberOfColumns()) {
         FWTableCellRendererBase* renderer = m_table->cellRenderer(row,col);
         if (renderer) {
            renderer->buttonEvent(event,relX,relY);
         }
         buttonReleased(row,col,event,relX,relY);
      }
      return true;
   }
   return false;
}

void 
FWTabularWidget::translateToRowColumn(Int_t iX, Int_t iY, Int_t& oRow, Int_t& oCol, Int_t& oRelX, Int_t& oRelY) const
{
   if( iX < 0 ) {
      oCol = -1;
      oRelX = 0;
   } else {
      if(iX+static_cast<Int_t>(m_hOffset) > static_cast<Int_t>(m_tableWidth) ) {
         oCol = m_widthOfTextInColumns.size();
         oRelX = 0;
      } else {
         iX +=m_hOffset;
         oCol = 0;
         for(std::vector<unsigned int>::const_iterator it = m_widthOfTextInColumns.begin(), itEnd = m_widthOfTextInColumns.end();
         it!=itEnd;
         ++it,++oCol){
            oRelX=iX-kTextBuffer;
            iX-=2*kTextBuffer+kSeperatorWidth+*it;
            if(iX <= 0) {
               break;
            }
         }
      }
   }
   if( iY < 0) {
      oRow = -1;
      oRelY=0;
   } else {
      oRow = (int)(float(iY+m_vOffset)/(m_textHeight+2*kTextBuffer+kSeperatorWidth));
      oRelY = iY-oRow*(m_textHeight+2*kTextBuffer+kSeperatorWidth)+m_vOffset-kTextBuffer;
      Int_t numRows = m_table->numberOfRows();
      if(oRow > numRows) {
         oRow = numRows;
         oRelY=0;
      }
   }
}

void 
FWTabularWidget::buttonPressed(Int_t row, Int_t column, Event_t* event, Int_t relX, Int_t relY)
{
   //std::cout <<"buttonPressed "<<row<<" "<<column<<std::endl;
   Long_t args[5];
   args[0]=(Long_t)row;
   args[1]=(Long_t)column;
   args[2]=(Long_t)event;
   args[3]=(Long_t)relX;
   args[4]=(Long_t)relY;
   Emit("buttonPressed(Int_t,Int_t,Event_t*,Int_t,Int_t)",args);
}
void 
FWTabularWidget::buttonReleased(Int_t row, Int_t column, Event_t* event, Int_t relX, Int_t relY)
{
   //std::cout <<"buttonReleased "<<row<<" "<<column<<std::endl;
   Long_t args[6];
   args[0]=(Long_t)row;
   args[1]=(Long_t)column;
   args[2]=(Long_t)event;
   args[3]=(Long_t)relX;
   args[4]=(Long_t)relY;
   Emit("buttonReleased(Int_t,Int_t,Event_t*,Int_t,Int_t)",args);   
}

void
FWTabularWidget::DoRedraw()
{
   TGFrame::DoRedraw();
   
   //std::cout <<"DoRedraw "<<m_tableWidth<<std::endl;
   
   const int yOrigin = -m_vOffset;
   const int xOrigin = -m_hOffset;
   const int visibleWidth = m_tableWidth+xOrigin-kSeperatorWidth;
   int y=yOrigin;
   if(m_backgroundGC != ULONG_MAX) {
      gVirtualX->FillRectangle(fId,m_backgroundGC,xOrigin,y,m_tableWidth,
                               GetHeight());
   }
   gVirtualX->DrawLine(fId, m_normGC, xOrigin, y, visibleWidth, y);
   //Draw data
   const  int numRows=m_table->numberOfRows();
   
   //figure out which rows and columns are visible
   Int_t startRow, startColumn,relX,relY;
   translateToRowColumn(0,0,startRow,startColumn,relX,relY);
   if(startRow<0) { startRow = 0;}
   if(startColumn<0) { startColumn=0;}
   Int_t endRow, endColumn;
   translateToRowColumn(GetWidth(),GetHeight(),endRow,endColumn,relX,relY);
   if(endRow >= numRows) {
      endRow = numRows-1;
   }
   if(endColumn >= static_cast<Int_t>(m_widthOfTextInColumns.size())) {
      endColumn = m_widthOfTextInColumns.size()-1;
   }
   //std::cout <<"start "<<startRow<<" "<<startColumn<<" end "<<endRow<<" "<<endColumn<<std::endl;
   
   //calculate offset for rows and columns
   Int_t rowOffset = (kSeperatorWidth+2*kTextBuffer+m_textHeight)*startRow;
   Int_t columnOffset=kSeperatorWidth+kTextBuffer+xOrigin;
   for(std::vector<unsigned int>::iterator itTextWidth = m_widthOfTextInColumns.begin(), itEnd = m_widthOfTextInColumns.begin()+startColumn;
   itTextWidth != itEnd; ++itTextWidth) {
      columnOffset+=*itTextWidth+kTextBuffer+kSeperatorWidth+kTextBuffer;
   }
   
   
   y+=rowOffset;
   for(int row = startRow; row <= endRow; ++row) {
      std::vector<unsigned int>::iterator itTextWidth = m_widthOfTextInColumns.begin()+startColumn;
      //int x=kSeperatorWidth+kTextBuffer+xOrigin;
      int x = columnOffset;
      y+=kTextBuffer+kSeperatorWidth;
      for(int col = startColumn;
      col <= endColumn;
      ++col,++itTextWidth) {
         m_table->cellRenderer(row,col)->draw(fId,x,y,*itTextWidth,m_textHeight);
         //UInt_t textWidth = font->TextWidth(itData->c_str(),-1);
         x+=*itTextWidth+kTextBuffer+kSeperatorWidth+kTextBuffer;
      }
      y+=+m_textHeight+kTextBuffer;
      gVirtualX->DrawLine(fId, m_normGC, xOrigin, y, visibleWidth, y);
   }

   //draw column separators
   int x=xOrigin;
   gVirtualX->DrawLine(fId,m_normGC,x,0,x,y);
   x+=kSeperatorWidth;
   for(std::vector<unsigned int>::iterator itTextWidth = m_widthOfTextInColumns.begin();
   itTextWidth != m_widthOfTextInColumns.end();
   ++itTextWidth) {
      x+=2*kTextBuffer+*itTextWidth;
      gVirtualX->DrawLine(fId,m_normGC,x,0,x,y);
      x+=kSeperatorWidth;
   }
}

void 
FWTabularWidget::setLineContext(GContext_t iContext)
{
   m_normGC = iContext;
}
void 
FWTabularWidget::setBackgroundAreaContext(GContext_t iContext)
{
   m_backgroundGC = iContext;
}

//
// const member functions
//
TGDimension 
FWTabularWidget::GetDefaultSize() const
{
   // returns default size

   UInt_t w = fWidth;
   if(! (GetOptions() & kFixedWidth) ) {
      w=m_tableWidth;
   }
   UInt_t h = fHeight;
   if(! (GetOptions() & kFixedHeight) ) {
      unsigned int numRows = m_table->numberOfRows();
      
      h = kSeperatorWidth+(m_textHeight+2*kTextBuffer+kSeperatorWidth)*(numRows);
   }
   return TGDimension(w, h);
}

//
// static member functions
//
const TGGC&  
FWTabularWidget::getDefaultGC()
{
   static const TGGC* s_default = gClient->GetResourcePool()->GetFrameGC();
   return *s_default;
}

ClassImp(FWTabularWidget)
