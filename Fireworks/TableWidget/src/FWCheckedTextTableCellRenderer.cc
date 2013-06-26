// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWCheckedTextTableCellRenderer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Feb  3 14:29:51 EST 2009
// $Id: FWCheckedTextTableCellRenderer.cc,v 1.3 2009/03/23 19:08:16 amraktad Exp $
//

// system include files
#include "TVirtualX.h"

// user include files
#include "Fireworks/TableWidget/interface/FWCheckedTextTableCellRenderer.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWCheckedTextTableCellRenderer::FWCheckedTextTableCellRenderer(const TGGC* iContext):
FWTextTableCellRenderer(iContext), 
m_isChecked(false) {}

// FWCheckedTextTableCellRenderer::FWCheckedTextTableCellRenderer(const FWCheckedTextTableCellRenderer& rhs)
// {
//    // do actual copying here;
// }

FWCheckedTextTableCellRenderer::~FWCheckedTextTableCellRenderer()
{
}

//
// assignment operators
//
// const FWCheckedTextTableCellRenderer& FWCheckedTextTableCellRenderer::operator=(const FWCheckedTextTableCellRenderer& rhs)
// {
//   //An exception safe implementation is
//   FWCheckedTextTableCellRenderer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void FWCheckedTextTableCellRenderer::setChecked(bool iChecked) {
   m_isChecked = iChecked;
}

void 
FWCheckedTextTableCellRenderer::draw(Drawable_t iID, int iX, int iY, unsigned int iWidth, unsigned int iHeight)
{
   const UInt_t h = height();
   
   //draw the check box
   GContext_t c = graphicsContext()->GetGC();
   gVirtualX->DrawLine(iID,c,iX,iY,iX,iY+h);
   gVirtualX->DrawLine(iID,c,iX+h,iY+h,iX,iY+h);
   gVirtualX->DrawLine(iID,c,iX+h,iY+h,iX+h,iY);
   gVirtualX->DrawLine(iID,c,iX+h,iY,iX,iY);
   
   if(m_isChecked) {
      gVirtualX->DrawLine(iID,c,iX,iY+h/2,iX+h/2,iY+h);      
      gVirtualX->DrawLine(iID,c,iX+h,iY,iX+h/2,iY+h);      
   }
   FWTextTableCellRenderer::draw(iID,iX+kGap+h,iY,iWidth-kGap-h,iHeight);
}

void 
FWCheckedTextTableCellRenderer::buttonEvent(Event_t* iClickEvent, int iRelClickX, int iRelClickY)
{
   const int h = height();
   
   bool wasClicked = iClickEvent->fType==kButtonRelease &&
                     iRelClickX >=0 &&
                     iRelClickX <=h &&
                     iRelClickY >=0 &&
                     iRelClickY <=h;
   if(wasClicked) {
      //std::cout <<"clicked"<<std::endl;
      checkBoxClicked();
   }
}

void
FWCheckedTextTableCellRenderer::checkBoxClicked()
{
   Emit("checkBoxClicked()");
}

//
// const member functions
//
bool FWCheckedTextTableCellRenderer::isChecked() const
{
   return m_isChecked;
}

UInt_t FWCheckedTextTableCellRenderer::width() const
{
   UInt_t h = height();
   return FWTextTableCellRenderer::width()+kGap+h;
}


//
// static member functions
//
ClassImp(FWCheckedTextTableCellRenderer)

