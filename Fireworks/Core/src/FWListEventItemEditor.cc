// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListEventItemEditor
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Mar  3 09:36:01 EST 2008
// $Id: FWListEventItemEditor.cc,v 1.5 2008/09/22 20:13:33 chrjones Exp $
//

// system include files
#include "TGTextEntry.h"
#include "TGButton.h"
#include "TEveManager.h"

// user include files
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/src/FWListEventItemEditor.h"
#include "Fireworks/Core/src/FWListEventItem.h"
#include "Fireworks/Core/interface/FWEventItem.h"


//
// constants, enums and typedefs
//
ClassImp(FWListEventItemEditor)

//
// static data member definitions
//

//
// constructors and destructor
//
FWListEventItemEditor::FWListEventItemEditor(const TGWindow* p,
                                             Int_t width,
                                             Int_t height,
                                             UInt_t options,
                                             Pixel_t back):
TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   //std::cout <<"FWListEventItemEditor"<<std::endl;
   /*
   MakeTitle("FWListEventItem");
   TGGroupFrame* vf = new TGGroupFrame(this,"Object Filter",kVerticalFrame);

   m_filterExpression = new TGTextEntry(vf);
   vf->AddFrame(m_filterExpression, new TGLayoutHints(kLHintsExpandX,0,5,5,5));

   m_filterExpression->Connect("ReturnPressed()","FWListEventItemEditor",this,"runFilter()");
   m_filterRunExpressionButton = new TGTextButton(vf,"Run Filter");
   vf->AddFrame(m_filterRunExpressionButton);
   m_filterRunExpressionButton->Connect("Clicked()","FWListEventItemEditor",this,"runFilter()");
   AddFrame(vf, new TGLayoutHints(kLHintsTop, 0, 0, 0, 0));

   TGTextButton* removeItemButton = new TGTextButton(this,"Remove Item");
   removeItemButton->Connect("Clicked()", "FWListEventItemEditor",this,"removeItem()");
   AddFrame(removeItemButton, new TGLayoutHints(kLHintsTop,0,0,0,0));
    */
}

// FWListEventItemEditor::FWListEventItemEditor(const FWListEventItemEditor& rhs)
// {
//    // do actual copying here;
// }

FWListEventItemEditor::~FWListEventItemEditor()
{
}

//
// assignment operators
//
// const FWListEventItemEditor& FWListEventItemEditor::operator=(const FWListEventItemEditor& rhs)
// {
//   //An exception safe implementation is
//   FWListEventItemEditor temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWListEventItemEditor::SetModel(TObject* iObj)
{
   //std::cout <<"SetModel "<<iObj<<std::endl;
   m_item = dynamic_cast<FWListEventItem*>(iObj);
   if(0!=iObj) {
      assert(0!=m_item);

      //m_filterExpression->SetText(m_item->eventItem()->filterExpression().c_str());
      FWGUIManager::getGUIManager()->updateEDI(m_item->eventItem());
   }
}

void
FWListEventItemEditor::runFilter()
{
   /*
   if(m_item!=0) {
     m_item->eventItem()->setFilterExpression(m_filterExpression->GetText());
   }
    */
}

void
FWListEventItemEditor::removeItem()
{
   /*
  if (m_item != 0) {
    m_item->eventItem()->destroy();
    //    delete m_item;
    m_item = 0;
    gEve->EditElement(0);
    gEve->Redraw3D();
  }
    */
}
//
// const member functions
//

//
// static member functions
//
