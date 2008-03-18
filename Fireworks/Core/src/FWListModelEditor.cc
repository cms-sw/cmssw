// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListModelEditor
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Mar  3 17:20:28 EST 2008
// $Id: FWListModelEditor.cc,v 1.1 2008/03/05 15:07:31 chrjones Exp $
//

// system include files
#include "TGButton.h"

// user include files
#include "Fireworks/Core/src/FWListModelEditor.h"
#include "Fireworks/Core/src/FWListModel.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//
ClassImp(FWListModelEditor)

//
// constructors and destructor
//
FWListModelEditor::FWListModelEditor(const TGWindow* p, Int_t width, Int_t height,
                                     UInt_t options, Pixel_t back) :
TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   MakeTitle("FWListModel");
   m_showDetailViewButton = new TGTextButton(this,"Open Detail View");
   m_showDetailViewButton->Connect("Clicked()","FWListModelEditor",this,"openDetailView()");

   AddFrame(m_showDetailViewButton);
}

// FWListModelEditor::FWListModelEditor(const FWListModelEditor& rhs)
// {
//    // do actual copying here;
// }

FWListModelEditor::~FWListModelEditor()
{
}

//
// assignment operators
//
// const FWListModelEditor& FWListModelEditor::operator=(const FWListModelEditor& rhs)
// {
//   //An exception safe implementation is
//   FWListModelEditor temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWListModelEditor::SetModel(TObject* obj)
{
   m_model = dynamic_cast<FWListModel*>(obj);
   m_showDetailViewButton->SetEnabled(m_model->hasDetailView());
}

void 
FWListModelEditor::openDetailView()
{
   m_model->openDetailView();
}

//
// const member functions
//

//
// static member functions
//
