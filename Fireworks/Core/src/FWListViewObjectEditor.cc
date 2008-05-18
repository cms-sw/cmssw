// -*- C++ -*-
//
// Package:     Core
// Class  :     FWListViewObjectEditor
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 10 09:02:57 CDT 2008
// $Id: FWListViewObjectEditor.cc,v 1.1 2008/03/11 02:45:43 chrjones Exp $
//

// system include files
#include <assert.h>
#include <boost/checked_delete.hpp>

// user include files
#include "Fireworks/Core/src/FWListViewObjectEditor.h"
#include "Fireworks/Core/src/FWListViewObject.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWParameterizable.h"
#include "Fireworks/Core/interface/FWViewBase.h"



//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWListViewObjectEditor::FWListViewObjectEditor(const TGWindow* p, 
                                               Int_t width,
                                               Int_t height,
                                               UInt_t options,
                                               Pixel_t back):
TGedFrame(p, width, height, options | kVerticalFrame, back),
m_frame(0)
{
   MakeTitle("FWListModel");
}

// FWListViewObjectEditor::FWListViewObjectEditor(const FWListViewObjectEditor& rhs)
// {
//    // do actual copying here;
// }

FWListViewObjectEditor::~FWListViewObjectEditor()
{
   std::for_each(m_setters.begin(),m_setters.end(),
                 boost::checked_deleter<FWParameterSetterBase>());
}

//
// assignment operators
//
// const FWListViewObjectEditor& FWListViewObjectEditor::operator=(const FWListViewObjectEditor& rhs)
// {
//   //An exception safe implementation is
//   FWListViewObjectEditor temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWListViewObjectEditor::SetModel(TObject* iView)
{
   FWListViewObject* obj = dynamic_cast<FWListViewObject*>(iView);
   assert(0!=obj);
   
   FWViewBase* view = obj->view();
   
   if (m_frame) {
      m_frame->UnmapWindow();
      RemoveFrame(m_frame);
      m_frame->DestroyWindow();
      delete m_frame;
   }
   m_frame = new TGVerticalFrame(this);
   AddFrame(m_frame);

   std::for_each(m_setters.begin(),m_setters.end(),
                 boost::checked_deleter<FWParameterSetterBase>());
   m_setters.clear();
   for(FWParameterizable::const_iterator itP = view->begin(), itPEnd = view->end();
       itP != itPEnd;
       ++itP) {
      std::auto_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(*itP) );
      ptr->attach(*itP,this);
      TGFrame* pframe = ptr->build(m_frame);
      m_frame->AddFrame(pframe,new TGLayoutHints(kLHintsTop));
      m_setters.push_back(ptr.release());
   }
   
   MapSubwindows();
}
//
// const member functions
//

//
// static member functions
//
ClassImp(FWListViewObjectEditor)
