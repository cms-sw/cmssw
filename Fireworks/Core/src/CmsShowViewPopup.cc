// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowViewPopup
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed Jun 25 15:15:04 EDT 2008
// $Id: CmsShowViewPopup.cc,v 1.1 2008/06/29 13:23:47 chrjones Exp $
//

// system include files
#include <iostream>
#include <boost/checked_delete.hpp>
#include "TGFrame.h"
#include "TGLabel.h"
#include "TGButton.h"

// user include files
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowViewPopup::CmsShowViewPopup(const TGWindow* p, UInt_t w, UInt_t h, FWViewBase* v) : 
TGTransientFrame(gClient->GetDefaultRoot(),p, w, h)
{
  m_view = v;
  SetCleanup(kDeepCleanup);
  TGHorizontalFrame* viewFrame = new TGHorizontalFrame(this);
  m_viewLabel = new TGLabel(viewFrame, v->typeName().c_str());
  TGFont* defaultFont = gClient->GetFontPool()->GetFont(m_viewLabel->GetDefaultFontStruct());
  m_viewLabel->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
  m_viewLabel->SetTextJustify(kTextLeft);
  viewFrame->AddFrame(m_viewLabel, new TGLayoutHints(kLHintsExpandX));
  m_removeButton = new TGTextButton(viewFrame, "Remove", -1, TGTextButton::GetDefaultGC()(), TGTextButton::GetDefaultFontStruct(), kRaisedFrame|kDoubleBorder|kFixedWidth);
  m_removeButton->SetWidth(60);
  m_removeButton->SetEnabled(kFALSE);
  viewFrame->AddFrame(m_removeButton);
  AddFrame(viewFrame, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
  m_viewContentFrame = new TGCompositeFrame(this);
  std::for_each(m_setters.begin(),m_setters.end(),
		boost::checked_deleter<FWParameterSetterBase>());
  m_setters.clear();
  for(FWParameterizable::const_iterator itP = v->begin(), itPEnd = v->end();
       itP != itPEnd;
       ++itP) {
      std::auto_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(*itP) );
      ptr->attach(*itP, this);
      TGFrame* pframe = ptr->build(m_viewContentFrame);
      m_viewContentFrame->AddFrame(pframe,new TGLayoutHints(kLHintsTop));
      m_setters.push_back(ptr.release());
  }
  AddFrame(m_viewContentFrame);
  SetWindowName("View");
  std::cout<<"Default size: "<<GetDefaultWidth()<<", "<<GetDefaultHeight()<<std::endl;
  Resize(GetDefaultSize());
  MapSubwindows();
  Layout();
  MapWindow();
}

// CmsShowViewPopup::CmsShowViewPopup(const CmsShowViewPopup& rhs)
// {
//    // do actual copying here;
// }

CmsShowViewPopup::~CmsShowViewPopup()
{
}

//
// assignment operators
//
// const CmsShowViewPopup& CmsShowViewPopup::operator=(const CmsShowViewPopup& rhs)
// {
//   //An exception safe implementation is
//   CmsShowViewPopup temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
CmsShowViewPopup::reset(FWViewBase* iView) {
  m_viewLabel->SetText(iView->typeName().c_str());
  //  m_viewContentFrame->RemoveFrame(m_view->frame());
  //  m_viewContentFrame->AddFrame(iView->frame());
  //  m_view = iView;
  m_viewContentFrame->UnmapWindow();
  RemoveFrame(m_viewContentFrame);
  m_viewContentFrame->DestroyWindow();
  delete m_viewContentFrame;
  m_viewContentFrame = new TGCompositeFrame(this);
  std::for_each(m_setters.begin(),m_setters.end(),
		boost::checked_deleter<FWParameterSetterBase>());
  m_setters.clear();
  for(FWParameterizable::const_iterator itP = iView->begin(), itPEnd = iView->end();
       itP != itPEnd;
       ++itP) {
      std::auto_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(*itP) );
      ptr->attach(*itP, this);
      TGFrame* pframe = ptr->build(m_viewContentFrame);
      m_viewContentFrame->AddFrame(pframe,new TGLayoutHints(kLHintsTop));
      m_setters.push_back(ptr.release());
  }
  AddFrame(m_viewContentFrame);

  std::cout<<"Default size: "<<GetDefaultWidth()<<", "<<GetDefaultHeight()<<std::endl;
  Resize(GetDefaultSize());
  MapSubwindows();
  Layout();
}  

void
CmsShowViewPopup::removeView() {
  printf("Removed!\n");
}


//
// const member functions
//

//
// static member functions
//
