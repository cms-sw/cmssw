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
// $Id: CmsShowViewPopup.cc,v 1.8 2009/01/23 21:35:42 amraktad Exp $
//

// system include files
#include <iostream>
#include <boost/checked_delete.hpp>
#include <boost/bind.hpp>
#include "TGFrame.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TG3DLine.h"

// user include files
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/interface/FWColorManager.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowViewPopup::CmsShowViewPopup(const TGWindow* p, UInt_t w, UInt_t h, FWColorManager* iCMgr, FWViewBase* v) :
   TGTransientFrame(gClient->GetDefaultRoot(),p, w, h),
   m_colorManager(iCMgr)
{
   m_view = v;
   SetCleanup(kDeepCleanup);
   m_colorManager->colorsHaveChanged_.connect(boost::bind(&CmsShowViewPopup::backgroundColorWasChanged,this));
   
   TGHorizontalFrame* viewFrame = new TGHorizontalFrame(this);
   m_viewLabel = new TGLabel(viewFrame, v->typeName().c_str());
   TGFont* defaultFont = gClient->GetFontPool()->GetFont(m_viewLabel->GetDefaultFontStruct());
   m_viewLabel->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
   m_viewLabel->SetTextJustify(kTextLeft);
   m_viewLabel->SetText("No view selected");
   viewFrame->AddFrame(m_viewLabel, new TGLayoutHints(kLHintsExpandX));
#if defined(CAN_REMOVE_ANY_VIEW)
   m_removeButton = new TGTextButton(viewFrame, "Remove", -1, TGTextButton::GetDefaultGC() (), TGTextButton::GetDefaultFontStruct(), kRaisedFrame|kDoubleBorder|kFixedWidth);
   m_removeButton->SetWidth(60);
   m_removeButton->SetEnabled(kFALSE);
   viewFrame->AddFrame(m_removeButton);
#endif
   AddFrame(viewFrame, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
   AddFrame(new TGHorizontal3DLine(this, 200, 5), new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   m_changeBackground = new TGTextButton(this,"Change Background Color");
   //initializes the text
   backgroundColorWasChanged();
   AddFrame(m_changeBackground);
   m_changeBackground->Connect("Clicked()","CmsShowViewPopup",this,"changeBackground()");
   m_saveImageButton= new TGTextButton(this,"Save Image ...");
   AddFrame(m_saveImageButton);
   if(!m_view) {
      m_saveImageButton->SetEnabled(kFALSE);
   }
   m_saveImageButton->Connect("Clicked()","CmsShowViewPopup",this,"saveImage()");

   AddFrame(new TGHorizontal3DLine(this, 200, 5), new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   m_viewContentFrame = new TGCompositeFrame(this);
   m_setters.clear();
   for(FWParameterizable::const_iterator itP = v->begin(), itPEnd = v->end();
       itP != itPEnd;
       ++itP) {
      boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(*itP) );
      ptr->attach(*itP, this);
      TGFrame* pframe = ptr->build(m_viewContentFrame);
      m_viewContentFrame->AddFrame(pframe,new TGLayoutHints(kLHintsTop));
      m_setters.push_back(ptr);
   }
   AddFrame(m_viewContentFrame,new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   SetWindowName("View Controller");
   //std::cout<<"Default size: "<<GetDefaultWidth()<<", "<<GetDefaultHeight()<<std::endl;
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
   //  m_viewContentFrame->RemoveFrame(m_view->frame());
   //  m_viewContentFrame->AddFrame(iView->frame());
   m_view = iView;
   m_viewContentFrame->UnmapWindow();
   RemoveFrame(m_viewContentFrame);
   m_viewContentFrame->DestroyWindow();
   delete m_viewContentFrame;
   m_viewContentFrame = new TGCompositeFrame(this);
   m_setters.clear();
   if(iView) {
      m_saveImageButton->SetEnabled(kTRUE);

      m_viewLabel->SetText(iView->typeName().c_str());
      for(FWParameterizable::const_iterator itP = iView->begin(), itPEnd = iView->end();
          itP != itPEnd;
          ++itP) {
         boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(*itP) );
         ptr->attach(*itP, this);
         TGFrame* pframe = ptr->build(m_viewContentFrame);
         m_viewContentFrame->AddFrame(pframe,new TGLayoutHints(kLHintsTop));
         m_setters.push_back(ptr);
      }
   } else {
      m_viewLabel->SetText("No view selected");
      m_saveImageButton->SetEnabled(kFALSE);
   }
   AddFrame(m_viewContentFrame);

   //std::cout<<"Default size: "<<GetDefaultWidth()<<", "<<GetDefaultHeight()<<std::endl;
   Resize(GetDefaultSize());
   MapSubwindows();
   Layout();
}

void
CmsShowViewPopup::removeView() {
   //printf("Removed!\n");
}

void
CmsShowViewPopup::saveImage()
{
   if(m_view) {
      m_view->promptForSaveImageTo(this);
   }
}

void 
CmsShowViewPopup::changeBackground()
{
   m_colorManager->setBackgroundColorIndex( FWColorManager::kBlackIndex == m_colorManager->backgroundColorIndex()?
                                            FWColorManager::kWhiteIndex: 
                                            FWColorManager::kBlackIndex);
}

void
CmsShowViewPopup::backgroundColorWasChanged()
{
   if(FWColorManager::kBlackIndex == m_colorManager->backgroundColorIndex()) {
      m_changeBackground->SetText("Change Background Color to White");
   } else {
      m_changeBackground->SetText("Change Background Color to Black");
   }
}

//
// const member functions
//

//
// static member functions
//
