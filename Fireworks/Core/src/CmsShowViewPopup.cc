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
// $Id: CmsShowViewPopup.cc,v 1.13 2009/07/17 08:01:57 amraktad Exp $
//

// system include files
#include <iostream>
#include <boost/checked_delete.hpp>
#include <boost/bind.hpp>
#include "TGFrame.h"
#include "TGLabel.h"
#include "TGButton.h"
#include "TG3DLine.h"
#include "TEveWindow.h"

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
CmsShowViewPopup::CmsShowViewPopup(const TGWindow* p, UInt_t w, UInt_t h, FWColorManager* iCMgr, TEveWindow* ew) :
   TGTransientFrame(gClient->GetDefaultRoot(),p, w, h),
   m_colorManager(iCMgr),
   m_eveWindow(0),
   m_mapped(kFALSE)
{
   m_colorManager->colorsHaveChanged_.connect(boost::bind(&CmsShowViewPopup::backgroundColorWasChanged,this));

   SetCleanup(kDeepCleanup);

   // label
   TGHorizontalFrame* viewFrame = new TGHorizontalFrame(this);
   m_viewLabel = new TGLabel(viewFrame, "No view selected");
   TGFont* defaultFont = gClient->GetFontPool()->GetFont(m_viewLabel->GetDefaultFontStruct());
   m_viewLabel->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
   m_viewLabel->SetTextJustify(kTextLeft);
   viewFrame->AddFrame(m_viewLabel, new TGLayoutHints(kLHintsExpandX));
   AddFrame(viewFrame, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
   // background
   AddFrame(new TGHorizontal3DLine(this, 200, 5), new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   m_changeBackground = new TGTextButton(this,"Change Background Color");
   backgroundColorWasChanged();
   AddFrame(m_changeBackground);
   m_changeBackground->Connect("Clicked()","CmsShowViewPopup",this,"changeBackground()");
   // save image
   m_saveImageButton= new TGTextButton(this,"Save Image ...");
   AddFrame(m_saveImageButton);
   m_saveImageButton->Connect("Clicked()","CmsShowViewPopup",this,"saveImage()");
   // content frame
   AddFrame(new TGHorizontal3DLine(this, 200, 5), new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   m_viewContentFrame = new TGCompositeFrame(this);
   AddFrame(m_viewContentFrame,new TGLayoutHints(kLHintsExpandX|kLHintsExpandY));
   reset(ew);

   SetWindowName("View Controller");
   MapWindow();
}

// CmsShowViewPopup::CmsShowViewPopup(const CmsShowViewPopup& rhs)
// {
//    // do actual copying here;
// }

CmsShowViewPopup::~CmsShowViewPopup()
{
}

void
CmsShowViewPopup::reset(TEveWindow* ew)
{
   m_eveWindow = ew;
   FWViewBase* viewBase = m_eveWindow ? (FWViewBase*)ew->GetUserData() : 0;

   // clear content (can be better: delete subframes)
   m_viewContentFrame->UnmapWindow();
   RemoveFrame(m_viewContentFrame);
   m_viewContentFrame->DestroyWindow();
   delete m_viewContentFrame;
   m_setters.clear();

   m_viewContentFrame = new TGCompositeFrame(this);
   AddFrame(m_viewContentFrame);
   // fill content
   if(viewBase) {
      m_saveImageButton->SetEnabled(kTRUE);

      m_viewLabel->SetText(viewBase->typeName().c_str());
      for(FWParameterizable::const_iterator itP = viewBase->begin(), itPEnd = viewBase->end();
          itP != itPEnd;
          ++itP) {
         boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(*itP) );
         ptr->attach(*itP, this);
         TGFrame* pframe = ptr->build(m_viewContentFrame);
         m_viewContentFrame->AddFrame(pframe,new TGLayoutHints(kLHintsTop));
         m_setters.push_back(ptr);
      }
   }
   else {
      m_viewLabel->SetText("No view selected");
      m_saveImageButton->SetEnabled(kFALSE);
   }

   MapSubwindows();
   Resize(GetDefaultSize());
   Layout();
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
CmsShowViewPopup::CloseWindow()
{
   UnmapWindow();
   closed_.emit();
}

void
CmsShowViewPopup::MapWindow()
{
   TGWindow::MapWindow();
   m_mapped = true;
}

void
CmsShowViewPopup::UnmapWindow()
{
   TGWindow::UnmapWindow();
   m_mapped = false;
}


//______________________________________________________________________________
// callbacks
//
void
CmsShowViewPopup::saveImage()
{
   if(m_eveWindow) {
      FWViewBase* viewBase = (FWViewBase*)(m_eveWindow->GetUserData());
      viewBase->promptForSaveImageTo(this);
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
