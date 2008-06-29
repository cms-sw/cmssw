// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowModelPopup
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Fri Jun 27 11:23:08 EDT 2008
// $Id$
//

// system include file
#include <iostream>
#include <sigc++/sigc++.h>
#include "TClass.h"
#include "TGFrame.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGString.h"
#include "TColor.h"
#include "TG3DLine.h"
#include "TGFont.h"

// user include files
#include "Fireworks/Core/interface/CmsShowModelPopup.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/src/FWListModel.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/src/FWListModel.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowModelPopup::CmsShowModelPopup(const TGWindow* p, UInt_t w, UInt_t h)
{
  SetCleanup(kDeepCleanup);
  m_model = 0;
  TGHorizontalFrame* objectFrame = new TGHorizontalFrame(this);
  m_modelLabel = new TGLabel(objectFrame, " ");
  TGFont* defaultFont = gClient->GetFontPool()->GetFont(m_modelLabel->GetDefaultFontStruct());
  m_modelLabel->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
  m_modelLabel->SetTextJustify(kTextLeft);
  objectFrame->AddFrame(m_modelLabel, new TGLayoutHints(kLHintsExpandX));
  m_removeButton = new TGTextButton(objectFrame, "Remove", -1, TGTextButton::GetDefaultGC()(), TGTextButton::GetDefaultFontStruct(), kRaisedFrame|kDoubleBorder|kFixedWidth);
  m_removeButton->SetWidth(60);
  m_removeButton->SetEnabled(kFALSE);
  objectFrame->AddFrame(m_removeButton);
  AddFrame(objectFrame, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
  TGHorizontal3DLine* nameObjectSeperator = new TGHorizontal3DLine(this, 200, 5);
  AddFrame(nameObjectSeperator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
  TGHorizontalFrame* colorSelectFrame = new TGHorizontalFrame(this, 200, 100);
  TGLabel* colorSelectLabel = new TGLabel(colorSelectFrame, "Color:");
  colorSelectFrame->AddFrame(colorSelectLabel, new TGLayoutHints(kLHintsNormal, 0, 50, 0, 0));
  TGString* graphicsLabel = new TGString(" ");
  Pixel_t selection = gVirtualX->GetPixel(kRed);
  std::vector<Pixel_t> colors;
  colors.push_back((Pixel_t)gVirtualX->GetPixel(kRed));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(kBlue));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(kYellow));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(kGreen));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(kCyan));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(kMagenta));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(kOrange));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kRed)));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kBlue)));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kYellow)));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kGreen)));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kCyan)));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kMagenta)));
  colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kOrange)));
  bool haveColor = false;
  for (std::vector<Pixel_t>::const_iterator iCol = colors.begin(); iCol != colors.end(); ++iCol) {
    if (*iCol == selection) haveColor = true;
  }
  if(!haveColor) {
    printf("Error: Color is not present in palette!\n");
    colors.push_back(selection);
  }
  m_colorSelectWidget = new FWColorSelect(colorSelectFrame, graphicsLabel, selection, colors, -1);
  m_colorSelectWidget->SetEnabled(kFALSE);
  colorSelectFrame->AddFrame(m_colorSelectWidget);
  AddFrame(colorSelectFrame);
  TGHorizontal3DLine* colorVisSeperator = new TGHorizontal3DLine(this, 200, 5);
  AddFrame(colorVisSeperator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
  m_isVisibleButton = new TGCheckButton(this, "Visible");
  m_isVisibleButton->SetState(kButtonDown, kFALSE);
  m_isVisibleButton->SetEnabled(kFALSE);
  AddFrame(m_isVisibleButton);
  SetWindowName("Model Inspector");
  Resize(GetDefaultSize());
  MapSubwindows();
  MapWindow();
  Layout();
}

// CmsShowModelPopup::CmsShowModelPopup(const CmsShowModelPopup& rhs)
// {
//    // do actual copying here;
// }

CmsShowModelPopup::~CmsShowModelPopup()
{
}

//
// assignment operators
//
// const CmsShowModelPopup& CmsShowModelPopup::operator=(const CmsShowModelPopup& rhs)
// {
//   //An exception safe implementation is
//   CmsShowModelPopup temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
CmsShowModelPopup::fillModelPopup(FWListModel* iModel) {
  if (iModel != m_model) {
    m_model = iModel;
    m_modelLabel->SetText(iModel->GetName());
    m_colorSelectWidget->SetColor(gVirtualX->GetPixel(iModel->GetMainColor()));
    m_isVisibleButton->SetDisabledAndSelected(m_model->GetRnrState());
    m_colorSelectWidget->SetEnabled(kTRUE);
    m_isVisibleButton->SetEnabled(kTRUE);
    m_removeButton->SetEnabled(kTRUE);
    if (!(m_colorSelectWidget->HasConnection("ColorSelected(Pixel_t)")))
      m_colorSelectWidget->Connect("ColorSelected(Pixel_t)", "CmsShowModelPopup", this, "changeModelColor(Pixel_t)");
    if (!(m_isVisibleButton->HasConnection("Toggled(Bool_T)")))
      m_isVisibleButton->Connect("Toggled(Bool_t)", "CmsShowModelPopup", this, "toggleModelVisible(Bool_t)");
    if (!(m_removeButton->HasConnection("Clicked()")))
      m_removeButton->Connect("Clicked()", "CmsShowModelPopup", this, "removeModel()");
    //    m_displayChangedConn = m_item->defaultDisplayPropertiesChanged_.connect(boost::bind(&CmsShowEDI::updateDisplay, this));
    //    m_modelChangedConn = m_item->changed_.connect(boost::bind(&CmsShowEDI::updateDisplay, this));
    //    m_selectionChangedConn = m_selectionManager->selectionChanged_.connect(boost::bind(&CmsShowEDI::updateSelection, this));
    //    m_destroyedConn = m_item->goingToBeDestroyed_.connect(boost::bind(&CmsShowEDI::disconnectAll, this));
    Layout();
 }
}

void
CmsShowModelPopup::removeModel() {
}

void
CmsShowModelPopup::updateDisplay() {
  if (m_model != 0) {
    m_colorSelectWidget->SetColor(gVirtualX->GetPixel(m_model->GetMainColor()));
    m_isVisibleButton->SetState(m_model->GetRnrState() ? kButtonDown : kButtonUp, kFALSE);
  }
}

void
CmsShowModelPopup::disconnectAll() {
  m_modelChangedConn.disconnect();
  m_destroyedConn.disconnect();
  m_colorSelectWidget->Disconnect("ColorSelected(Pixel_t)", this, "changeModelColor(Pixel_t)");
  m_isVisibleButton->Disconnect("Toggled(Bool_t)", this, "toggleModelVisible(Bool_t)");
  m_removeButton->Disconnect("Clicked()", this, "removeItem()");
  m_item = 0;
  m_model = 0;
  m_modelLabel->SetText(" ");
  m_colorSelectWidget->SetColor(gVirtualX->GetPixel(kRed));
  m_isVisibleButton->SetDisabledAndSelected(kTRUE);
  m_colorSelectWidget->SetEnabled(kFALSE);
  m_isVisibleButton->SetEnabled(kFALSE);
  m_removeButton->SetEnabled(kFALSE);
}

void
CmsShowModelPopup::changeModelColor(Pixel_t pixel) {
  std::cout<<"Changing color..."<<std::endl;
  Color_t color(TColor::GetColor(pixel));
  m_model->SetMainColor(color);
}

void
CmsShowModelPopup::toggleModelVisible(Bool_t on) {
  std::cout<<"Toggling visibility..."<<std::endl;
  m_model->SetRnrState(on);
}


//
// const member functions
//

//
// static member functions
//
