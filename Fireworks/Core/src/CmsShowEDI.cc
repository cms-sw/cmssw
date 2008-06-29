// -*- C++ -*-
//
// Package:     Core
// Class  :     CmsShowEDI
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Joshua Berger  
//         Created:  Mon Jun 23 15:48:11 EDT 2008
// $Id$
//

// system include files
#include <iostream>
#include <sigc++/sigc++.h>
#include <boost/bind.hpp>
#include "TClass.h"
#include "TGFrame.h"
#include "TGTab.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGString.h"
#include "TColor.h"
#include "TG3DLine.h"
#include "TGTextEntry.h"
#include "TGLayout.h"
#include "TGFont.h"
#include "TEveManager.h"

// user include files
#include "Fireworks/Core/interface/CmsShowEDI.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/src/FWListEventItem.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowEDI::CmsShowEDI(const TGWindow* p, UInt_t w, UInt_t h, FWSelectionManager* selMgr) : TGMainFrame(p, w, h)
{
  m_selectionManager = selMgr;
  SetCleanup(kDeepCleanup);
  TGHorizontalFrame* objectFrame = new TGHorizontalFrame(this);
  m_objectLabel = new TGLabel(objectFrame, " ");
  TGFont* defaultFont = gClient->GetFontPool()->GetFont(m_objectLabel->GetDefaultFontStruct());
  m_objectLabel->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
  m_objectLabel->SetTextJustify(kTextLeft);
  objectFrame->AddFrame(m_objectLabel, new TGLayoutHints(kLHintsExpandX));
  m_removeButton = new TGTextButton(objectFrame, "Remove", -1, TGTextButton::GetDefaultGC()(), TGTextButton::GetDefaultFontStruct(), kRaisedFrame|kDoubleBorder|kFixedWidth);
  m_removeButton->SetWidth(60);
  m_removeButton->SetEnabled(kFALSE);
  objectFrame->AddFrame(m_removeButton);
  AddFrame(objectFrame, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
  TGTab* ediTabs = new TGTab(this, GetWidth(), GetHeight());
  TGVerticalFrame* graphicsFrame = new TGVerticalFrame(ediTabs, 200, 400);
  TGHorizontalFrame* colorSelectFrame = new TGHorizontalFrame(graphicsFrame, 200, 100);
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
  graphicsFrame->AddFrame(colorSelectFrame);
  TGHorizontal3DLine* colorVisSeperator = new TGHorizontal3DLine(graphicsFrame, 200, 5);
  graphicsFrame->AddFrame(colorVisSeperator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
  m_isVisibleButton = new TGCheckButton(graphicsFrame, "Visible");
  m_isVisibleButton->SetState(kButtonDown, kFALSE);
  m_isVisibleButton->SetEnabled(kFALSE);
  graphicsFrame->AddFrame(m_isVisibleButton);
  ediTabs->AddTab("Graphics", graphicsFrame);
  
  // Filter tab
  TGVerticalFrame* filterFrame = new TGVerticalFrame(ediTabs, 200, 600);
  TGLabel* filterExpressionLabel = new TGLabel(filterFrame, "Expression:");
  filterFrame->AddFrame(filterExpressionLabel);
  m_filterExpressionEntry = new TGTextEntry(filterFrame);
  m_filterExpressionEntry->SetEnabled(kFALSE);
  filterFrame->AddFrame(m_filterExpressionEntry);
  m_filterButton = new TGTextButton(filterFrame, "Filter");
  m_filterButton->SetEnabled(kFALSE);
  filterFrame->AddFrame(m_filterButton);
  ediTabs->AddTab("Filter", filterFrame);

  // Select tab
  TGVerticalFrame* selectFrame = new TGVerticalFrame(ediTabs, 200, 600);
  TGLabel* expressionLabel = new TGLabel(selectFrame, "Expression:");
  selectFrame->AddFrame(expressionLabel);
  m_selectExpressionEntry = new TGTextEntry(selectFrame);
  m_selectExpressionEntry->SetEnabled(kFALSE);
  selectFrame->AddFrame(m_selectExpressionEntry);
  m_selectButton = new TGTextButton(selectFrame, "Select");
  m_selectButton->SetEnabled(kFALSE);
  selectFrame->AddFrame(m_selectButton);  
  TGHorizontal3DLine* selectSeperator1 = new TGHorizontal3DLine(selectFrame, 200, 5);
  selectFrame->AddFrame(selectSeperator1, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
  m_selectAllButton = new TGTextButton(selectFrame, "Select All");
  m_selectAllButton->SetEnabled(kFALSE);
  selectFrame->AddFrame(m_selectAllButton);
  ediTabs->AddTab("Select", selectFrame);
  
  // Data tab
  TGVerticalFrame* dataFrame = new TGVerticalFrame(ediTabs, 200, 600);
  TGLabel* nameLabel = new TGLabel(dataFrame, "Name:");
  dataFrame->AddFrame(nameLabel);
  m_nameEntry = new TGTextEntry(dataFrame);
  m_nameEntry->SetEnabled(kFALSE);
  dataFrame->AddFrame(m_nameEntry);
  TGHorizontal3DLine* dataSeperator = new TGHorizontal3DLine(dataFrame, 200, 5);
  dataFrame->AddFrame(dataSeperator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
  TGLabel* labelsLabel = new TGLabel(dataFrame, "Labels:");
  dataFrame->AddFrame(labelsLabel);
  UInt_t textWidth = (UInt_t)(0.4 * dataFrame->GetWidth());
  TGHorizontalFrame* typeFrame = new TGHorizontalFrame(dataFrame);
  TGLabel* typeLabel = new TGLabel(typeFrame, "Type: ", TGLabel::GetDefaultGC()(), TGLabel::GetDefaultFontStruct(), kFixedWidth);
  typeLabel->SetWidth(textWidth);
  typeLabel->SetTextJustify(kTextLeft);
  typeFrame->AddFrame(typeLabel, new TGLayoutHints(kLHintsNormal, 2, 0, 0, 0));
  m_typeEntry = new TGTextEntry(typeFrame);
  m_typeEntry->SetEnabled(kFALSE);
  typeFrame->AddFrame(m_typeEntry, new TGLayoutHints(kLHintsExpandX, 0, 2, 0, 0));
  dataFrame->AddFrame(typeFrame, new TGLayoutHints(kLHintsExpandX));
  TGHorizontalFrame* moduleFrame = new TGHorizontalFrame(dataFrame);
  TGLabel* moduleLabel = new TGLabel(moduleFrame, "Module: ", TGLabel::GetDefaultGC()(), TGLabel::GetDefaultFontStruct(), kFixedWidth);
  moduleLabel->SetWidth(textWidth);
  moduleLabel->SetTextJustify(kTextLeft);
  moduleFrame->AddFrame(moduleLabel, new TGLayoutHints(kLHintsNormal, 2, 0, 0, 0));
  m_moduleEntry = new TGTextEntry(moduleFrame);
  m_moduleEntry->SetEnabled(kFALSE);
  moduleFrame->AddFrame(m_moduleEntry, new TGLayoutHints(kLHintsExpandX, 0, 2, 0, 0));
  dataFrame->AddFrame(moduleFrame, new TGLayoutHints(kLHintsExpandX));
  TGHorizontalFrame* instanceFrame = new TGHorizontalFrame(dataFrame);
  TGLabel* instanceLabel = new TGLabel(instanceFrame, "Instance: ", TGLabel::GetDefaultGC()(), TGLabel::GetDefaultFontStruct(), kFixedWidth);
  instanceLabel->SetWidth(textWidth);
  instanceLabel->SetTextJustify(kTextLeft);
  instanceFrame->AddFrame(instanceLabel, new TGLayoutHints(kLHintsNormal, 2, 0, 0, 0));
  m_instanceEntry = new TGTextEntry(instanceFrame);
  //  m_instanceEntry->SetWidth(boxWidth);
  m_instanceEntry->SetEnabled(kFALSE);
  instanceFrame->AddFrame(m_instanceEntry, new TGLayoutHints(kLHintsExpandX, 0, 2, 0, 0));
  dataFrame->AddFrame(instanceFrame, new TGLayoutHints(kLHintsExpandX));
  TGHorizontalFrame* processFrame = new TGHorizontalFrame(dataFrame);
  TGLabel* processLabel = new TGLabel(processFrame, "Process: ", TGLabel::GetDefaultGC()(), TGLabel::GetDefaultFontStruct(), kFixedWidth);
  processLabel->SetWidth(textWidth);
  processLabel->SetTextJustify(kTextLeft);
  processFrame->AddFrame(processLabel, new TGLayoutHints(kLHintsNormal, 2, 0, 0, 0));
  m_processEntry = new TGTextEntry(processFrame);
  //  m_processEntry->SetWidth(boxWidth);
  m_processEntry->SetEnabled(kFALSE);
  processFrame->AddFrame(m_processEntry, new TGLayoutHints(kLHintsExpandX, 0, 2, 0, 0));
  dataFrame->AddFrame(processFrame, new TGLayoutHints(kLHintsExpandX));
  ediTabs->AddTab("Data", dataFrame);
  AddFrame(ediTabs, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 0, 0, 0, 0));
  SetWindowName("Event Display Inspector");
  Resize(GetDefaultSize());
  MapSubwindows();
  MapWindow();
  Layout();
}

// CmsShowEDI::CmsShowEDI(const CmsShowEDI& rhs)
// {
//    // do actual copying here;
// }

CmsShowEDI::~CmsShowEDI()
{
  //  delete m_objectLabel;
  //  delete m_colorSelectWidget;
  //  delete m_isVisibleButton;
}

//
// assignment operators
//
// const CmsShowEDI& CmsShowEDI::operator=(const CmsShowEDI& rhs)
// {
//   //An exception safe implementation is
//   CmsShowEDI temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
CmsShowEDI::fillEDIFrame(FWEventItem* iItem) {
  if (iItem != m_item) {
    m_item = iItem;
    m_objectLabel->SetText(iItem->name().c_str());
    m_colorSelectWidget->SetColor(gVirtualX->GetPixel(iItem->defaultDisplayProperties().color()));
    m_isVisibleButton->SetDisabledAndSelected(iItem->defaultDisplayProperties().isVisible());
    m_filterExpressionEntry->SetText(iItem->filterExpression().c_str());
    m_nameEntry->SetText(iItem->name().c_str());
    m_typeEntry->SetText(iItem->type()->GetName());
    m_moduleEntry->SetText(iItem->moduleLabel().c_str());
    m_instanceEntry->SetText(iItem->productInstanceLabel().c_str());
    m_processEntry->SetText(iItem->processName().c_str());
    //  else m_isVisibleButton->SetState(kButtonDown, kFALSE);
    m_colorSelectWidget->SetEnabled(kTRUE);
    m_isVisibleButton->SetEnabled(kTRUE);
    m_filterExpressionEntry->SetEnabled(kTRUE);
    m_selectExpressionEntry->SetEnabled(kTRUE);
    m_filterButton->SetEnabled(kTRUE);
    m_selectButton->SetEnabled(kTRUE);
    m_selectAllButton->SetEnabled(kTRUE);
    m_removeButton->SetEnabled(kTRUE);
    if (!(m_colorSelectWidget->HasConnection("ColorSelected(Pixel_t)")))
      m_colorSelectWidget->Connect("ColorSelected(Pixel_t)", "CmsShowEDI", this, "changeItemColor(Pixel_t)");
    if (!(m_isVisibleButton->HasConnection("Toggled(Bool_T)")))
      m_isVisibleButton->Connect("Toggled(Bool_t)", "CmsShowEDI", this, "toggleItemVisible(Bool_t)");
    if (!(m_filterExpressionEntry->HasConnection("ReturnPressed()")))
      m_filterExpressionEntry->Connect("ReturnPressed()", "CmsShowEDI", this, "runFilter()");
    if (!(m_filterButton->HasConnection("Clicked()")))
      m_filterButton->Connect("Clicked()", "CmsShowEDI", this, "runFilter()");
    if (!(m_selectExpressionEntry->HasConnection("ReturnPressed()")))
      m_selectExpressionEntry->Connect("ReturnPressed()", "CmsShowEDI", this, "runSelection()");
    if (!(m_selectButton->HasConnection("Clicked()")))
      m_selectButton->Connect("Clicked()", "CmsShowEDI", this, "runSelection()");
    if (!(m_removeButton->HasConnection("Clicked()")))
      m_removeButton->Connect("Clicked()", "CmsShowEDI", this, "removeItem()");
    if (!(m_selectAllButton->HasConnection("Clicked()")))
      m_selectAllButton->Connect("Clicked()", "CmsShowEDI", this, "selectAll()");
    m_displayChangedConn = m_item->defaultDisplayPropertiesChanged_.connect(boost::bind(&CmsShowEDI::updateDisplay, this));
    m_modelChangedConn = m_item->changed_.connect(boost::bind(&CmsShowEDI::updateFilter, this));
    //    m_selectionChangedConn = m_selectionManager->selectionChanged_.connect(boost::bind(&CmsShowEDI::updateSelection, this));
    m_destroyedConn = m_item->goingToBeDestroyed_.connect(boost::bind(&CmsShowEDI::disconnectAll, this));
    Layout();
  }
}

void
CmsShowEDI::removeItem() {
  m_item->destroy();
  delete m_item;
  m_item = 0;
  gEve->EditElement(0);
  gEve->Redraw3D();
}

void
CmsShowEDI::updateDisplay() {
  std::cout<<"Updating display"<<std::endl;
  m_colorSelectWidget->SetColor(gVirtualX->GetPixel(m_item->defaultDisplayProperties().color()));
  m_isVisibleButton->SetState(m_item->defaultDisplayProperties().isVisible() ? kButtonDown : kButtonUp, kFALSE);
}

void
CmsShowEDI::updateFilter() {
  m_filterExpressionEntry->SetText(m_item->filterExpression().c_str());
}

void
CmsShowEDI::disconnectAll() {
  m_displayChangedConn.disconnect();
  m_modelChangedConn.disconnect();
  m_destroyedConn.disconnect();
  m_colorSelectWidget->Disconnect("ColorSelected(Pixel_t)", this, "changeItemColor(Pixel_t)");
  m_isVisibleButton->Disconnect("Toggled(Bool_t)", this, "toggleItemVisible(Bool_t)");
  m_filterExpressionEntry->Disconnect("ReturnPressed()", this, "runFilter()");
  m_selectExpressionEntry->Disconnect("ReturnPressed()", this, "runSelection()");
  m_filterButton->Disconnect("Clicked()", this, "runFilter()");
  m_selectButton->Disconnect("Clicked()", this, "runSelection()");
  m_selectAllButton->Disconnect("Clicked()", this, "selectAll()");
  m_removeButton->Disconnect("Clicked()", this, "removeItem()");
  m_item = 0;
  m_objectLabel->SetText(" ");
  m_colorSelectWidget->SetColor(gVirtualX->GetPixel(kRed));
  m_isVisibleButton->SetDisabledAndSelected(kTRUE);
  m_filterExpressionEntry->SetText(0);
  m_selectExpressionEntry->SetText(0);
  m_nameEntry->SetText(0);
  m_typeEntry->SetText(0);
  m_moduleEntry->SetText(0);
  m_instanceEntry->SetText(0);
  m_processEntry->SetText(0);
  //  else m_isVisibleButton->SetState(kButtonDown, kFALSE);                                                                                               
  m_colorSelectWidget->SetEnabled(kFALSE);
  m_isVisibleButton->SetEnabled(kFALSE);
  m_filterExpressionEntry->SetEnabled(kFALSE);
  m_filterButton->SetEnabled(kFALSE);
  m_selectExpressionEntry->SetEnabled(kFALSE);
  m_selectButton->SetEnabled(kFALSE);
  m_selectAllButton->SetEnabled(kFALSE);
  m_removeButton->SetEnabled(kFALSE);
}
      
void
CmsShowEDI::changeItemColor(Pixel_t pixel) {
  Color_t color(TColor::GetColor(pixel));
  const FWDisplayProperties changeProperties(color, m_item->defaultDisplayProperties().isVisible());
  m_item->setDefaultDisplayProperties(changeProperties);
}

void
CmsShowEDI::toggleItemVisible(Bool_t on) {
  const FWDisplayProperties changeProperties(m_item->defaultDisplayProperties().color(), on);
  m_item->setDefaultDisplayProperties(changeProperties);
}

void
CmsShowEDI::runFilter() {
  const std::string filter(m_filterExpressionEntry->GetText());
  if (m_item != 0) m_item->setFilterExpression(filter);
}

void
CmsShowEDI::runSelection() {
  FWModelExpressionSelector selector;
  const std::string selection(m_selectExpressionEntry->GetText());
  if (m_item != 0) selector.select(m_item, selection);
}

void
CmsShowEDI::selectAll() {
  FWChangeSentry sentry(*(m_item->changeManager()));
  for (int i = 0; i < static_cast<int>(m_item->size()); i++) {
    m_item->select(i);
  }
}  
//
// const member functions
//

//
// static member functions
//
