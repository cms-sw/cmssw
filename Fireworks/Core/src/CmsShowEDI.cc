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
// $Id: CmsShowEDI.cc,v 1.26 2009/06/29 19:16:28 amraktad Exp $
//

// system include files
#include <iostream>
#include <sstream>
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
#include "TGTextView.h"
#include "TGLayout.h"
#include "TGFont.h"
#include "TEveManager.h"

#include "TGMsgBox.h"
#include "TGComboBox.h"

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
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWExpressionException.h"
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "Fireworks/Core/src/FWExpressionValidator.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowEDI::CmsShowEDI(const TGWindow* p, UInt_t w, UInt_t h, FWSelectionManager* selMgr, FWColorManager* colorMgr) :
   TGTransientFrame(gClient->GetDefaultRoot(),p, w, h),
   m_item(0),
   m_validator( new FWExpressionValidator),
   m_colorManager(colorMgr)
{
   m_selectionManager = selMgr;
   SetCleanup(kDeepCleanup);

   m_selectionManager->itemSelectionChanged_.connect(boost::bind(&CmsShowEDI::fillEDIFrame,this));
   TGHorizontalFrame* objectFrame = new TGHorizontalFrame(this);
   m_objectLabel = new TGLabel(objectFrame, " ");
   TGFont* defaultFont = gClient->GetFontPool()->GetFont(m_objectLabel->GetDefaultFontStruct());
   m_objectLabel->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
   m_objectLabel->SetTextJustify(kTextLeft);
   objectFrame->AddFrame(m_objectLabel, new TGLayoutHints(kLHintsExpandX));
   AddFrame(objectFrame, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
   m_tabs = new TGTab(this, GetWidth(), GetHeight());
   TGVerticalFrame* graphicsFrame = new TGVerticalFrame(m_tabs, 200, 400);
   TGHorizontalFrame* colorSelectFrame = new TGHorizontalFrame(graphicsFrame, 200, 100);
   TGLabel* colorSelectLabel = new TGLabel(colorSelectFrame, "Color:");
   colorSelectFrame->AddFrame(colorSelectLabel, new TGLayoutHints(kLHintsNormal, 0, 50, 0, 0));
   const char* graphicsLabel = " ";
   std::vector<Color_t> colors;
   m_colorSelectWidget = new FWColorSelect(colorSelectFrame, graphicsLabel, 0, colorMgr, -1);
   m_colorSelectWidget->SetEnabled(kFALSE);
   colorSelectFrame->AddFrame(m_colorSelectWidget);
   graphicsFrame->AddFrame(colorSelectFrame);
   TGHorizontal3DLine* colorVisSeperator = new TGHorizontal3DLine(graphicsFrame, 200, 5);
   graphicsFrame->AddFrame(colorVisSeperator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   m_isVisibleButton = new TGCheckButton(graphicsFrame, "Visible");
   m_isVisibleButton->SetState(kButtonDown, kFALSE);
   m_isVisibleButton->SetEnabled(kFALSE);
   graphicsFrame->AddFrame(m_isVisibleButton);
   TGHorizontal3DLine* separator = new TGHorizontal3DLine(graphicsFrame, 200, 5);
   graphicsFrame->AddFrame(separator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   TGLabel* orderLabel = new TGLabel(graphicsFrame, "Set Drawing Order");
   graphicsFrame->AddFrame(orderLabel);
   m_frontButton = new TGTextButton(graphicsFrame,"Move Projected to Front");
   m_frontButton->SetEnabled(kFALSE);
   graphicsFrame->AddFrame(m_frontButton);
   m_backButton = new TGTextButton(graphicsFrame,"Move Projected to Back");
   m_backButton->SetEnabled(kFALSE);
   graphicsFrame->AddFrame(m_backButton);
   m_tabs->AddTab("Graphics", graphicsFrame);

   // Filter tab
   TGVerticalFrame* filterFrame = new TGVerticalFrame(m_tabs, 200, 600);
   TGLabel* filterExpressionLabel = new TGLabel(filterFrame, "Expression:");
   filterFrame->AddFrame(filterExpressionLabel);
   m_filterExpressionEntry = new FWGUIValidatingTextEntry(filterFrame);
   m_filterExpressionEntry->setValidator(m_validator);
   m_filterExpressionEntry->SetEnabled(kFALSE);
   filterFrame->AddFrame(m_filterExpressionEntry, new TGLayoutHints(kLHintsExpandX));
   m_filterButton = new TGTextButton(filterFrame, "Filter");
   m_filterButton->SetEnabled(kFALSE);
   filterFrame->AddFrame(m_filterButton);
   m_filterError = new TGTextView(filterFrame);
   m_filterError->SetForegroundColor(gVirtualX->GetPixel(kRed));
   m_filterError->SetBackgroundColor(TGFrame::GetDefaultFrameBackground());
   m_filterError->ChangeOptions(0);
   filterFrame->AddFrame(m_filterError, new TGLayoutHints(kLHintsExpandX| kLHintsExpandY));
   //taken from TGComboBox.cxx
   m_tabs->AddTab("Filter", filterFrame);

   // Select tab
   TGVerticalFrame* selectFrame = new TGVerticalFrame(m_tabs, 200, 600);
   TGLabel* expressionLabel = new TGLabel(selectFrame, "Expression:");
   selectFrame->AddFrame(expressionLabel);
   m_selectExpressionEntry = new FWGUIValidatingTextEntry(selectFrame);
   m_selectExpressionEntry->setValidator(m_validator);
   m_selectExpressionEntry->SetEnabled(kFALSE);
   selectFrame->AddFrame(m_selectExpressionEntry,new TGLayoutHints(kLHintsExpandX));
   m_selectButton = new TGTextButton(selectFrame, "Select");
   m_selectButton->SetEnabled(kFALSE);
   selectFrame->AddFrame(m_selectButton);
   TGHorizontal3DLine* selectSeperator1 = new TGHorizontal3DLine(selectFrame, 200, 5);
   selectFrame->AddFrame(selectSeperator1, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   m_selectAllButton = new TGTextButton(selectFrame, "Select All");
   m_selectAllButton->SetEnabled(kFALSE);
   selectFrame->AddFrame(m_selectAllButton);
   m_selectError = new TGTextView(selectFrame);
   m_selectError->SetForegroundColor(gVirtualX->GetPixel(kRed));
   m_selectError->SetBackgroundColor(TGFrame::GetDefaultFrameBackground());
   m_selectError->ChangeOptions(0);
   selectFrame->AddFrame(m_selectError, new TGLayoutHints(kLHintsExpandX| kLHintsExpandY));
   m_tabs->AddTab("Select", selectFrame);

   // Data tab
   TGVerticalFrame* dataFrame = new TGVerticalFrame(m_tabs, 200, 600);
   TGLabel* nameLabel = new TGLabel(dataFrame, "Name:");
   dataFrame->AddFrame(nameLabel);
   m_nameEntry = new TGTextEntry(dataFrame);
   m_nameEntry->SetEnabled(kFALSE);
   dataFrame->AddFrame(m_nameEntry,new TGLayoutHints(kLHintsExpandX));
   TGHorizontal3DLine* dataSeperator = new TGHorizontal3DLine(dataFrame, 200, 5);
   dataFrame->AddFrame(dataSeperator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   TGLabel* labelsLabel = new TGLabel(dataFrame, "Labels:");
   dataFrame->AddFrame(labelsLabel);
   UInt_t textWidth = (UInt_t)(0.4 * dataFrame->GetWidth());
   TGHorizontalFrame* typeFrame = new TGHorizontalFrame(dataFrame);
   TGLabel* typeLabel = new TGLabel(typeFrame, "Type: ", TGLabel::GetDefaultGC() (), TGLabel::GetDefaultFontStruct(), kFixedWidth);
   typeLabel->SetWidth(textWidth);
   typeLabel->SetTextJustify(kTextLeft);
   typeFrame->AddFrame(typeLabel, new TGLayoutHints(kLHintsNormal, 2, 0, 0, 0));
   m_typeEntry = new TGTextEntry(typeFrame);
   m_typeEntry->SetEnabled(kFALSE);
   typeFrame->AddFrame(m_typeEntry, new TGLayoutHints(kLHintsExpandX, 0, 2, 0, 0));
   dataFrame->AddFrame(typeFrame, new TGLayoutHints(kLHintsExpandX));
   TGHorizontalFrame* moduleFrame = new TGHorizontalFrame(dataFrame);
   TGLabel* moduleLabel = new TGLabel(moduleFrame, "Module: ", TGLabel::GetDefaultGC() (), TGLabel::GetDefaultFontStruct(), kFixedWidth);
   moduleLabel->SetWidth(textWidth);
   moduleLabel->SetTextJustify(kTextLeft);
   moduleFrame->AddFrame(moduleLabel, new TGLayoutHints(kLHintsNormal, 2, 0, 0, 0));
   m_moduleEntry = new TGTextEntry(moduleFrame);
   m_moduleEntry->SetEnabled(kFALSE);
   moduleFrame->AddFrame(m_moduleEntry, new TGLayoutHints(kLHintsExpandX, 0, 2, 0, 0));
   dataFrame->AddFrame(moduleFrame, new TGLayoutHints(kLHintsExpandX));
   TGHorizontalFrame* instanceFrame = new TGHorizontalFrame(dataFrame);
   TGLabel* instanceLabel = new TGLabel(instanceFrame, "Instance: ", TGLabel::GetDefaultGC() (), TGLabel::GetDefaultFontStruct(), kFixedWidth);
   instanceLabel->SetWidth(textWidth);
   instanceLabel->SetTextJustify(kTextLeft);
   instanceFrame->AddFrame(instanceLabel, new TGLayoutHints(kLHintsNormal, 2, 0, 0, 0));
   m_instanceEntry = new TGTextEntry(instanceFrame);
   //  m_instanceEntry->SetWidth(boxWidth);
   m_instanceEntry->SetEnabled(kFALSE);
   instanceFrame->AddFrame(m_instanceEntry, new TGLayoutHints(kLHintsExpandX, 0, 2, 0, 0));
   dataFrame->AddFrame(instanceFrame, new TGLayoutHints(kLHintsExpandX));
   TGHorizontalFrame* processFrame = new TGHorizontalFrame(dataFrame);
   TGLabel* processLabel = new TGLabel(processFrame, "Process: ", TGLabel::GetDefaultGC() (), TGLabel::GetDefaultFontStruct(), kFixedWidth);
   processLabel->SetWidth(textWidth);
   processLabel->SetTextJustify(kTextLeft);
   processFrame->AddFrame(processLabel, new TGLayoutHints(kLHintsNormal, 2, 0, 0, 0));
   m_processEntry = new TGTextEntry(processFrame);
   //  m_processEntry->SetWidth(boxWidth);
   m_processEntry->SetEnabled(kFALSE);
   processFrame->AddFrame(m_processEntry, new TGLayoutHints(kLHintsExpandX, 0, 2, 0, 0));
   dataFrame->AddFrame(processFrame, new TGLayoutHints(kLHintsExpandX));
   dataSeperator = new TGHorizontal3DLine(dataFrame, 200, 5);
   dataFrame->AddFrame(dataSeperator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   m_removeButton = new TGTextButton(dataFrame, "Remove Collection");
   m_removeButton->SetEnabled(kFALSE);
   dataFrame->AddFrame(m_removeButton);

   m_tabs->AddTab("Data", dataFrame);
   AddFrame(m_tabs, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 0, 0, 0, 0));

   m_colorSelectWidget->Connect("ColorChosen(Color_t)", "CmsShowEDI", this, "changeItemColor(Color_t)");
   m_isVisibleButton->Connect("Toggled(Bool_t)", "CmsShowEDI", this, "toggleItemVisible(Bool_t)");
   m_filterExpressionEntry->Connect("ReturnPressed()", "CmsShowEDI", this, "runFilter()");
   m_filterButton->Connect("Clicked()", "CmsShowEDI", this, "runFilter()");
   m_selectExpressionEntry->Connect("ReturnPressed()", "CmsShowEDI", this, "runSelection()");
   m_selectButton->Connect("Clicked()", "CmsShowEDI", this, "runSelection()");
   m_removeButton->Connect("Clicked()", "CmsShowEDI", this, "removeItem()");
   m_selectAllButton->Connect("Clicked()", "CmsShowEDI", this, "selectAll()");
   m_frontButton->Connect("Clicked()","CmsShowEDI",this,"moveToFront()");
   m_backButton->Connect("Clicked()","CmsShowEDI",this,"moveToBack()");


   SetWindowName("Collection Controller");
   Resize(GetDefaultSize());
   MapSubwindows();
   Layout();
   
   fillEDIFrame();
}

// CmsShowEDI::CmsShowEDI(const CmsShowEDI& rhs)
// {
//    // do actual copying here;
// }

CmsShowEDI::~CmsShowEDI()
{
   disconnectAll();
   m_colorSelectWidget->Disconnect("ColorSelected(Pixel_t)", this, "changeItemColor(Pixel_t)");
   m_isVisibleButton->Disconnect("Toggled(Bool_t)", this, "toggleItemVisible(Bool_t)");
   m_filterExpressionEntry->Disconnect("ReturnPressed()", this, "runFilter()");
   m_selectExpressionEntry->Disconnect("ReturnPressed()", this, "runSelection()");
   m_filterButton->Disconnect("Clicked()", this, "runFilter()");
   m_selectButton->Disconnect("Clicked()", this, "runSelection()");
   m_selectAllButton->Disconnect("Clicked()", this, "selectAll()");
   m_removeButton->Disconnect("Clicked()", this, "removeItem()");
   m_frontButton->Disconnect("Clicked()",this,"moveToFront()");
   m_backButton->Disconnect("Clicked()",this,"moveToBack()");

   //  delete m_objectLabel;
   //  delete m_colorSelectWidget;
   //  delete m_isVisibleButton;
   delete m_validator;
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
CmsShowEDI::fillEDIFrame() {
   FWEventItem* iItem =0;
   bool multipleCollections = false;
   if(!m_selectionManager->selectedItems().empty()) {
      if(m_selectionManager->selectedItems().size()==1) {
         iItem=*(m_selectionManager->selectedItems().begin());
      } else {
         multipleCollections=true;
      }
   }
   //m_item can be zero because we had 0 or many collections selected
   if (0 == m_item || iItem != m_item) {
      disconnectAll();
      m_item = iItem;
      if(0 != m_item) {
         m_objectLabel->SetText(iItem->name().c_str());
         m_colorSelectWidget->SetColorByIndex(m_colorManager->colorToIndex(iItem->defaultDisplayProperties().color()),kFALSE);
         m_isVisibleButton->SetDisabledAndSelected(iItem->defaultDisplayProperties().isVisible());
         m_validator->setType(ROOT::Reflex::Type::ByTypeInfo(*(iItem->modelType()->GetTypeInfo())));
         m_filterExpressionEntry->SetText(iItem->filterExpression().c_str());
         m_filterError->Clear();
         m_selectError->Clear();
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
         if(!m_item->isInFront()) {m_frontButton->SetEnabled(kTRUE);}
         if(!m_item->isInBack()) {m_backButton->SetEnabled(kTRUE);}
         m_displayChangedConn = m_item->defaultDisplayPropertiesChanged_.connect(boost::bind(&CmsShowEDI::updateDisplay, this));
         m_modelChangedConn = m_item->changed_.connect(boost::bind(&CmsShowEDI::updateFilter, this));
         //    m_selectionChangedConn = m_selectionManager->selectionChanged_.connect(boost::bind(&CmsShowEDI::updateSelection, this));
         m_destroyedConn = m_item->goingToBeDestroyed_.connect(boost::bind(&CmsShowEDI::disconnectAll, this));
      } else if(multipleCollections) {
         std::ostringstream s;
         s<<m_selectionManager->selectedItems().size()<<" Collections Selected";
         m_objectLabel->SetText(s.str().c_str());
      }
   
      Layout();
   }
}

void
CmsShowEDI::removeItem() {
   Int_t chosen=0;
   std::string message("This action will remove the ");
   message += m_item->name();
   message +=" collection from the display."
              "\nIf you wish to return the collection you would have to use the 'Add Collection' window.";
   new TGMsgBox(gClient->GetDefaultRoot(),
                this,
                "Remove Collection Confirmation",
                message.c_str(),
                kMBIconExclamation,
                kMBCancel | kMBApply,
                &chosen);
   if(kMBApply == chosen) {
      m_item->destroy();
      m_item = 0;
      //make sure the ROOT global editor does not try to use this
      gEve->EditElement(0);
      gEve->Redraw3D();
   }
}

void
CmsShowEDI::updateDisplay() {
   //std::cout<<"Updating display"<<std::endl;
   m_colorSelectWidget->SetColorByIndex(m_colorManager->colorToIndex(m_item->defaultDisplayProperties().color()),kFALSE);
   m_isVisibleButton->SetState(m_item->defaultDisplayProperties().isVisible() ? kButtonDown : kButtonUp, kFALSE);
}

void
CmsShowEDI::moveToBack()
{
   m_item->moveToBack();
   m_backButton->SetEnabled(kFALSE);
   if(!m_item->isInFront()) {
      m_frontButton->SetEnabled(kTRUE);
   }
}
void
CmsShowEDI::moveToFront()
{
   m_item->moveToFront();
   m_frontButton->SetEnabled(kFALSE);
   if(!m_item->isInBack()) {
      m_backButton->SetEnabled(kTRUE);
   }
}


void
CmsShowEDI::updateFilter() {
   m_filterExpressionEntry->SetText(m_item->filterExpression().c_str());
}

void
CmsShowEDI::disconnectAll() {
   m_objectLabel->SetText("No Collection Selected");
   if(0 != m_item) {
      m_displayChangedConn.disconnect();
      m_modelChangedConn.disconnect();
      m_destroyedConn.disconnect();
      m_item = 0;
      m_colorSelectWidget->SetColorByIndex(0,kFALSE);
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
}

void
CmsShowEDI::changeItemColor(Color_t color) {
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
   if (m_item != 0) {
      try {
         m_filterError->Clear();
         m_item->setFilterExpression(filter);
      } catch( const FWExpressionException& e) {
         m_filterError->AddLine(e.what().c_str());
         m_filterError->Update();
         if(e.column() > -1) {
            m_filterExpressionEntry->SetCursorPosition(e.column());
         }
      }
   }
}



void
CmsShowEDI::runSelection() {
   FWModelExpressionSelector selector;
   const std::string selection(m_selectExpressionEntry->GetText());
   if (m_item != 0){
      try {
         m_selectError->Clear();
         m_item->selectionManager()->clearSelection();

         selector.select(m_item, selection);
      } catch( const FWExpressionException& e) {
         m_selectError->AddLine(e.what().c_str());
         m_selectError->Update();
         if(e.column() > -1) {
            m_selectExpressionEntry->SetCursorPosition(e.column());
         }
      }
   }
}

void
CmsShowEDI::selectAll() {
   FWChangeSentry sentry(*(m_item->changeManager()));
   for (int i = 0; i < static_cast<int>(m_item->size()); i++) {
      m_item->select(i);
   }
}

void 
CmsShowEDI::show(FWDataCategories iToView)
{
   m_tabs->SetTab(iToView);
}

//
// const member functions
//

//
// static member functions
//
