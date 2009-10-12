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
// $Id: CmsShowModelPopup.cc,v 1.19 2009/05/27 15:40:12 chrjones Exp $
//

// system include file
#include <iostream>
#include <sstream>
#include <set>
#include <sigc++/sigc++.h>
#include <boost/bind.hpp>
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
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWModelId.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWDetailViewManager.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowModelPopup::CmsShowModelPopup(FWDetailViewManager* iManager,
                                     FWSelectionManager* iSelMgr,
                                     const FWColorManager* iColorMgr,
                                     const TGWindow* p, UInt_t w, UInt_t h) :
   TGTransientFrame(gClient->GetDefaultRoot(),p,w,h),
   m_detailViewManager(iManager),
   m_colorManager(iColorMgr)
{
   m_changes = iSelMgr->selectionChanged_.connect(boost::bind(&CmsShowModelPopup::fillModelPopup, this, _1));

   SetCleanup(kDeepCleanup);
   TGHorizontalFrame* objectFrame = new TGHorizontalFrame(this);
   m_modelLabel = new TGLabel(objectFrame, " ");
   TGFont* defaultFont = gClient->GetFontPool()->GetFont(m_modelLabel->GetDefaultFontStruct());
   m_modelLabel->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily, 14, defaultFont->GetFontAttributes().fWeight + 2, defaultFont->GetFontAttributes().fSlant));
   m_modelLabel->SetTextJustify(kTextLeft);
   objectFrame->AddFrame(m_modelLabel, new TGLayoutHints(kLHintsExpandX));
   AddFrame(objectFrame, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
   TGHorizontal3DLine* nameObjectSeperator = new TGHorizontal3DLine(this, 200, 5);
   AddFrame(nameObjectSeperator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   TGHorizontalFrame* colorSelectFrame = new TGHorizontalFrame(this, 200, 100);
   TGLabel* colorSelectLabel = new TGLabel(colorSelectFrame, "Color:");
   colorSelectFrame->AddFrame(colorSelectLabel, new TGLayoutHints(kLHintsNormal, 0, 50, 0, 0));
   const char* graphicsLabel = " ";
   m_colorSelectWidget = new FWColorSelect(colorSelectFrame, graphicsLabel, 0, iColorMgr, -1);
   m_colorSelectWidget->SetEnabled(kFALSE);
   colorSelectFrame->AddFrame(m_colorSelectWidget);
   AddFrame(colorSelectFrame);
   TGHorizontal3DLine* colorVisSeperator = new TGHorizontal3DLine(this, 200, 5);
   AddFrame(colorVisSeperator, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   m_isVisibleButton = new TGCheckButton(this, "Visible");
   m_isVisibleButton->SetState(kButtonDown, kFALSE);
   m_isVisibleButton->SetEnabled(kFALSE);
   AddFrame(m_isVisibleButton);
   AddFrame(new TGHorizontal3DLine(this, 200, 5), new TGLayoutHints(kLHintsNormal, 0, 0, 5, 5));
   m_openDetailedViewButton = new TGTextButton(this,"Open Detailed View");
   m_openDetailedViewButton->SetEnabled(kFALSE);
   AddFrame(m_openDetailedViewButton);
   m_openDetailedViewButton->Connect("Clicked()","CmsShowModelPopup", this, "openDetailedView()");

   m_colorSelectWidget->Connect("ColorChosen(Color_t)", "CmsShowModelPopup", this, "changeModelColor(Color_t)");
   m_isVisibleButton->Connect("Toggled(Bool_t)", "CmsShowModelPopup", this, "toggleModelVisible(Bool_t)");


   SetWindowName("Object Display Controller");
   Resize(GetDefaultSize());
   MapSubwindows();
   Layout();

   fillModelPopup(*iSelMgr);
}

// CmsShowModelPopup::CmsShowModelPopup(const CmsShowModelPopup& rhs)
// {
//    // do actual copying here;
// }

CmsShowModelPopup::~CmsShowModelPopup()
{
   m_changes.disconnect();
   m_colorSelectWidget->Disconnect("ColorSelected(Pixel_t)", this, "changeModelColor(Pixel_t)");
   m_isVisibleButton->Disconnect("Toggled(Bool_t)", this, "toggleModelVisible(Bool_t)");
   disconnectAll();
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
CmsShowModelPopup::fillModelPopup(const FWSelectionManager& iSelMgr) {
   disconnectAll();
   if (iSelMgr.selected().size() > 0) {
      bool multipleNames(false);
      bool multipleColors(false);
      bool multipleVis(false);
      m_models = iSelMgr.selected();
      FWModelId id;
      FWModelId prevId;
      const FWEventItem* item = 0;
      const FWEventItem* prevItem = 0;
      for (std::set<FWModelId>::iterator it_mod = m_models.begin(); it_mod != m_models.end(); ++it_mod) {
         if (it_mod != m_models.begin()) {
            item = (*it_mod).item();
            if (item->name() != prevItem->name()) multipleNames = true;
            if (item->modelInfo((*it_mod).index()).displayProperties().color() != prevItem->modelInfo(prevId.index()).displayProperties().color())
               multipleColors = true;
            if (item->modelInfo((*it_mod).index()).displayProperties().isVisible() != prevItem->modelInfo(prevId.index()).displayProperties().isVisible())
               multipleVis = true;
         }
         prevId = *it_mod;
         prevItem = (*it_mod).item();
      }
      id = *m_models.begin();
      item = (*(m_models.begin())).item();
      if (multipleNames) {
         std::ostringstream s;
         s<<m_models.size()<<" objects";
         m_modelLabel->SetText(s.str().c_str());
      } else {
         std::ostringstream s;
         s<<m_models.size()<<" "<<item->name();
         m_modelLabel->SetText(s.str().c_str());
      }
      if(m_models.size()==1) {
         m_modelLabel->SetText(item->modelName(id.index()).c_str());
         m_openDetailedViewButton->SetEnabled(m_detailViewManager->haveDetailViewFor(id));
      }
      m_colorSelectWidget->SetColorByIndex(m_colorManager->colorToIndex(item->modelInfo(id.index()).displayProperties().color()), kFALSE);
      m_isVisibleButton->SetDisabledAndSelected(item->modelInfo(id.index()).displayProperties().isVisible());
      m_colorSelectWidget->SetEnabled(kTRUE);
      m_isVisibleButton->SetEnabled(kTRUE);
      //    m_displayChangedConn = m_item->defaultDisplayPropertiesChanged_.connect(boost::bind(&CmsShowEDI::updateDisplay, this));
      m_modelChangedConn = item->changed_.connect(boost::bind(&CmsShowModelPopup::updateDisplay, this));
      //    m_selectionChangedConn = m_selectionManager->selectionChanged_.connect(boost::bind(&CmsShowEDI::updateSelection, this));
      m_destroyedConn = item->goingToBeDestroyed_.connect(boost::bind(&CmsShowModelPopup::disconnectAll, this));
      Layout();
   }
}

void
CmsShowModelPopup::updateDisplay() {
   const FWEventItem* item;
   for (std::set<FWModelId>::iterator it_mod = m_models.begin(); it_mod != m_models.end(); ++it_mod) {
      item = (*it_mod).item();
      m_colorSelectWidget->SetColor(gVirtualX->GetPixel(item->modelInfo((*it_mod).index()).displayProperties().color()),kFALSE);
      m_isVisibleButton->SetState(item->modelInfo((*it_mod).index()).displayProperties().isVisible() ? kButtonDown : kButtonUp, kFALSE);
   }
}

void
CmsShowModelPopup::disconnectAll() {
   m_modelChangedConn.disconnect();
   m_destroyedConn.disconnect();
   //  m_item = 0;
   //  m_model = 0;
   m_modelLabel->SetText("No object selected");
   m_colorSelectWidget->SetColor(gVirtualX->GetPixel(kRed),kFALSE);
   m_isVisibleButton->SetDisabledAndSelected(kTRUE);
   m_colorSelectWidget->SetEnabled(kFALSE);
   m_isVisibleButton->SetEnabled(kFALSE);
   m_openDetailedViewButton->SetEnabled(kFALSE);
}

void
CmsShowModelPopup::changeModelColor(Color_t color) {
   const FWEventItem* item;
   if(m_models.size()) {
      FWChangeSentry sentry(*(m_models.begin()->item()->changeManager()));
      for (std::set<FWModelId>::iterator it_mod = m_models.begin(); it_mod != m_models.end(); ++it_mod) {
         item = (*it_mod).item();
         const FWDisplayProperties changeProperties(color, item->modelInfo((*it_mod).index()).displayProperties().isVisible());
         item->setDisplayProperties((*it_mod).index(), changeProperties);
      }
   }
}

void
CmsShowModelPopup::toggleModelVisible(Bool_t on) {
   const FWEventItem* item;
   if(m_models.size()) {
      FWChangeSentry sentry(*(m_models.begin()->item()->changeManager()));
      for (std::set<FWModelId>::iterator it_mod = m_models.begin(); it_mod != m_models.end(); ++it_mod) {
         item = (*it_mod).item();
         const FWDisplayProperties changeProperties(item->modelInfo((*it_mod).index()).displayProperties().color(), on);
         item->setDisplayProperties((*it_mod).index(), changeProperties);
      }
   }
   //  const FWDisplayProperties changeProperties(m_item->modelInfo(m_model->index()).displayProperties().color(), on);
   //  m_item->setDisplayProperties(m_model->index(), changeProperties);
}

void
CmsShowModelPopup::openDetailedView()
{
   m_detailViewManager->openDetailViewFor( *(m_models.begin()) );
}

//
// const member functions
//

//
// static member functions
//
