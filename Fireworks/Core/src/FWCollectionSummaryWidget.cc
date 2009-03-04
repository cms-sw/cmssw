// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCollectionSummaryWidget
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Feb 14 10:02:32 CST 2009
// $Id$
//

// system include files
#include <iostream>
#include <vector>
#include <boost/bind.hpp>
#include "TSystem.h"
#include "TColor.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGResourcePool.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/src/FWBoxIconButton.h"
#include "Fireworks/Core/src/FWCheckBoxIcon.h"
#include "Fireworks/Core/src/FWColorBoxIcon.h"

// user include files
#include "Fireworks/Core/src/FWCollectionSummaryWidget.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"

#include "Fireworks/Core/src/FWCollectionSummaryTableManager.h"
#include "Fireworks/TableWidget/interface/FWTableWidget.h"

#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//
static const TString& coreIcondir() {
   static TString s = Form("%s/src/Fireworks/Core/icons/",gSystem->Getenv("CMSSW_BASE"));
   return s;
}

static 
const TGPicture* filtered()
{
   static const TGPicture* s = gClient->GetPicture(coreIcondir()+"filtered-blackbg.png");
   return s;
}

static 
const TGPicture* filtered_over()
{
   static const TGPicture* s = gClient->GetPicture(coreIcondir()+"filtered-blackbg-over.png");
   return s;
}

/*
static 
const TGPicture* alert_over()
{
   static const TGPicture* s = gClient->GetPicture(coreIcondir()+"alert-blackbg-over.png");
   return s;
}

static 
const TGPicture* alert()
{
   static const TGPicture* s = gClient->GetPicture(coreIcondir()+"alert-blackbg.png");
   return s;
}
*/
static 
const TGPicture* unfiltered()
{
   static const TGPicture* s = gClient->GetPicture(coreIcondir()+"unfiltered-blackbg.png");
   return s;
}
static 
const TGPicture* unfiltered_over()
{
   static const TGPicture* s = gClient->GetPicture(coreIcondir()+"unfiltered-blackbg-over.png");
   return s;
}

static const unsigned long kWidgetColor = 0x2f2f2f;
//
// constructors and destructor
//
FWCollectionSummaryWidget::FWCollectionSummaryWidget(TGFrame* iParent, FWEventItem& iItem, TGLayoutHints* iHints):
TGCompositeFrame(iParent),
m_collection(&iItem),
m_hints(iHints),
m_parent(iParent),
m_collectionShown(false),
m_indexForColor(-1),
m_colorPopup(0),
m_tableManager(0),
m_tableWidget(0)
{
   SetBackgroundColor(kWidgetColor);
   const unsigned int backgroundColor=kBlack;
   
   TGCompositeFrame* hFrame = new TGHorizontalFrame(this, 10, 10, 0, backgroundColor);
   this->AddFrame(hFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX) );
   
   m_showHideButton = new FWCustomIconsButton(this,
                                              fClient->GetPicture(coreIcondir()+"arrow-white-right-blackbg.png"),
                                              fClient->GetPicture(coreIcondir()+"arrow-white-right-disabled-blackbg.png"),
                                              fClient->GetPicture(coreIcondir()+"arrow-white-right-disabled-blackbg.png"));
   m_showHideButton->Connect("Clicked()","FWCollectionSummaryWidget",this,"toggleShowHide()");
   m_collectionShown = false;
   hFrame->AddFrame(m_showHideButton,new TGLayoutHints(kLHintsCenterY | kLHintsLeft,6,10));
   
   //m_isVisibleButton = new TGCheckButton(this,"");
   //m_isVisibleButton->SetState(kButtonDown, kFALSE);
   m_isVisibleCheckBox = new FWCheckBoxIcon(12);
   m_isVisibleButton = new FWBoxIconButton(this, m_isVisibleCheckBox,-1,GetWhiteGC()());
   m_isVisibleButton->SetBackgroundColor(backgroundColor);
   m_isVisibleButton->SetToolTipText("make all items in collection visible/invisible");
   hFrame->AddFrame(m_isVisibleButton,new TGLayoutHints(kLHintsCenterY | kLHintsLeft,0,1));
   m_isVisibleButton->Connect("Clicked()", "FWCollectionSummaryWidget", this, "toggleItemVisible()");

   m_colorSelectBox = new FWColorBoxIcon(12);
   m_colorSelectWidget = new FWBoxIconButton(this,m_colorSelectBox,-1,GetWhiteGC()());
   hFrame->AddFrame(m_colorSelectWidget,new TGLayoutHints(kLHintsCenterY | kLHintsLeft,1));
   //m_colorSelectWidget->Connect("ColorSelected(Pixel_t)", "FWCollectionSummaryWidget", this, "colorChangeRequested(Pixel_t)");
   m_colorSelectWidget->Connect("Clicked()", "FWCollectionSummaryWidget",this,"colorClicked()");
   m_colorSelectWidget->SetBackgroundColor(backgroundColor);
   m_colorSelectWidget->SetToolTipText("set default color of items in collection");
   GCValues_t t = *(GetWhiteGC().GetAttributes());
   m_graphicsContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&t,kTRUE);
   m_colorSelectBox->setColor(m_graphicsContext->GetGC());
   
   
   TGLabel* label = new TGLabel(this,m_collection->name().c_str());
   label->SetBackgroundColor(backgroundColor);
   label->SetTextJustify(kTextLeft|kTextCenterY);
   label->SetTextColor(static_cast<Pixel_t>(gVirtualX->GetPixel(kWhite)));
   hFrame->AddFrame(label, new TGLayoutHints(kLHintsCenterY | kLHintsLeft | kLHintsExpandX,5,5));
   
   m_stateButton = new FWCustomIconsButton(this,unfiltered(),unfiltered_over(),unfiltered_over());
   hFrame->AddFrame(m_stateButton, new TGLayoutHints(kLHintsCenterY| kLHintsLeft));
   itemChanged();
   displayChanged();
   m_stateButton->Connect("Clicked()","FWCollectionSummaryWidget",this,"stateClicked()");
   m_stateButton->SetToolTipText("show collection filter");
   
   m_infoButton = new FWCustomIconsButton(this,
                                          fClient->GetPicture(coreIcondir()+"info2-blackbg.png"),
                                          fClient->GetPicture(coreIcondir()+"info2-blackbg-over.png"),
                                          fClient->GetPicture(coreIcondir()+"info2-blackbg-disabled.png")
   );
   hFrame->AddFrame(m_infoButton, new TGLayoutHints(kLHintsCenterY| kLHintsRight,2,2));
   m_infoButton->Connect("Clicked()","FWCollectionSummaryWidget",this,"infoClicked()");
   m_infoButton->SetToolTipText("show Collection Controller");

   m_collection->defaultDisplayPropertiesChanged_.connect(boost::bind(&FWCollectionSummaryWidget::displayChanged, this));
   m_collection->itemChanged_.connect(boost::bind(&FWCollectionSummaryWidget::itemChanged,this));
   
   MapSubwindows();
   Layout();
   MapWindow();
}

// FWCollectionSummaryWidget::FWCollectionSummaryWidget(const FWCollectionSummaryWidget& rhs)
// {
//    // do actual copying here;
// }

FWCollectionSummaryWidget::~FWCollectionSummaryWidget()
{
   delete m_colorPopup;
   gClient->GetResourcePool()->GetGCPool()->FreeGC(m_graphicsContext->GetGC());
}

//
// assignment operators
//
// const FWCollectionSummaryWidget& FWCollectionSummaryWidget::operator=(const FWCollectionSummaryWidget& rhs)
// {
//   //An exception safe implementation is
//   FWCollectionSummaryWidget temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
FWCollectionSummaryWidget::displayChanged()
{
   //m_colorSelectWidget->SetColor(gVirtualX->GetPixel(m_collection->defaultDisplayProperties().color()),kFALSE);
   m_graphicsContext->SetForeground(gVirtualX->GetPixel(m_collection->defaultDisplayProperties().color()));
   fClient->NeedRedraw(m_colorSelectWidget);
   m_isVisibleCheckBox->setChecked(m_collection->defaultDisplayProperties().isVisible());
   fClient->NeedRedraw(m_isVisibleButton);
}

void
FWCollectionSummaryWidget::itemChanged()
{
   const TGPicture* picture = 0;
   const TGPicture* down = 0;
   const TGPicture* disabled=0;
   if(m_collection->filterExpression().size()) {
      picture = filtered();
      down = filtered_over();
      disabled = filtered_over();
   } else {
      picture = unfiltered();
      down = unfiltered_over();
      disabled = unfiltered_over();
   }
   m_stateButton->swapIcons(picture,down,disabled);
}

void 
FWCollectionSummaryWidget::colorChangeRequested(Pixel_t iPixel)
{
   Color_t color(TColor::GetColor(iPixel));
   if(-1 == m_indexForColor) {
      const FWDisplayProperties changeProperties(color, m_collection->defaultDisplayProperties().isVisible());
      m_collection->setDefaultDisplayProperties(changeProperties);
      return;
   }
   const FWDisplayProperties changeProperties(color, m_collection->modelInfo(m_indexForColor).displayProperties().isVisible());
   m_collection->setDisplayProperties(m_indexForColor,changeProperties);

}

void
FWCollectionSummaryWidget::toggleItemVisible() 
{
   m_isVisibleCheckBox->setChecked(!m_isVisibleCheckBox->isChecked());
   const FWDisplayProperties changeProperties(m_collection->defaultDisplayProperties().color(), m_isVisibleCheckBox->isChecked());
   m_collection->setDefaultDisplayProperties(changeProperties);
   fClient->NeedRedraw(m_isVisibleButton);
}

static
TGGC* selectContext()
{
   static TGGC* s_context = 0;
   if(0==s_context) {
      GCValues_t hT = *(gClient->GetResourcePool()->GetSelectedGC()->GetAttributes());
      s_context = gClient->GetResourcePool()->GetGCPool()->GetGC(&hT,kTRUE);
      s_context->SetForeground(s_context->GetBackground());
      //s_context->SetForeground(gVirtualX->GetPixel(kBlue+2));
   }
   return s_context;
}

void 
FWCollectionSummaryWidget::toggleShowHide()
{
   const TGPicture* picture = 0;
   const TGPicture* down = 0;
   const TGPicture* disabled=0;
   
   if(m_collectionShown) {
      picture = fClient->GetPicture(coreIcondir()+"arrow-white-right-blackbg.png");
      down = fClient->GetPicture(coreIcondir()+"arrow-white-right-disabled-blackbg.png");
      disabled = fClient->GetPicture(coreIcondir()+"arrow-white-right-disabled-blackbg.png");
      m_collectionShown = false;
      HideFrame(m_tableWidget);
      m_hints->SetLayoutHints(kLHintsExpandX);
   } else {
      picture = fClient->GetPicture(coreIcondir()+"arrow-white-down-blackbg.png");
      down = fClient->GetPicture(coreIcondir()+"arrow-white-down-disabled-blackbg.png");
      disabled = fClient->GetPicture(coreIcondir()+"arrow-white-down-blackbg.png");
      m_collectionShown = true;
      
      if(0 == m_tableManager) {
         GCValues_t t = *(GetWhiteGC().GetAttributes());
         t.fFont = gClient->GetResourcePool()->GetIconFont()->GetFontHandle();
         TGGC* tableContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&t,kTRUE);
         
         TGGC* hilightContext=selectContext();
         m_tableManager= new FWCollectionSummaryTableManager(m_collection,tableContext,hilightContext,this);
         m_tableWidget = new FWTableWidget(m_tableManager,this);
         m_tableWidget->SetBackgroundColor(kWidgetColor);
         AddFrame(m_tableWidget, new TGLayoutHints(kLHintsBottom | kLHintsExpandX | kLHintsExpandY));
         m_tableWidget->Connect("rowClicked(Int_t,Int_t,Int_t)","FWCollectionSummaryWidget",this,"modelSelected(Int_t,Int_t,Int_t)");

         MapSubwindows();
         Layout();
      }
      ShowFrame(m_tableWidget);
      m_hints->SetLayoutHints(kLHintsExpandX|kLHintsExpandY);
      //NOTE: if I don't do the resize then the vertical scrollbars for the table are 
      // messed up when the number of entries in the table can be fully scene but 
      // a scrollbar is still added which thinks only a tiny area of the list can be seen
      m_tableWidget->Resize(m_tableWidget->GetWidth(),m_tableWidget->GetHeight());
   }
   
   if(0!=m_parent) {
      m_parent->Layout();
   }
   m_showHideButton->swapIcons(picture,down,disabled);
}



void 
FWCollectionSummaryWidget::createColorPopup()
{
   if(0==m_colorPopup) {
      TGString* graphicsLabel = new TGString(m_collection->name().c_str());
      Pixel_t selection = gVirtualX->GetPixel(m_collection->defaultDisplayProperties().color());
      std::vector<Pixel_t> colors;
      colors.push_back((Pixel_t)gVirtualX->GetPixel(kRed));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(kBlue));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(kYellow));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(kGreen));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(kCyan));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(kMagenta));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(kOrange));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(880));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(840));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kRed)));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kBlue)));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kYellow)));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kGreen)));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kCyan)));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kMagenta)));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(kOrange)));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(880)));
      colors.push_back((Pixel_t)gVirtualX->GetPixel(TColor::GetColorDark(840)));
      bool haveColor = false;
      for (std::vector<Pixel_t>::const_iterator iCol = colors.begin(); iCol != colors.end(); ++iCol) {
         if (*iCol == selection) haveColor = true;
      }
      if(!haveColor) {
         std::cout<<"Do not have color ! "<<m_collection->defaultDisplayProperties().color()<<std::endl;
         std::cout <<"  "<<m_collection->name()<<std::endl;
         colors.push_back(selection);
      }
      
      m_colorPopup = new FWColorPopup(gClient->GetDefaultRoot(), selection);
      m_colorPopup->InitContent(graphicsLabel, colors);
      m_colorPopup->Connect("ColorBookkeeping(Int_t)","FWCollectionSummaryWidget", this, "colorChangeRequested(Pixel_t)");
   }   
}

void 
FWCollectionSummaryWidget::colorClicked() {
   createColorPopup();
   Window_t wdummy;
   Int_t ax, ay;
   gVirtualX->TranslateCoordinates(m_colorSelectWidget->GetId(), gClient->GetDefaultRoot()->GetId(), 0,
                                   m_colorSelectWidget->GetHeight(), ax, ay, wdummy);
   m_indexForColor=-1;
   m_colorPopup->SetName(m_collection->name().c_str());
   m_colorPopup->SetSelection(gVirtualX->GetPixel(m_collection->defaultDisplayProperties().color()));
   m_colorPopup->PlacePopup(ax, ay, m_colorPopup->GetDefaultWidth(), m_colorPopup->GetDefaultHeight());
}

void 
FWCollectionSummaryWidget::itemColorClicked(int iIndex, Int_t iRootX, Int_t iRootY)
{
   createColorPopup();
   m_indexForColor=iIndex;
   m_colorPopup->SetName(m_collection->modelName(iIndex).c_str());
   m_colorPopup->SetSelection(gVirtualX->GetPixel(m_collection->modelInfo(iIndex).displayProperties().color()));
   m_colorPopup->PlacePopup(iRootX, iRootY, m_colorPopup->GetDefaultWidth(), m_colorPopup->GetDefaultHeight());
}

void 
FWCollectionSummaryWidget::modelSelected(Int_t iRow,Int_t iButton,Int_t iKeyMod)
{
   if(iKeyMod & kKeyControlMask) {      
      m_collection->toggleSelect(iRow);
   } else {
      FWChangeSentry sentry(*(m_collection->changeManager()));
      m_collection->selectionManager()->clearSelection();
      m_collection->select(iRow);
   }
}


void 
FWCollectionSummaryWidget::requestForInfo(FWEventItem* iItem)
{
   Emit("requestForInfo(FWEventItem*)",reinterpret_cast<long>(iItem));
}

void 
FWCollectionSummaryWidget::requestForFilter(FWEventItem* iItem)
{
   Emit("requestForFilter(FWEventItem*)", reinterpret_cast<long>(iItem));
}

void 
FWCollectionSummaryWidget::requestForErrorInfo(FWEventItem* iItem)
{
   Emit("requestForErrorInfo(FWEventItem*)",reinterpret_cast<long>(iItem));
}

void
FWCollectionSummaryWidget::infoClicked()
{
   requestForInfo(m_collection);
}

void
FWCollectionSummaryWidget::stateClicked()
{
   requestForFilter(m_collection);
}

//
// const member functions
//

//
// static member functions
//

ClassImp(FWCollectionSummaryWidget)
