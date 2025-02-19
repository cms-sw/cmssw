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
// $Id: FWCollectionSummaryWidget.cc,v 1.24 2010/06/18 10:17:15 yana Exp $
//

// system include files
#include <iostream>
#include <vector>
#include <boost/bind.hpp>
#include "TGButton.h"
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
#include "Fireworks/Core/interface/FWColorManager.h"

//
// constants, enums and typedefs
//
struct FWCollectionSummaryWidgetConnectionHolder  {
   std::vector<sigc::connection> m_connections;
   
   ~FWCollectionSummaryWidgetConnectionHolder() {
      for_each(m_connections.begin(),m_connections.end(), boost::bind(&sigc::connection::disconnect,_1));
   }
   void push_back(const sigc::connection& iC) {
      m_connections.push_back(iC);
   }
   void reserve(size_t iSize) {
      m_connections.reserve(iSize);
   }
};


//
// static data member definitions
//
namespace {
   class BorderlessTextButton : public TGTextButton {
   public:
      BorderlessTextButton(const TGWindow *p = 0, const char *s = 0, Int_t id = -1,
                           GContext_t norm = GetDefaultGC()(),
                           FontStruct_t font = GetDefaultFontStruct()):
      TGTextButton(p,s,id,norm,font,0){
         //by default, it uses too much padding
         fMTop = -3;
         fMBottom = -3;
      }
      
      void DoRedraw();
   };
   
   void BorderlessTextButton::DoRedraw() {
      gVirtualX->ClearArea(fId, fBorderWidth, fBorderWidth,
                           fWidth - (fBorderWidth << 1), fHeight - (fBorderWidth << 1));
      //TGFrame::DoRedraw();
      
      int x, y;
      if (fTMode & kTextLeft) {
         x = fMLeft + 4;
      } else if (fTMode & kTextRight) {
         x = fWidth - fTWidth - fMRight - 4;
      } else {
         x = (fWidth - fTWidth + fMLeft - fMRight) >> 1;
      }

      if (fTMode & kTextTop) {
         y = fMTop + 3;
      } else if (fTMode & kTextBottom) {
         y = fHeight - fTHeight - fMBottom - 3;
      } else {
         y = (fHeight - fTHeight + fMTop - fMBottom) >> 1;
      }

      Int_t hotpos = fLabel->GetHotPos();

      fTLayout->DrawText(fId, fNormGC, x, y, 0, -1);
      if (hotpos) fTLayout->UnderlineChar(fId, fNormGC, x, y, hotpos - 1);
   }
}


static 
const TGPicture* filtered(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"filtered-blackbg.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"filtered-whitebg.png");
   return s;
   
}

static 
const TGPicture* filtered_over(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"filtered-whitebg-over.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"filtered-whitebg-over.png");
   return s;
}

static 
const TGPicture* alert_over(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"icon-alert-blackbg-over.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"icon-alert-whitebg-over.png");
   return s;
}

static 
const TGPicture* alert(bool iBackgroundIsBlack)
{
 
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"icon-alert-blackbg.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"icon-alert-whitebg.png");
   return s;
}

static 
const TGPicture* unfiltered(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"unfiltered-blackbg.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"unfiltered-whitebg.png");
   return s;
}
static 
const TGPicture* unfiltered_over(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"unfiltered-blackbg-over.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"unfiltered-whitebg-over.png");
   return s;   
}

static
const TGPicture* info(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"info2-blackbg.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"info2-whitebg.png");
   return s;   
}

static
const TGPicture* info_over(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"info2-blackbg-over.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"info2-whitebg-over.png");
   return s;
}

static
const TGPicture* info_disabled(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"info2-blackbg-disabled.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"info2-whitebg-disabled.png");
   return s;
}

static
const TGPicture* arrow_right(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"arrow-white-right-blackbg.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"arrow-black-right-whitebg.png");
   return s;
}

static
const TGPicture* arrow_right_disabled(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"arrow-white-right-disabled-blackbg.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"arrow-black-right-disabled-whitebg.png");
   return s;
}

static
const TGPicture* arrow_down(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"arrow-white-down-blackbg.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"arrow-black-down-whitebg.png");
   return s;
}

static
const TGPicture* arrow_down_disabled(bool iBackgroundIsBlack)
{
   if(iBackgroundIsBlack) {
      static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"arrow-white-down-disabled-blackbg.png");
      return s;
   }
   static const TGPicture* s = gClient->GetPicture(FWCheckBoxIcon::coreIcondir()+"arrow-black-down-disabled-whitebg.png");
   return s;
}


static const unsigned long kWidgetColor = 0x2f2f2f;
static const unsigned long kWidgetColorLight = 0xdfdfdf;
//
// constructors and destructor
//
FWCollectionSummaryWidget::FWCollectionSummaryWidget(TGFrame* iParent, FWEventItem& iItem, TGLayoutHints* iHints):
TGCompositeFrame(iParent),
m_collection(&iItem),
m_hints(iHints),
m_parent(iParent),
m_collectionShown(false),
m_tableContext(0),
m_indexForColor(-1),
m_colorPopup(0),
m_tableManager(0),
m_tableWidget(0),
m_backgroundIsWhite(false),
m_connectionHolder( new FWCollectionSummaryWidgetConnectionHolder)
{
   SetBackgroundColor(kWidgetColor);
   const unsigned int backgroundColor=kBlack;
   
   TGCompositeFrame* hFrame = new TGHorizontalFrame(this, 10, 10, 0, backgroundColor);
   m_holder = hFrame;
   this->AddFrame(hFrame, new TGLayoutHints(kLHintsTop | kLHintsExpandX) );
   
   m_showHideButton = new FWCustomIconsButton(this,
                                              arrow_right(!m_backgroundIsWhite),
                                              arrow_right_disabled(!m_backgroundIsWhite),
                                              arrow_right_disabled(!m_backgroundIsWhite));
   m_showHideButton->Connect("Clicked()","FWCollectionSummaryWidget",this,"toggleShowHide()");
   m_showHideButton->SetToolTipText("show/hide individual collection items");
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
   GCValues_t t = *(   GetWhiteGC().GetAttributes());
   m_graphicsContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&t,kTRUE);
   m_colorSelectBox->setColor(m_graphicsContext->GetGC());
   
   
   m_label = new BorderlessTextButton(this,
                                      m_collection->name().c_str());
   m_label->SetBackgroundColor(backgroundColor);
   m_label->SetTextJustify(kTextLeft|kTextCenterY);
   m_label->SetTextColor(static_cast<Pixel_t>(gVirtualX->GetPixel(kWhite)));
   hFrame->AddFrame(m_label, new TGLayoutHints(kLHintsCenterY | kLHintsLeft | kLHintsExpandX,5,5));
   m_label->Connect("Clicked()","FWCollectionSummaryWidget",this,"labelClicked()");
   m_label->SetToolTipText("select collection and show controller");
   
   m_stateButton = new FWCustomIconsButton(this,unfiltered(!m_backgroundIsWhite),
                                           unfiltered_over(!m_backgroundIsWhite),
                                           unfiltered_over(!m_backgroundIsWhite));
   hFrame->AddFrame(m_stateButton, new TGLayoutHints(kLHintsCenterY| kLHintsLeft));
   itemChanged();
   displayChanged();
   m_stateButton->Connect("Clicked()","FWCollectionSummaryWidget",this,"stateClicked()");
   m_stateButton->SetToolTipText("select collection and show filter");
   
   m_infoButton = new FWCustomIconsButton(this,
                                          info(!m_backgroundIsWhite),
                                          info_over(!m_backgroundIsWhite),
                                          info_disabled(!m_backgroundIsWhite)
   );
   hFrame->AddFrame(m_infoButton, new TGLayoutHints(kLHintsCenterY| kLHintsRight,2,2));
   m_infoButton->Connect("Clicked()","FWCollectionSummaryWidget",this,"infoClicked()");
   m_infoButton->SetToolTipText("select collection and show data info");

   m_connectionHolder->reserve(3);
   m_connectionHolder->push_back(m_collection->defaultDisplayPropertiesChanged_.connect(boost::bind(&FWCollectionSummaryWidget::displayChanged, this)));
   m_connectionHolder->push_back(m_collection->itemChanged_.connect(boost::bind(&FWCollectionSummaryWidget::itemChanged,this)));
   m_connectionHolder->push_back(m_collection->filterChanged_.connect(boost::bind(&FWCollectionSummaryWidget::itemChanged,this)));
   
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
   /* the following deletes lead to an infinite loop at the end of the job
   delete m_hints;
   delete m_showHideButton;
   delete m_isVisibleButton;
   delete m_colorSelectWidget;
   delete m_stateButton;
   delete m_infoButton;
    */
   gClient->GetResourcePool()->GetGCPool()->FreeGC(m_graphicsContext->GetGC());
   delete m_connectionHolder;
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
static
void
setLabelBackgroundColor(TGTextButton* iLabel, bool iIsSelected, bool iBackgroundIsWhite)
{
   if(iIsSelected) {
      if(iBackgroundIsWhite) {
         iLabel->SetBackgroundColor(0x7777FF);
      } else {
         iLabel->SetBackgroundColor(0x0000FF);
      }
   } else {
      if(iBackgroundIsWhite) {
         iLabel->SetBackgroundColor(0xFFFFFF);
      } else {
         iLabel->SetBackgroundColor(0x000000);
      }
   }
}

void
FWCollectionSummaryWidget::displayChanged()
{
   m_graphicsContext->SetForeground(gVirtualX->GetPixel(m_collection->defaultDisplayProperties().color()));
   fClient->NeedRedraw(m_colorSelectWidget);
   m_isVisibleCheckBox->setChecked(m_collection->defaultDisplayProperties().isVisible());
   fClient->NeedRedraw(m_isVisibleButton);
   setLabelBackgroundColor(m_label,m_collection->itemIsSelected(),m_backgroundIsWhite);
   fClient->NeedRedraw(m_label);
}

void
FWCollectionSummaryWidget::itemChanged()
{
   const TGPicture* picture = 0;
   const TGPicture* down = 0;
   const TGPicture* disabled=0;
   if(m_collection->hasError()) {
      picture = alert(!m_backgroundIsWhite);
      down = alert_over(!m_backgroundIsWhite);
      disabled = alert_over(!m_backgroundIsWhite);
      m_stateButton->SetToolTipText(m_collection->errorMessage().c_str());
   } else {
      if(m_collection->filterExpression().size()) {
         picture = filtered(!m_backgroundIsWhite);
         down = filtered_over(!m_backgroundIsWhite);
         disabled = filtered_over(!m_backgroundIsWhite);
      } else {
         picture = unfiltered(!m_backgroundIsWhite);
         down = unfiltered_over(!m_backgroundIsWhite);
         disabled = unfiltered_over(!m_backgroundIsWhite);
      }
      m_stateButton->SetToolTipText("select collection and show filter");
   }
   m_stateButton->swapIcons(picture,down,disabled);
}

void 
FWCollectionSummaryWidget::colorChangeRequested(Color_t color)
{
   if(-1 == m_indexForColor) {
      FWDisplayProperties changeProperties = m_collection->defaultDisplayProperties();
      changeProperties.setColor(color);
      m_collection->setDefaultDisplayProperties(changeProperties);
      return;
   }

   FWDisplayProperties changeProperties = m_collection->modelInfo(m_indexForColor).displayProperties();
   changeProperties.setColor(color);
   m_collection->setDisplayProperties(m_indexForColor,changeProperties);
}

void
FWCollectionSummaryWidget::toggleItemVisible() 
{
   m_isVisibleCheckBox->setChecked(!m_isVisibleCheckBox->isChecked());
   FWDisplayProperties changeProperties = m_collection->defaultDisplayProperties();
   changeProperties.setIsVisible(m_isVisibleCheckBox->isChecked());
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
      picture = arrow_right(!m_backgroundIsWhite);
      down = arrow_right_disabled(!m_backgroundIsWhite);
      disabled = arrow_right_disabled(!m_backgroundIsWhite);
      m_collectionShown = false;
      HideFrame(m_tableWidget);
      m_hints->SetLayoutHints(kLHintsExpandX);
   } else {
      picture = arrow_down(!m_backgroundIsWhite);
      down = arrow_down_disabled(!m_backgroundIsWhite);
      disabled = arrow_down_disabled(!m_backgroundIsWhite);
      m_collectionShown = true;
      
      if(0 == m_tableManager) {
         GCValues_t t = *(GetWhiteGC().GetAttributes());
         t.fFont = gClient->GetResourcePool()->GetIconFont()->GetFontHandle();
         m_tableContext = gClient->GetResourcePool()->GetGCPool()->GetGC(&t,kTRUE);
         
         TGGC* hilightContext=selectContext();
         m_tableManager= new FWCollectionSummaryTableManager(m_collection,m_tableContext,hilightContext,this);
         m_tableWidget = new FWTableWidget(m_tableManager,this);
         m_tableWidget->SetHeaderBackgroundColor(fClient->GetResourcePool()->GetFrameGC()->GetBackground());
         colorTable();
         AddFrame(m_tableWidget, new TGLayoutHints(kLHintsBottom | kLHintsExpandX | kLHintsExpandY));
         m_tableWidget->Connect("rowClicked(Int_t,Int_t,Int_t,Int_t,Int_t)","FWCollectionSummaryWidget",this,"modelSelected(Int_t,Int_t,Int_t,Int_t,Int_t)");

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
   if (0==m_colorPopup)
   {
      std::vector<Color_t> colors;
      m_collection->colorManager()->fillLimitedColors(colors);
     
      m_colorPopup = new FWColorPopup(gClient->GetDefaultRoot(), m_collection->defaultDisplayProperties().color());
      m_colorPopup->InitContent(m_collection->name().c_str(), colors);
      m_colorPopup->Connect("ColorSelected(Color_t)","FWCollectionSummaryWidget", this, "colorChangeRequested(Color_t)");
   }
}

void 
FWCollectionSummaryWidget::colorClicked()
{
   FWColorManager* cm = m_collection->colorManager();
   m_indexForColor=-1;

   createColorPopup();
   Window_t wdummy;
   Int_t ax, ay;
   gVirtualX->TranslateCoordinates(m_colorSelectWidget->GetId(), gClient->GetDefaultRoot()->GetId(), 0,
                                   m_colorSelectWidget->GetHeight(), ax, ay, wdummy);
   m_colorPopup->SetName(m_collection->name().c_str());
   std::vector<Color_t> colors;
   cm->fillLimitedColors(colors);
   m_colorPopup->ResetColors(colors, cm->backgroundColorIndex()==FWColorManager::kBlackIndex);
   m_colorPopup->SetSelection(m_collection->defaultDisplayProperties().color());
   m_colorPopup->PlacePopup(ax, ay, m_colorPopup->GetDefaultWidth(), m_colorPopup->GetDefaultHeight());
}

void 
FWCollectionSummaryWidget::itemColorClicked(int iIndex, Int_t iRootX, Int_t iRootY)
{
   FWColorManager* cm = m_collection->colorManager();
   m_indexForColor=iIndex;

   createColorPopup();
   std::vector<Color_t> colors;
   cm->fillLimitedColors(colors);
   m_colorPopup->ResetColors(colors, cm->backgroundColorIndex()==FWColorManager::kBlackIndex);
   m_colorPopup->SetName(m_collection->modelName(iIndex).c_str());
   m_colorPopup->SetSelection(m_collection->modelInfo(iIndex).displayProperties().color());
   m_colorPopup->PlacePopup(iRootX, iRootY, m_colorPopup->GetDefaultWidth(), m_colorPopup->GetDefaultHeight());
}

void 
FWCollectionSummaryWidget::modelSelected(Int_t iRow,Int_t iButton,Int_t iKeyMod,Int_t iGlobalX, Int_t iGlobalY)
{
   if(iKeyMod & kKeyControlMask) {      
      m_collection->toggleSelect(iRow);
   } else {
      FWChangeSentry sentry(*(m_collection->changeManager()));
      m_collection->selectionManager()->clearSelection();
      m_collection->select(iRow);
   }
   if(iButton==kButton3) {
      requestForModelContextMenu(iGlobalX,iGlobalY);
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
FWCollectionSummaryWidget::requestForController(FWEventItem* iItem)
{
   Emit("requestForController(FWEventItem*)",reinterpret_cast<long>(iItem));
}

void 
FWCollectionSummaryWidget::requestForModelContextMenu(Int_t iGlobalX,Int_t iGlobalY)
{
   Long_t args[2];
   args[0]=static_cast<Long_t>(iGlobalX);
   args[1]=static_cast<Long_t> (iGlobalY);
   Emit("requestForModelContextMenu(Int_t,Int_t)",args); 
}   

void
FWCollectionSummaryWidget::infoClicked()
{
   if(!m_collection->itemIsSelected()) {
      //NOTE: Want to be sure if models are selected then their collection is also selected
      m_collection->selectionManager()->clearSelection();
      m_collection->selectItem();
   }
   requestForInfo(m_collection);
}

void
FWCollectionSummaryWidget::stateClicked()
{
   if(!m_collection->itemIsSelected()) {
      //NOTE: Want to be sure if models are selected then their collection is also selected
      m_collection->selectionManager()->clearSelection();
      m_collection->selectItem();
   }
   requestForFilter(m_collection);
}

void
FWCollectionSummaryWidget::labelClicked()
{
   if(!m_collection->itemIsSelected()) {
      //NOTE: Want to be sure if models are selected then their collection is also selected
      m_collection->selectionManager()->clearSelection();
      m_collection->selectItem();
   }
   requestForController(m_collection);
}

void 
FWCollectionSummaryWidget::setBackgroundToWhite(bool iToWhite)
{
   if(iToWhite == m_backgroundIsWhite) {
      return;
   }
   Pixel_t bc = 0x000000;
   Pixel_t fg = 0xffffff;
   if(iToWhite) {
      bc = 0xffffff;
      fg = 0x000000;
      m_backgroundIsWhite=true;
      SetBackgroundColor(kWidgetColorLight);
      m_isVisibleButton->setNormCG(GetBlackGC()());
      m_colorSelectWidget->setNormCG(GetBlackGC()());
      selectContext()->SetForeground(0xafafFF);
   } else {
      m_backgroundIsWhite=false;
      SetBackgroundColor(kWidgetColor);
      m_isVisibleButton->setNormCG(GetWhiteGC()());
      m_colorSelectWidget->setNormCG(GetWhiteGC()());
      selectContext()->SetForeground(gClient->GetResourcePool()->GetSelectedGC()->GetBackground());
   }
   //this forces the icons to be changed to the correct background
   itemChanged();
   m_graphicsContext->SetForeground(gVirtualX->GetPixel(m_collection->defaultDisplayProperties().color()));
   {
      const TGPicture* picture = info(!m_backgroundIsWhite);
      const TGPicture* over = info_over(!m_backgroundIsWhite);
      const TGPicture* disabled = info_disabled(!m_backgroundIsWhite);
      m_infoButton->swapIcons(picture,
                              over,
                              disabled);
   }
   if(m_collectionShown) {
      const TGPicture* picture = arrow_down(!m_backgroundIsWhite);
      const TGPicture* down = arrow_down_disabled(!m_backgroundIsWhite);
      const TGPicture* disabled = arrow_down_disabled(!m_backgroundIsWhite);
      m_showHideButton->swapIcons(picture,down,disabled);
   } else {
      const TGPicture* picture = arrow_right(!m_backgroundIsWhite);
      const TGPicture* down = arrow_right_disabled(!m_backgroundIsWhite);
      const TGPicture* disabled = arrow_right_disabled(!m_backgroundIsWhite);
      m_showHideButton->swapIcons(picture,down,disabled);
   }
   colorTable();
   m_holder->SetBackgroundColor(bc);
   setLabelBackgroundColor(m_label,m_collection->itemIsSelected(),m_backgroundIsWhite);
   m_label->SetTextColor(fg);
   m_isVisibleButton->SetBackgroundColor(bc);
   m_colorSelectWidget->SetBackgroundColor(bc);
   fClient->NeedRedraw(m_isVisibleButton);
   fClient->NeedRedraw(m_colorSelectWidget);
   fClient->NeedRedraw(m_holder);
   fClient->NeedRedraw(this);
}

void
FWCollectionSummaryWidget::colorTable()
{
   if(0==m_tableWidget) {
      return;
   }
   if(m_backgroundIsWhite) {
      m_tableWidget->SetBackgroundColor(kWidgetColorLight);
      m_tableWidget->SetLineSeparatorColor(0x000000);
      m_tableContext->SetForeground(0x000000);
   } else {
      m_tableWidget->SetBackgroundColor(kWidgetColor);
      m_tableWidget->SetLineSeparatorColor(0xffffff);
      m_tableContext->SetForeground(0xffffff);
   }
}
//
// const member functions
//

//
// static member functions
//

ClassImp(FWCollectionSummaryWidget)

