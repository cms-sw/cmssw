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
// $Id: CmsShowEDI.cc,v 1.51 2013/04/05 22:39:44 amraktad Exp $
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
#include "TGNumberEntry.h"
#include "TGTextEntry.h"
#include "TGTextView.h"
#include "TGLayout.h"
#include "TGFont.h"
#include "TEveManager.h"
#include "TGSlider.h"
#include "TGMsgBox.h"
#include "TGComboBox.h"

// user include files
#include "Fireworks/Core/interface/CmsShowEDI.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWDisplayProperties.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWProxyBuilderConfiguration.h"
#include "Fireworks/Core/src/FWColorSelect.h"
#include "Fireworks/Core/interface/FWModelChangeSignal.h"
#include "Fireworks/Core/interface/FWModelExpressionSelector.h"
#include "Fireworks/Core/interface/FWModelChangeManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWExpressionException.h"
#include "Fireworks/Core/src/FWGUIValidatingTextEntry.h"
#include "Fireworks/Core/src/FWExpressionValidator.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"

//
// constants, enums and typedefs
///

//
// static data member definitions
//

//
// constructors and destructor
//
CmsShowEDI::CmsShowEDI(const TGWindow* p, UInt_t w, UInt_t h, FWSelectionManager* selMgr, FWColorManager* colorMgr) :
   TGTransientFrame(gClient->GetDefaultRoot(), p, w, h),
   m_item(0),
   m_validator(new FWExpressionValidator),
   m_colorManager(colorMgr),
   m_settersFrame(0)
{
   m_selectionManager = selMgr;
   SetCleanup(kDeepCleanup);

   m_selectionManager->itemSelectionChanged_.connect(boost::bind(&CmsShowEDI::fillEDIFrame,this));

   TGVerticalFrame* vf = new TGVerticalFrame(this);
   AddFrame(vf, new TGLayoutHints(kLHintsExpandX|kLHintsExpandY, 0, 0, 0, 0));
   FWDialogBuilder builder(vf);

   builder.indent(0)
      .addLabel(" ", 14, 2, &m_objectLabel)
      .vSpacer()
      .tabs(&m_tabs)
      .beginTab("Graphics")
      .indent(3)
      .addLabel("Color", 8)
      .addColorPicker(colorMgr, &m_colorSelectWidget).expand(false)
      .addHSeparator()
      .addLabel("Opacity", 8)
      .addHSlider(150, &m_opacitySlider)
      .addHSeparator()
      .addCheckbox("Visible", &m_isVisibleButton)
      .addHSeparator()
      .addLabel("Drawing order", 8)
      .addTextButton("To back", &m_backButton).floatLeft().expand(false)
      .addNumberEntry(0.0, 4, TGNumberFormat::kNESInteger,
                      FWEventItem::minLayerValue(), 
                      FWEventItem::maxLayerValue(), 
                      &m_layerEntry).expand(false).floatLeft()
      .addTextButton("To front", &m_frontButton).expand(false)
      .vSpacer()
      .addHSeparator()
      .endTab()
      .beginTab("Filter")
      .indent(3)
      .addLabel("Expression", 8)
      .addValidatingTextEntry(0, &m_filterExpressionEntry).floatLeft()
      .addTextButton("Filter", &m_filterButton).expand(false)
      .addTextView("", &m_filterError)
      .vSpacer()
      .endTab()
      .beginTab("Select")
      .indent(3)
      .addLabel("Expression", 8)
      .addValidatingTextEntry(0, &m_selectExpressionEntry)
      .addTextButton("Select", &m_selectButton).floatLeft().expand(false)
      .addTextButton("Select all", &m_selectAllButton).expand(false).floatLeft()
      .addTextButton("Unselect all", &m_deselectAllButton).expand(false)
      .indent(3)
      .addLabel("Color Selection", 8)
      .addColorPicker(colorMgr, &m_cw).expand(false)
      .addTextView("", &m_selectError)
      .vSpacer()
      .endTab()
      .beginTab("Data")
      .indent(3)
      .addLabel("Name:", 8)
      .addTextEntry("None", &m_nameEntry)
      .addLabel("Labels:", 8)
      .addLabel("Type:", 8)
      .addTextEntry("None", &m_typeEntry)
      .addLabel("Module:", 8)
      .addTextEntry("None", &m_moduleEntry)
      .addLabel("Instance:", 8)
      .addTextEntry("None", &m_instanceEntry)
      .addLabel("Process:", 8)
      .addTextEntry("None", &m_processEntry)
      .addHSeparator()
      .addTextButton("Remove collection", &m_removeButton).expand(false)
      .vSpacer()
      .endTab()
      .untabs();

   m_filterError->SetForegroundColor(gVirtualX->GetPixel(kRed));
   m_filterError->SetBackgroundColor(TGFrame::GetDefaultFrameBackground());
   m_filterError->ChangeOptions(0);
   
   m_selectError->SetForegroundColor(gVirtualX->GetPixel(kRed));
   m_selectError->SetBackgroundColor(TGFrame::GetDefaultFrameBackground());
   m_selectError->ChangeOptions(0);

   m_filterExpressionEntry->setValidator(m_validator);
   m_selectExpressionEntry->setValidator(m_validator);

   m_colorSelectWidget->Connect("ColorChosen(Color_t)", "CmsShowEDI", this, "changeItemColor(Color_t)");
   m_cw->Connect("ColorChosen(Color_t)", "CmsShowEDI", this, "changeSelectionColor(Color_t)");
   m_opacitySlider->Connect("PositionChanged(Int_t)", "CmsShowEDI", this, "changeItemOpacity(Int_t)");
   m_isVisibleButton->Connect("Toggled(Bool_t)", "CmsShowEDI", this, "toggleItemVisible(Bool_t)");
   m_filterExpressionEntry->Connect("ReturnPressed()", "CmsShowEDI", this, "runFilter()");
   m_filterButton->Connect("Clicked()", "CmsShowEDI", this, "runFilter()");
   m_selectExpressionEntry->Connect("ReturnPressed()", "CmsShowEDI", this, "runSelection()");
   m_selectButton->Connect("Clicked()", "CmsShowEDI", this, "runSelection()");
   m_removeButton->Connect("Clicked()", "CmsShowEDI", this, "removeItem()");
   m_selectAllButton->Connect("Clicked()", "CmsShowEDI", this, "selectAll()");
   m_deselectAllButton->Connect("Clicked()", "CmsShowEDI", this, "deselectAll()");
   m_frontButton->Connect("Clicked()","CmsShowEDI",this,"moveToFront()");
   m_backButton->Connect("Clicked()","CmsShowEDI",this,"moveToBack()");
   m_layerEntry->Connect("ValueSet(Long_t)","CmsShowEDI",this,"moveToLayer(Long_t)");
   

   TGCompositeFrame* cf = m_tabs->GetTabContainer(0);
   m_settersFrame = new TGVerticalFrame(cf);
   m_settersFrame->SetCleanup(kDeepCleanup);
   // m_settersFrame->SetBackgroundColor(0xff00ff);
   cf->AddFrame(m_settersFrame, new TGLayoutHints(kLHintsExpandX| kLHintsExpandY ));

   SetWindowName("Collection Controller");
   MapSubwindows();
   Resize(GetDefaultSize());
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
   m_cw->Disconnect("ColorSelected(Pixel_t)", this, "changeSelectionColor(Pixel_t)");
   m_opacitySlider->Disconnect("PositionChanged(Int_t)", this, "changeItemColor");
   m_isVisibleButton->Disconnect("Toggled(Bool_t)", this, "toggleItemVisible(Bool_t)");
   m_filterExpressionEntry->Disconnect("ReturnPressed()", this, "runFilter()");
   m_selectExpressionEntry->Disconnect("ReturnPressed()", this, "runSelection()");
   m_filterButton->Disconnect("Clicked()", this, "runFilter()");
   m_selectButton->Disconnect("Clicked()", this, "runSelection()");
   m_selectAllButton->Disconnect("Clicked()", this, "selectAll()");
   m_deselectAllButton->Disconnect("Clicked()", this, "deselectAll()");
   m_removeButton->Disconnect("Clicked()", this, "removeItem()");
   m_frontButton->Disconnect("Clicked()",this,"moveToFront()");
   m_backButton->Disconnect("Clicked()",this,"moveToBack()");
   m_layerEntry->Disconnect("ValueSet(Long_t)",this,"moveToLayer(Long_t)");
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

void CmsShowEDI::clearPBFrame()
{
   if (!m_settersFrame->GetList()->IsEmpty())
   {
      // printf("remove FRAME \n");
      TGFrameElement *el = (TGFrameElement*) m_settersFrame->GetList()->First();
      TGFrame* f = el->fFrame;
      f->UnmapWindow();
      m_settersFrame->RemoveFrame(f);
      f->DestroyWindow();
   }
}

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
         const FWDisplayProperties &p = iItem->defaultDisplayProperties();
         m_objectLabel->SetText(iItem->name().c_str());
         m_colorSelectWidget->SetColorByIndex(p.color(),kFALSE);
         m_opacitySlider->SetPosition(100 - p.transparency());
         m_isVisibleButton->SetDisabledAndSelected(p.isVisible());
         m_validator->setType(edm::TypeWithDict(*(iItem->modelType()->GetTypeInfo())));
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
         m_opacitySlider->SetEnabled(kTRUE);
         m_isVisibleButton->SetEnabled(kTRUE);
         m_filterExpressionEntry->SetEnabled(kTRUE);
         m_selectExpressionEntry->SetEnabled(kTRUE);
         m_filterButton->SetEnabled(kTRUE);
         m_selectButton->SetEnabled(kTRUE);
         m_selectAllButton->SetEnabled(kTRUE);
         m_deselectAllButton->SetEnabled(kTRUE);
	 m_cw->SetColorByIndex(p.color(),kFALSE);
         m_cw->SetEnabled(kTRUE);
         m_removeButton->SetEnabled(kTRUE);
         updateLayerControls();
         m_layerEntry->SetState(kTRUE);
         m_displayChangedConn = m_item->defaultDisplayPropertiesChanged_.connect(boost::bind(&CmsShowEDI::updateDisplay, this));
         m_modelChangedConn = m_item->changed_.connect(boost::bind(&CmsShowEDI::updateFilter, this));
         //    m_selectionChangedConn = m_selectionManager->selectionChanged_.connect(boost::bind(&CmsShowEDI::updateSelection, this));
         m_destroyedConn = m_item->goingToBeDestroyed_.connect(boost::bind(&CmsShowEDI::disconnectAll, this));
                       
         clearPBFrame();
         m_item->getConfig()->populateFrame(m_settersFrame);
      }
      else if(multipleCollections) {
         std::ostringstream s;
         s<<m_selectionManager->selectedItems().size()<<" Collections Selected";
         m_objectLabel->SetText(s.str().c_str());
      }

      Resize(GetDefaultSize());
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
   const FWDisplayProperties &props = m_item->defaultDisplayProperties();
   m_colorSelectWidget->SetColorByIndex(props.color(),kFALSE);
   m_opacitySlider->SetPosition(100 - props.transparency());
   m_isVisibleButton->SetState(props.isVisible() ? kButtonDown : kButtonUp, kFALSE);
}

void
CmsShowEDI::updateLayerControls()
{
   m_backButton->SetEnabled(!m_item->isInBack());
   m_frontButton->SetEnabled(!m_item->isInFront());
   m_layerEntry->SetIntNumber(m_item->layer());
}
void
CmsShowEDI::moveToBack()
{
   m_item->moveToBack();
   updateLayerControls();
}
void
CmsShowEDI::moveToFront()
{
   m_item->moveToFront();
   updateLayerControls();
}
void
CmsShowEDI::moveToLayer(Long_t)
{
   m_item->moveToLayer(static_cast<Int_t>(m_layerEntry->GetIntNumber()));
   updateLayerControls(); 
}

void
CmsShowEDI::updateFilter() {
   m_filterExpressionEntry->SetText(m_item->filterExpression().c_str());
}

void
CmsShowEDI::disconnectAll() {
   m_objectLabel->SetText("No Collection Selected");
   clearPBFrame();
   if(0 != m_item) {
      m_displayChangedConn.disconnect();
      m_modelChangedConn.disconnect();
      m_destroyedConn.disconnect();
      m_item = 0;
      m_colorSelectWidget->SetColorByIndex(0,kFALSE);
      m_opacitySlider->SetPosition(100);
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
      m_opacitySlider->SetEnabled(kFALSE);
      
      m_isVisibleButton->SetEnabled(kFALSE);
      m_filterExpressionEntry->SetEnabled(kFALSE);
      m_filterButton->SetEnabled(kFALSE);
      m_selectExpressionEntry->SetEnabled(kFALSE);
      m_selectButton->SetEnabled(kFALSE);
      m_selectAllButton->SetEnabled(kFALSE);
      m_deselectAllButton->SetEnabled(kFALSE);
      m_removeButton->SetEnabled(kFALSE);
      m_backButton->SetEnabled(kFALSE);
      m_frontButton->SetEnabled(kFALSE);
      m_layerEntry->SetIntNumber(0);
      m_layerEntry->SetState(kFALSE);
      m_layerEntry->GetNumberEntry()->SetEnabled(kFALSE);
      m_layerEntry->GetButtonUp()->SetEnabled(kFALSE);
      m_layerEntry->GetButtonDown()->SetEnabled(kFALSE);
   }
}

/** Set the item color. 
    
    Notice that I changed this to use a "Copy and modify approach", rather
    than a "create with old properties" method which was not propagating 
    transparency.
  */
void
CmsShowEDI::changeItemColor(Color_t color) {
   FWDisplayProperties changeProperties = m_item->defaultDisplayProperties();
   changeProperties.setColor(color);
   m_item->setDefaultDisplayProperties(changeProperties);
   m_cw->SetColorByIndex(color,kFALSE);
}

/** See changeItemColor for additional details.*/
void
CmsShowEDI::toggleItemVisible(Bool_t on) {
   FWDisplayProperties changeProperties = m_item->defaultDisplayProperties();
   changeProperties.setIsVisible(on);
   m_item->setDefaultDisplayProperties(changeProperties);
}

/** Changes selected item opacity. Notice that we use opacity rather than
    transparency because this way the slider is by default 100% rather than 0.
    This is more a more natural and positive way of looking at things. 
    
    Glass is full! 
    
    See changeItemColor for additional details.*/
void
CmsShowEDI::changeItemOpacity(Int_t opacity) {
   FWDisplayProperties changeProperties = m_item->defaultDisplayProperties();
   changeProperties.setTransparency(100 - opacity);
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
   if (m_item != 0)
   {
      try
      {
         m_selectError->Clear();
         //NOTE call clearModelSelectionLeaveItem so that the item does not get deselected
         // just for safety use a copy of the pointer to m_item
         FWEventItem* item = m_item;
         item->selectionManager()-> clearModelSelectionLeaveItem();

         selector.select(item, selection, TColor::GetColor(m_cw->GetColor()));
      }
      catch( const FWExpressionException& e)
      {
         m_selectError->AddLine(e.what().c_str());
         m_selectError->Update();
         if (e.column() > -1)
         {
            m_selectExpressionEntry->SetCursorPosition(e.column());
         }
      }
   }
}

void
CmsShowEDI::selectAll()
{
   FWChangeSentry sentry(*(m_item->changeManager()));
   for (int i = 0; i < static_cast<int>(m_item->size()); i++)
   {
      m_item->select(i);
   }
}
void
CmsShowEDI::deselectAll()
{
   FWChangeSentry sentry(*(m_item->changeManager()));
   for (int i = 0; i < static_cast<int>(m_item->size()); i++)
   {
      m_item->unselect(i);
   }
}

void 
CmsShowEDI::changeSelectionColor(Color_t c)
{
   FWChangeSentry sentry(*(m_item->changeManager()));
   const std::set<FWModelId>& ss =  m_item->selectionManager()->selected();
   FWDisplayProperties dp = m_item->defaultDisplayProperties();
   dp.setColor(c);
   for (std::set<FWModelId>::const_iterator i = ss.begin(); i != ss.end(); ++i ) {
     m_item->setDisplayProperties(i->index(), dp);
   }
}


void 
CmsShowEDI::show(FWDataCategories iToView)
{
   m_tabs->SetTab(iToView);
}

/* Called by FWGUIManager when change background/colorset. */
void 
CmsShowEDI::colorSetChanged()
{
   if (m_item)
   {
      const FWDisplayProperties &p = m_item->defaultDisplayProperties();
      m_colorSelectWidget->SetColorByIndex(p.color(),kFALSE);
      m_cw->SetColorByIndex(p.color(),kFALSE);
   }
}
