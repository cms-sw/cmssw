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
//

// system include files
#include <iostream>
#include <functional>
#include <cassert>
#include "TGLabel.h"
#include "TGButton.h"
#include "TG3DLine.h"
#include "TGFrame.h"
#include "TGTab.h"
#include "TG3DLine.h"
#include "TEveWindow.h"

// user include files
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"
#include "Fireworks/Core/src/FWDialogBuilder.h"
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
CmsShowViewPopup::CmsShowViewPopup(
    const TGWindow* p, UInt_t w, UInt_t h, FWColorManager* iCMgr, FWViewBase* vb, TEveWindow* ew)
    : TGTransientFrame(gClient->GetDefaultRoot(), p, w, h),
      m_mapped(kFALSE),
      m_viewLabel(nullptr),
      m_paramGUI(nullptr),
      m_saveImageButton(nullptr),
      m_changeBackground(nullptr),
      m_colorManager(iCMgr),
      m_viewBase(nullptr),
      m_eveWindow(nullptr) {
  m_colorManager->colorsHaveChanged_.connect(std::bind(&CmsShowViewPopup::backgroundColorWasChanged, this));

  SetCleanup(kDeepCleanup);

  // label
  TGHorizontalFrame* viewFrame = new TGHorizontalFrame(this);
  m_viewLabel = new TGLabel(viewFrame, "No view selected");
  try {
    TGFont* defaultFont = gClient->GetFontPool()->GetFont(m_viewLabel->GetDefaultFontStruct());
    m_viewLabel->SetTextFont(gClient->GetFontPool()->GetFont(defaultFont->GetFontAttributes().fFamily,
                                                             14,
                                                             defaultFont->GetFontAttributes().fWeight + 2,
                                                             defaultFont->GetFontAttributes().fSlant));
  } catch (...) {
    // FIXME: looks like under certain conditions (e.g. in full framework)
    // GetFontPool() throws when the default font is not found. This is a
    // quick workaround, but we should probably investigate more.
  }

  m_viewLabel->SetTextJustify(kTextLeft);
  viewFrame->AddFrame(m_viewLabel, new TGLayoutHints(kLHintsExpandX));
  AddFrame(viewFrame, new TGLayoutHints(kLHintsExpandX, 2, 2, 0, 0));
  // background
  m_changeBackground = new TGTextButton(this, "Change Background Color");
  backgroundColorWasChanged();
  AddFrame(m_changeBackground, new TGLayoutHints(kLHintsExpandX, 2, 2, 5, 5));
  m_changeBackground->Connect("Clicked()", "CmsShowViewPopup", this, "changeBackground()");
  // save image
  m_saveImageButton = new TGTextButton(this, "Save Image ...");
  AddFrame(m_saveImageButton, new TGLayoutHints(kLHintsExpandX, 2, 2, 5, 5));
  m_saveImageButton->Connect("Clicked()", "CmsShowViewPopup", this, "saveImage()");

  // content frame
  AddFrame(new TGHorizontal3DLine(this), new TGLayoutHints(kLHintsExpandX, 0, 0, 5, 5));
  m_paramGUI = new ViewerParameterGUI(this);
  AddFrame(m_paramGUI, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

  SetWindowName("View Controller");
}

// CmsShowViewPopup::CmsShowViewPopup(const CmsShowViewPopup& rhs)
// {
//    // do actual copying here;
// }

CmsShowViewPopup::~CmsShowViewPopup() {}

void CmsShowViewPopup::reset(FWViewBase* vb, TEveWindow* ew) {
  m_viewBase = vb;
  m_eveWindow = ew;

  m_paramGUI->reset();

  // fill content
  if (m_viewBase) {
    m_saveImageButton->SetEnabled(kTRUE);
    m_viewLabel->SetText(m_viewBase->typeName().c_str());
    m_viewBase->populateController(*m_paramGUI);
    m_paramGUI->populateComplete();

    fMain = m_eveWindow->GetEveFrame();

    if (vb->typeId() >= FWViewType::kTable)
      m_saveImageButton->SetText("Print Text To Terminal");
    else
      m_saveImageButton->SetText("Save Image ...");
  } else {
    fMain = nullptr;
    m_viewLabel->SetText("No view selected");
    m_saveImageButton->SetEnabled(kFALSE);
  }

  MapSubwindows();
  Resize(GetDefaultSize());
  Layout();
  if (fMain) {
    CenterOnParent(kTRUE, TGTransientFrame::kTopRight);
  }
}

void CmsShowViewPopup::CloseWindow() {
  UnmapWindow();
  closed_.emit();
}

void CmsShowViewPopup::MapWindow() {
  TGWindow::MapWindow();
  m_mapped = true;
}

void CmsShowViewPopup::UnmapWindow() {
  TGWindow::UnmapWindow();
  m_mapped = false;
}

void CmsShowViewPopup::saveImage() {
  if (m_viewBase)
    m_viewBase->promptForSaveImageTo(this);
}

void CmsShowViewPopup::changeBackground() {
  m_colorManager->setBackgroundColorIndex(FWColorManager::kBlackIndex == m_colorManager->backgroundColorIndex()
                                              ? FWColorManager::kWhiteIndex
                                              : FWColorManager::kBlackIndex);
}

void CmsShowViewPopup::backgroundColorWasChanged() {
  if (FWColorManager::kBlackIndex == m_colorManager->backgroundColorIndex()) {
    m_changeBackground->SetText("Change Background Color to White");
  } else {
    m_changeBackground->SetText("Change Background Color to Black");
  }
}

//==============================================================================

ViewerParameterGUI::ViewerParameterGUI(const TGFrame* p)
    : TGCompositeFrame(p), m_tab(nullptr), m_selectedTabName("Style") {
  SetCleanup(kDeepCleanup);
}

void ViewerParameterGUI::reset() {
  // remember selected tab
  if (m_tab)
    m_selectedTabName = m_tab->GetCurrentTab()->GetString();
  else
    m_selectedTabName = "Style";

  // remove TGTab as the only child
  m_setters.clear();
  if (m_tab) {
    assert(GetList()->GetSize() == 1);
    TGFrameElement* el = (TGFrameElement*)GetList()->First();
    TGFrame* f = el->fFrame;

    assert(f == m_tab);
    f->UnmapWindow();
    RemoveFrame(f);
    f->DeleteWindow();
    m_tab = nullptr;
  }
}

ViewerParameterGUI& ViewerParameterGUI::requestTab(const char* name) {
  if (!m_tab) {
    m_tab = new TGTab(this);
    AddFrame(m_tab, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
  }

  if (!m_tab->GetTabContainer(name))
    m_tab->AddTab(name);

  m_tab->SetTab(name);

  return *this;
}

/* Add parameter setter in the current tab.*/
ViewerParameterGUI& ViewerParameterGUI::addParam(const FWParameterBase* param) {
  std::shared_ptr<FWParameterSetterBase> ptr(FWParameterSetterBase::makeSetterFor((FWParameterBase*)param));
  ptr->attach((FWParameterBase*)param, this);
  TGCompositeFrame* parent = m_tab->GetCurrentContainer();

  TGFrame* pframe = ptr->build(parent);
  parent->AddFrame(pframe, new TGLayoutHints(kLHintsExpandX));
  m_setters.push_back(ptr);

  pframe->MapWindow();
  pframe->MapSubwindows();
  pframe->Layout();
  parent->MapSubwindows();
  parent->Layout();
  m_tab->Layout();
  parent->Resize(parent->GetDefaultSize());
  return *this;
}

/* Add separator in current tab. */
ViewerParameterGUI& ViewerParameterGUI::separator() {
  assert(m_tab);
  TGHorizontal3DLine* s = new TGHorizontal3DLine(m_tab->GetCurrentContainer());
  m_tab->GetCurrentContainer()->AddFrame(s, new TGLayoutHints(kLHintsExpandX, 4, 4, 2, 2));

  return *this;
}

TGCompositeFrame* ViewerParameterGUI::getTabContainer() {
  assert(m_tab);
  return m_tab->GetCurrentContainer();
}

void ViewerParameterGUI::addFrameToContainer(TGCompositeFrame* x) {
  assert(m_tab);
  TGCompositeFrame* parent = m_tab->GetCurrentContainer();
  parent->AddFrame(x, new TGLayoutHints(kLHintsExpandX, 4, 4, 2, 2));

  parent->MapSubwindows();
  parent->Layout();
  m_tab->Layout();
  parent->Resize(parent->GetDefaultSize());
}

void ViewerParameterGUI::populateComplete() {
  // Set tab - same as it was before, if not exisiting select first time.
  if (m_tab) {
    bool x = m_tab->SetTab(m_selectedTabName.c_str(), false);
    if (!x)
      m_tab->SetTab("Style");
  }
}
