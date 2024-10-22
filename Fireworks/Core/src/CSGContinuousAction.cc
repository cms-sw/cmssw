// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGContinuousAction
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Jul 29 10:21:18 EDT 2008
//

// system include files
#include <functional>
#include "TGMenu.h"

// user include files
#include "Fireworks/Core/interface/CSGContinuousAction.h"
#include "Fireworks/Core/interface/FWCustomIconsButton.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CSGContinuousAction::CSGContinuousAction(CSGActionSupervisor* iSupervisor, const char* iName)
    : CSGAction(iSupervisor, iName),
      m_upPic(nullptr),
      m_downPic(nullptr),
      m_disabledPic(nullptr),
      m_runningUpPic(nullptr),
      m_runningDownPic(nullptr),
      m_button(nullptr),
      m_isRunning(false) {
  activated.connect(std::bind(&CSGContinuousAction::switchMode, this));
}

void CSGContinuousAction::createCustomIconsButton(TGCompositeFrame* p,
                                                  const TGPicture* upPic,
                                                  const TGPicture* downPic,
                                                  const TGPicture* disabledPic,
                                                  const TGPicture* upRunningPic,
                                                  const TGPicture* downRunningPic,
                                                  TGLayoutHints* l,
                                                  Int_t id,
                                                  GContext_t norm,
                                                  UInt_t option) {
  m_upPic = upPic;
  m_downPic = downPic;
  m_disabledPic = disabledPic;
  m_runningUpPic = upRunningPic;
  m_runningDownPic = downRunningPic;
  m_button = CSGAction::createCustomIconsButton(p, upPic, downPic, disabledPic, l, id, norm, option);
}

void CSGContinuousAction::switchMode() {
  if (!m_isRunning) {
    m_isRunning = true;
    CSGAction::globalEnable();
    if (getToolBar() && !m_runningImageFileName.empty()) {
      getToolBar()->ChangeIcon(getToolBarData(), m_runningImageFileName.c_str());
    }
    if (nullptr != m_button) {
      const TGPicture* tUp = m_runningUpPic;
      const TGPicture* tDown = m_runningDownPic;
      m_button->swapIcons(tUp, tDown, m_disabledPic);
    }
    if (nullptr != getMenu()) {
      getMenu()->CheckEntry(getMenuEntry());
    }
    started_();
  } else {
    stop();
    stopped_();
  }
}

void CSGContinuousAction::stop() {
  m_isRunning = false;
  if (getToolBar() && !m_imageFileName.empty()) {
    getToolBar()->ChangeIcon(getToolBarData(), m_imageFileName.c_str());
  }
  if (nullptr != m_button) {
    const TGPicture* tUp = m_upPic;
    const TGPicture* tDown = m_downPic;

    m_button->swapIcons(tUp, tDown, m_disabledPic);
  }
  if (nullptr != getMenu()) {
    getMenu()->UnCheckEntry(getMenuEntry());
  }
}

void CSGContinuousAction::globalEnable() { CSGAction::globalEnable(); }

void CSGContinuousAction::globalDisable() {
  if (!m_isRunning) {
    CSGAction::globalDisable();
  }
}

//
// const member functions
//

//
// static member functions
//
