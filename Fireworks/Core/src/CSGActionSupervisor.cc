// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGActionSupervisor
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Aug 2009
//
#include <sigc++/sigc++.h>

#include "Fireworks/Core/interface/CSGActionSupervisor.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/fwLog.h"

// constructors and destructor
//
CSGActionSupervisor::CSGActionSupervisor() : m_tooltipDelay(3) {}

CSGActionSupervisor::~CSGActionSupervisor() {
  for (std::vector<CSGAction*>::iterator it = m_actionList.begin(), itEnd = m_actionList.end(); it != itEnd; ++it) {
    delete *it;
  }
}

CSGAction* CSGActionSupervisor::getAction(const std::string& name) {
  std::vector<CSGAction*>::iterator it_act;
  for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
    if ((*it_act)->getName() == name)
      return *it_act;
  }
  fwLog(fwlog::kWarning) << "No action is found with name " << name.c_str() << std::endl;
  return nullptr;
}

void CSGActionSupervisor::addToActionMap(CSGAction* action) { m_actionList.push_back(action); }

const std::vector<CSGAction*>& CSGActionSupervisor::getListOfActions() const { return m_actionList; }

void CSGActionSupervisor::defaultAction() { fwLog(fwlog::kInfo) << "Default action.\n"; }

void CSGActionSupervisor::enableActions(bool enable) {
  std::vector<CSGAction*>::iterator it_act;
  for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
    if (enable)
      (*it_act)->globalEnable();
    else
      (*it_act)->globalDisable();
  }
}

Bool_t CSGActionSupervisor::activateMenuEntry(int entry) {
  std::vector<CSGAction*>::iterator it_act;
  for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
    if (entry == (*it_act)->getMenuEntry()) {
      (*it_act)->activated.emit();
      return kTRUE;
    }
  }
  return kFALSE;
}

Bool_t CSGActionSupervisor::activateToolBarEntry(int entry) {
  std::vector<CSGAction*>::iterator it_act;
  for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
    if ((*it_act)->getToolBarData() && (*it_act)->getToolBarData()->fId == entry) {
      (*it_act)->activated.emit();
      return kTRUE;
    }
  }
  return kFALSE;
}

void CSGActionSupervisor::HandleMenu(int id) {}

void CSGActionSupervisor::resizeMenu(TGPopupMenu* menu) {
  std::vector<CSGAction*>::iterator it_act;
  for (it_act = m_actionList.begin(); it_act != m_actionList.end(); ++it_act) {
    if ((*it_act)->getMenu() == menu && (*it_act)->getKeycode() != 0) {
      (*it_act)->resizeMenuEntry();
    }
  }
}

Long_t CSGActionSupervisor::getToolTipDelay() const { return m_tooltipDelay; }
