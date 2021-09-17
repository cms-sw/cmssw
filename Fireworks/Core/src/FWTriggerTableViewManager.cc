// -*- C++ -*-
//
// Package:     Core
// Class  :     FWTriggerTableViewManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 22:01:27 EST 2008
//

// system include files
#include <cassert>
#include <iostream>
#include <functional>

// user include files

#include "Fireworks/Core/interface/FWTriggerTableViewManager.h"
#include "Fireworks/Core/interface/FWHLTTriggerTableView.h"
#include "Fireworks/Core/interface/FWL1TriggerTableView.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/FWTypeToRepresentations.h"
#include "Fireworks/Core/interface/FWJobMetadataManager.h"

FWTriggerTableViewManager::FWTriggerTableViewManager(FWGUIManager* iGUIMgr) : FWViewManagerBase() {
  FWGUIManager::ViewBuildFunctor f;
  f = std::bind(&FWTriggerTableViewManager::buildView, this, std::placeholders::_1, std::placeholders::_2);
  iGUIMgr->registerViewBuilder(FWViewType::idToName(FWViewType::kTableHLT), f);
  iGUIMgr->registerViewBuilder(FWViewType::idToName(FWViewType::kTableL1), f);
}

FWTriggerTableViewManager::~FWTriggerTableViewManager() {}

class FWViewBase* FWTriggerTableViewManager::buildView(TEveWindowSlot* iParent, const std::string& type) {
  std::shared_ptr<FWTriggerTableView> view;

  if (type == FWViewType::sName[FWViewType::kTableHLT])
    view.reset(new FWHLTTriggerTableView(iParent));
  else
    view.reset(new FWL1TriggerTableView(iParent));

  view->setProcessList(&(context().metadataManager()->processNamesInJob()));

  view->setBackgroundColor(colorManager().background());
  m_views.push_back(std::shared_ptr<FWTriggerTableView>(view));
  view->beingDestroyed_.connect(std::bind(&FWTriggerTableViewManager::beingDestroyed, this, std::placeholders::_1));
  return view.get();
}

void FWTriggerTableViewManager::beingDestroyed(const FWViewBase* iView) {
  for (std::vector<std::shared_ptr<FWTriggerTableView> >::iterator it = m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
    if (it->get() == iView) {
      m_views.erase(it);
      return;
    }
  }
}

void FWTriggerTableViewManager::colorsChanged() {
  for (std::vector<std::shared_ptr<FWTriggerTableView> >::iterator it = m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
    (*it)->setBackgroundColor(colorManager().background());
  }
}

void FWTriggerTableViewManager::eventEnd() {
  for (std::vector<std::shared_ptr<FWTriggerTableView> >::iterator it = m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
    (*it)->dataChanged();
  }
}

void FWTriggerTableViewManager::updateProcessList() {
  // printf("FWTriggerTableViewManager::updateProcessLi\n");
  for (std::vector<std::shared_ptr<FWTriggerTableView> >::iterator it = m_views.begin(), itEnd = m_views.end();
       it != itEnd;
       ++it) {
    (*it)->setProcessList(&(context().metadataManager()->processNamesInJob()));
    (*it)->resetCombo();
  }
}
