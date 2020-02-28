// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGeometryTableViewManager
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel
//         Created:  Fri Jul  8 00:40:37 CEST 2011
//

#include <boost/bind.hpp>

#include "TFile.h"
#include "TSystem.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TEveManager.h"

#include "Fireworks/Core/interface/FWGeometryTableViewManager.h"
#include "Fireworks/Core/src/FWGeometryTableView.h"
#include "Fireworks/Core/src/FWOverlapTableView.h"
#include "Fireworks/Core/interface/FWGeometry.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/fwLog.h"

TGeoManager* FWGeometryTableViewManager::s_geoManager = nullptr;

TGeoManager* FWGeometryTableViewManager_GetGeoManager() { return FWGeometryTableViewManager::getGeoMangeur(); }

FWGeometryTableViewManager::FWGeometryTableViewManager(FWGUIManager* iGUIMgr, std::string fileName, std::string geoName)
    : FWViewManagerBase(), m_fileName(fileName), m_TGeoName(geoName) {
  FWGUIManager::ViewBuildFunctor f;
  f = boost::bind(&FWGeometryTableViewManager::buildView, this, _1, _2);
  iGUIMgr->registerViewBuilder(FWViewType::idToName(FWViewType::kGeometryTable), f);
  iGUIMgr->registerViewBuilder(FWViewType::idToName(FWViewType::kOverlapTable), f);
}

FWGeometryTableViewManager::~FWGeometryTableViewManager() {}

FWViewBase* FWGeometryTableViewManager::buildView(TEveWindowSlot* iParent, const std::string& type) {
  if (!s_geoManager)
    setGeoManagerFromFile();
  std::shared_ptr<FWGeometryTableViewBase> view;

  FWViewType::EType typeId =
      (type == FWViewType::sName[FWViewType::kGeometryTable]) ? FWViewType::kGeometryTable : FWViewType::kOverlapTable;
  if (typeId == FWViewType::kGeometryTable)
    view.reset(new FWGeometryTableView(iParent, &colorManager()));
  else
    view.reset(new FWOverlapTableView(iParent, &colorManager()));

  view->setBackgroundColor();
  m_views.push_back(std::shared_ptr<FWGeometryTableViewBase>(view));
  view->beingDestroyed_.connect(boost::bind(&FWGeometryTableViewManager::beingDestroyed, this, _1));

  return view.get();
}

void FWGeometryTableViewManager::beingDestroyed(const FWViewBase* iView) {
  for (std::vector<std::shared_ptr<FWGeometryTableViewBase> >::iterator it = m_views.begin(); it != m_views.end();
       ++it) {
    if (it->get() == iView) {
      m_views.erase(it);
      return;
    }
  }
}

void FWGeometryTableViewManager::colorsChanged() {
  for (std::vector<std::shared_ptr<FWGeometryTableViewBase> >::iterator it = m_views.begin(); it != m_views.end(); ++it)
    (*it)->setBackgroundColor();
}

//______________________________________________________________________________
TGeoManager* FWGeometryTableViewManager::getGeoMangeur() {
  // Function used in geometry table views.

  assert(s_geoManager);
  return s_geoManager;
}

//______________________________________________________________________________
void FWGeometryTableViewManager::setGeoManagerRuntime(TGeoManager* x) {
  // Function called from FWFFLooper to set geometry created in runtime.

  s_geoManager = x;
}

//______________________________________________________________________________
void FWGeometryTableViewManager::setGeoManagerFromFile() {
  TFile* file = FWGeometry::findFile(m_fileName.c_str());
  fwLog(fwlog::kInfo) << "Geometry table file: " << m_fileName.c_str() << std::endl;
  try {
    if (!file) {
      // Try it as a GDML file
      s_geoManager = TGeoManager::Import(m_fileName.c_str(), m_TGeoName.c_str());
    } else {
      file->ls();
      s_geoManager = (TGeoManager*)file->Get(m_TGeoName.c_str());
    }
    if (!s_geoManager)
      throw std::runtime_error("Can't find TGeoManager object in selected file.");

  } catch (std::runtime_error& e) {
    fwLog(fwlog::kError) << e.what();
    exit(0);
  }
}
