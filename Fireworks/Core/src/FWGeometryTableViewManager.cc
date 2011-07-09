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
// $Id: FWGeometryTableViewManager.cc,v 1.1 2011/07/08 04:39:59 amraktad Exp $
//

#include <boost/bind.hpp>

#include "TFile.h"
#include "TSystem.h"

#include "Fireworks/Core/interface/FWGeometryTableViewManager.h"
#include "Fireworks/Core/interface/FWGeometryTableView.h"
#include "Fireworks/Core/interface/FWGUIManager.h"
#include "Fireworks/Core/interface/FWColorManager.h"
#include "Fireworks/Core/interface/fwLog.h"

FWGeometryTableViewManager::FWGeometryTableViewManager(FWGUIManager* iGUIMgr):
   FWViewManagerBase(),
   m_geoManager(0)
{
   FWGUIManager::ViewBuildFunctor f;
   f=boost::bind(&FWGeometryTableViewManager::buildView, this, _1, _2);                
   iGUIMgr->registerViewBuilder(FWViewType::idToName(FWViewType::kGeometryTable), f);
}

FWGeometryTableViewManager::~FWGeometryTableViewManager()
{ 
}


class FWViewBase*
FWGeometryTableViewManager::buildView(TEveWindowSlot* iParent, const std::string& /*type*/)
{
   boost::shared_ptr<FWGeometryTableView> view;
   if (!m_geoManager) loadGeometry();
   view.reset( new FWGeometryTableView(iParent, &colorManager(), m_geoManager));

   view->setBackgroundColor();
   m_views.push_back(boost::shared_ptr<FWGeometryTableView> (view));
   view->beingDestroyed_.connect(boost::bind(&FWGeometryTableViewManager::beingDestroyed, this,_1));
                                            
   return view.get();
}


void
FWGeometryTableViewManager::beingDestroyed(const FWViewBase* iView)
{
   for(std::vector<boost::shared_ptr<FWGeometryTableView> >::iterator it=m_views.begin(); it != m_views.end(); ++it) {
      if(it->get() == iView) {
         m_views.erase(it);
         return;
      }
   }
}

void
FWGeometryTableViewManager::colorsChanged()
{
  for(std::vector<boost::shared_ptr<FWGeometryTableView> >::iterator it=m_views.begin(); it != m_views.end(); ++it)
      (*it)->setBackgroundColor();
}

void
FWGeometryTableViewManager::loadGeometry()
{
   const char* defaultPath = Form("%s/cmsSimGeom-14.root",  gSystem->Getenv( "CMSSW_BASE" ));
   if( !gSystem->AccessPathName(defaultPath))
   {
      TFile* file = new TFile( defaultPath, "READ");
      try {
         if ( ! file )
            throw std::runtime_error("No root file.");
  
         file->ls();
      
         if ( !file->Get("cmsGeo;1"))
            throw std::runtime_error("Can't find TGeoManager object in selected file.");
         m_geoManager = (TGeoManager*) file->Get("cmsGeo;1");

      }
      catch (std::runtime_error &e)
      {
         fwLog(fwlog::kError) << "Failed to load simulation geomtery.\n";
      }
   }

}
