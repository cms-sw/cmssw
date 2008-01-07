// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRhoPhiZViewManager
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Sat Jan  5 14:08:51 EST 2008
// $Id$
//

// system include files
#include <stdexcept>
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TEveBrowser.h"
#include "TEveGeoNode.h"
#include "TSystem.h"
#include "TEveProjectionManager.h"
#include "TEveScene.h"
#include "TGLViewer.h"
#include "TClass.h"
#include "TFile.h"
#include "TEveGeoShapeExtract.h"

#include <iostream>
#include <exception>

// user include files
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FWRPZDataProxyBuilder.h"
#include "Fireworks/Core/interface/FWEventItem.h"

//
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWRhoPhiZViewManager::FWRhoPhiZViewManager():
  FWViewManagerBase("Proxy3DBuilder"),
  m_geom(0),
  m_rhoPhiProjMgr(0)
{
  //setup projection
  TEveViewer* nv = gEve->SpawnNewViewer("Rho Phi");
  nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
  TEveScene* ns = gEve->SpawnNewScene("Rho Phi");
  nv->AddScene(ns);

  m_rhoPhiProjMgr = new TEveProjectionManager;
  gEve->AddToListTree(m_rhoPhiProjMgr,kTRUE);
  gEve->AddElement(m_rhoPhiProjMgr,ns);


  //kTRUE tells it to reset the camera so we see everything 
  gEve->Redraw3D(kTRUE);  
}

// FWRhoPhiZViewManager::FWRhoPhiZViewManager(const FWRhoPhiZViewManager& rhs)
// {
//    // do actual copying here;
// }

//FWRhoPhiZViewManager::~FWRhoPhiZViewManager()
//{
//}

//
// assignment operators
//
// const FWRhoPhiZViewManager& FWRhoPhiZViewManager::operator=(const FWRhoPhiZViewManager& rhs)
// {
//   //An exception safe implementation is
//   FWRhoPhiZViewManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWRhoPhiZViewManager::newEventAvailable()
{
  using namespace std;
  if(0==gEve) {
    cout <<"Eve not initialized"<<endl;
    return;
  }

  {
     //while inside this scope, do not let
     // Eve do any redrawing
     TEveManager::TRedrawDisabler disableRedraw(gEve);

     // build models
     for ( std::vector<FWRPZModelProxy>::iterator proxy = m_modelProxies.begin();
	   proxy != m_modelProxies.end(); ++proxy )
       proxy->builder->build(&(proxy->product) );
     
     // R-Phi projections
     
     // setup the projection
     // each projection knows what model proxies it needs
     // NOTE: this should be encapsulated and made configurable 
     //       somewhere else.
     m_rhoPhiProjMgr->DestroyElements();
     
	  
     // FIXME - standard way of loading geomtry failed
     // ----------- from here 
     if ( ! m_geom ) {
	TFile f("tracker.root");
	if(not f.IsOpen()) {
	   std::cerr <<"failed to open 'tracker.root'"<<std::endl;
	   throw std::runtime_error("Failed to open 'tracker.root' geometry file");
	}
	TEveGeoShapeExtract* gse = dynamic_cast<TEveGeoShapeExtract*>(f.Get("Tracker"));
	TEveGeoShape* gsre = TEveGeoShape::ImportShapeExtract(gse,0);
	f.Close();
	m_geom = gsre;
     }
     // ---------- to here
     
     m_rhoPhiProjMgr->ImportElements(m_geom);
     for ( std::vector<FWRPZModelProxy>::iterator proxy = m_modelProxies.begin();
	   proxy != m_modelProxies.end(); ++proxy )  {
       m_rhoPhiProjMgr->ImportElements(proxy->product);
     }  
  }
}

void 
FWRhoPhiZViewManager::newItem(const FWEventItem* iItem)
{
  TypeToBuilder::iterator itFind = m_typeToBuilder.find(iItem->name());
  if(itFind != m_typeToBuilder.end()) {
    FWRPZDataProxyBuilder* builder = reinterpret_cast<
      FWRPZDataProxyBuilder*>( 
        createInstanceOf(TClass::GetClass(typeid(FWRPZDataProxyBuilder)),
			 itFind->second.c_str())
	);
    if(0!=builder) {
      boost::shared_ptr<FWRPZDataProxyBuilder> pB( builder );
      builder->setItem(iItem);
      m_modelProxies.push_back(FWRPZModelProxy(pB) );
    }
  }
}

void 
FWRhoPhiZViewManager::registerProxyBuilder(const std::string& iType,
					   const std::string& iBuilder)
{
  m_typeToBuilder[iType]=iBuilder;
}

//
// const member functions
//

//
// static member functions
//
