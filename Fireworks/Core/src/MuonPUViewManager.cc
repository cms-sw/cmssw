// -*- C++ -*-
//
// Package:     Core
// Class  :     MuonPUViewManager
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Sun Jan  6 22:01:27 EST 2008
// $Id: MuonPUViewManager.cc,v 1.3 2008/11/06 22:05:26 amraktad Exp $
//

// system include files
#include <iostream>
#include "THStack.h"
#include "TCanvas.h"
#include "TVirtualHistPainter.h"
#include "TH2F.h"
#include "TView.h"
#include "TList.h"
#include "TEveManager.h"
#include "TEveViewer.h"
#include "TEveProjectionManager.h"
#include "TEveScene.h"
#include "TGLViewer.h"
#include "TClass.h"
#include "TColor.h"

// user include files
#include "Fireworks/Core/interface/MuonPUViewManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
#if defined(THIS_SHOULD_NOT_BE_USED)
MuonPUViewManager::MuonPUViewManager():
  FWViewManagerBase("ProxyPUBuilder")  // Change this to 3DBuilder?  I don't know.
{
     //setup projection
     nv = gEve->SpawnNewViewer("Muon");
     nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraPerspXOY);
     // nv->GetGLViewer()->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
     nv->GetGLViewer()->SetStyle(TGLRnrCtx::kOutline);
     nv->GetGLViewer()->SetClearColor(kBlack);

     ns = gEve->SpawnNewScene("Muon");
     nv->AddScene(ns);
//      m_projMgr = new TEveProjectionManager;
//      gEve->AddToListTree(m_projMgr, true);
//      gEve->AddElement(m_projMgr, ns);
//      gEve->Redraw3D(true);
     gEve->AddToListTree(ns, true);
     gEve->AddElement(ns);
    gEve->Redraw3D();
   // gEve->Redraw3D(true);
}

// MuonPUViewManager::MuonPUViewManager(const MuonPUViewManager& rhs)
// {
//    // do actual copying here;
// }

MuonPUViewManager::~MuonPUViewManager()
{
}

//
// assignment operators
//
// const MuonPUViewManager& MuonPUViewManager::operator=(const MuonPUViewManager& rhs)
// {
//   //An exception safe implementation is
//   MuonPUViewManager temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
MuonPUViewManager::newEventAvailable()
{
  Double_t rotation_center[3] = { 0, 0, 0 };

  for (std::vector<MuonPUModelProxy>::iterator proxy =
	 m_modelProxies.begin(); proxy != m_modelProxies.end(); ++proxy ) {
    proxy->builder->build( &(proxy->product) );
    proxy->builder->getCenter( rotation_center );
  }

   // set default view
   TGLViewer* viewer = nv->GetGLViewer();
   if ( !viewer ) printf("cannot get GLViewer\n");

   addElements();
}

void
MuonPUViewManager::newItem(const FWEventItem* iItem)
{
     TypeToBuilder::iterator itFind = m_typeToBuilder.find(iItem->name());
     if(itFind != m_typeToBuilder.end()) {
	  printf("MuonPUViewManager: adding item... ");
     	  MuonsProxyPUBuilder *builder =
	       reinterpret_cast<MuonsProxyPUBuilder *>(
		    createInstanceOf(
			 TClass::GetClass(typeid(MuonsProxyPUBuilder)),
			 itFind->second.c_str()));
	  if (builder != 0) {
	       printf("added\n");
	       boost::shared_ptr<MuonsProxyPUBuilder> pB( builder );
	       builder->setItem(iItem);
	       m_modelProxies.push_back(MuonPUModelProxy(pB) );  // Might have to change this to 3D as well.
	  } else printf("not added\n");
     }
}

void
MuonPUViewManager::registerProxyBuilder(const std::string& iType,
					    const std::string& iBuilder)
{
     m_typeToBuilder[iType] = iBuilder;
     printf("MuonPUViewManager: registering %s, %s\n", iType.c_str(),
	    iBuilder.c_str());
}

void
MuonPUViewManager::modelChangesComing()
{
}
void
MuonPUViewManager::modelChangesDone()
{
   newEventAvailable();
}

//
// const member functions
//

//
// static member functions
//

void MuonPUViewManager::addElements ()
{
     printf("Running addElements\n");
     //keep track of the last element added
//      TEveElement::List_i itLastElement = m_projMgr->BeginChildren();
//      bool rpHasMoreChildren = m_projMgr->GetNChildren();
//      int index = 0;
//      while(++index < m_projMgr->GetNChildren()) {++itLastElement;}

     for ( std::vector<MuonPUModelProxy>::iterator proxy =
		m_modelProxies.begin();
	   proxy != m_modelProxies.end(); ++proxy )  {
	  gEve->AddElement(proxy->product, ns);
// 	  if(proxy == m_modelProxies.begin()) {
// 	       if(rpHasMoreChildren) {
// 		    ++itLastElement;
// 	       }
// 	  } else {
// 	       ++itLastElement;
// 	  }
     }
}
#endif

