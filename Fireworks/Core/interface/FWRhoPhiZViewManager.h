#ifndef Fireworks_Core_FWRhoPhiZViewManager_h
#define Fireworks_Core_FWRhoPhiZViewManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRhoPhiZViewManager
//
/**\class FWRhoPhiZViewManager FWRhoPhiZViewManager.h Fireworks/Core/interface/FWRhoPhiZViewManager.h

   Description: Manages the data and views for Rho/Phi and Rho/Z Views

   Usage:
    <usage>

 */
//
// Original Author:
//         Created:  Sat Jan  5 11:27:34 EST 2008
// $Id: FWRhoPhiZViewManager.h,v 1.29 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include <boost/shared_ptr.hpp>
#include <vector>
#include <map>
#include <string>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/FWEvePtr.h"

// forward declarations
class TEveElement;
class TEveElementList;
class TEveProjectionManager;
class TGLEmbeddedViewer;
class TEvePad;
class FWRPZDataProxyBuilderBase;


class TGeoHMatrix;
class TGeoShape;
class TEveGeoShape;
class FWGUIManager;
class FWRhoPhiZView;
class FWViewBase;
class TEveSelection;
class TEveElement;
class TEveWindowSlot;
class FWSelectionManager;

class FWRhoPhiZViewManager : public FWViewManagerBase
{

public:
   FWRhoPhiZViewManager(FWGUIManager*);
   virtual ~FWRhoPhiZViewManager();

   // ---------- const member functions ---------------------
   FWTypeToRepresentations supportedTypesAndRepresentations() const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   virtual void newItem(const FWEventItem*);

   FWViewBase* createRhoPhiView(TEveWindowSlot* iParent);
   FWViewBase* createRhoZView(TEveWindowSlot* iParent);

   //connect to ROOT signals
   void selectionAdded(TEveElement*);
   void selectionRemoved(TEveElement*);
   void selectionCleared();

protected:
   virtual void modelChangesComing() ;
   virtual void modelChangesDone() ;

private:
   FWRhoPhiZViewManager(const FWRhoPhiZViewManager&);    // stop default

   const FWRhoPhiZViewManager& operator=(const FWRhoPhiZViewManager&);    // stop default

   void makeProxyBuilderFor(const FWEventItem* iItem);

   void beingDestroyed(const FWViewBase*);

   void setupGeometry();
   void makeMuonGeometryRhoPhi();
   void makeMuonGeometryRhoZ();
   void makeMuonGeometryRhoZAdvance();
   void makeTrackerGeometryRhoPhi();
   void makeTrackerGeometryRhoZ();
   void estimateProjectionSizeDT( const TGeoHMatrix*, const TGeoShape*, double&, double&, double&, double& );
   void estimateProjectionSizeCSC( const TGeoHMatrix*, const TGeoShape*, double&, double&, double&, double& );
   void estimateProjectionSize( const Double_t*, double&, double&, double&, double& );
   TEveGeoShape* makeShape( const char*, double, double, double, double );


   // ---------- member data --------------------------------
   typedef  std::map<std::string,std::pair<std::string,bool> > TypeToBuilder;
   TypeToBuilder m_typeToBuilder;
   std::vector<boost::shared_ptr<FWRPZDataProxyBuilderBase> > m_builders;

   FWEvePtr<TEveProjectionManager> m_rhoPhiGeomProjMgr;
   FWEvePtr<TEveProjectionManager> m_rhoZGeomProjMgr;
   std::vector<TEveElement*> m_rhoPhiGeom;
   std::vector<TEveElement*> m_rhoZGeom;

   std::vector<boost::shared_ptr<FWRhoPhiZView> > m_rhoPhiViews;
   std::vector<boost::shared_ptr<FWRhoPhiZView> > m_rhoZViews;

   TEveElementList* m_eveStore;

   TEveSelection* m_eveSelection;

   FWSelectionManager* m_selectionManager;
   bool m_isBeingDestroyed;
};


#endif
