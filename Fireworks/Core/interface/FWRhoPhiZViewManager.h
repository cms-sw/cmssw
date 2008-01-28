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
// $Id: FWRhoPhiZViewManager.h,v 1.6 2008/01/28 10:45:04 dmytro Exp $
//

// system include files
#include <boost/shared_ptr.hpp>
#include <vector>
#include <map>
#include <string>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"

// forward declarations
class TEveElement;
class TEveElementList;
class TEveProjectionManager;
class FWRPZDataProxyBuilder;
class FWRPZ2DDataProxyBuilder;

class FWRPZModelProxyBase
{
public:
   FWRPZModelProxyBase() {}
   virtual ~FWRPZModelProxyBase() {}
   void itemChanged(const FWEventItem*);
   virtual TEveElementList* getRhoPhiProduct() const =0;
   virtual TEveElementList* getRhoZProduct() const = 0;
   virtual void setRhoPhiProj(TEveElement*) = 0;
   virtual void setRhoZProj(TEveElement*) = 0;
private:
   virtual void itemChangedImp(const FWEventItem*) = 0;
   
};

class FWRPZ3DModelProxy : public FWRPZModelProxyBase
{
public:
   FWRPZ3DModelProxy():m_product(0),m_mustRebuild(true){}
   FWRPZ3DModelProxy(boost::shared_ptr<FWRPZDataProxyBuilder> iBuilder):
   m_builder(iBuilder),m_product(0),m_mustRebuild(true) {}
   TEveElementList* getRhoPhiProduct() const;
   TEveElementList* getRhoZProduct() const;
   void setRhoPhiProj(TEveElement*);
   void setRhoZProj(TEveElement*);
private:
   void itemChangedImp(const FWEventItem*) ;
   TEveElementList* getProduct() const;
   boost::shared_ptr<FWRPZDataProxyBuilder>   m_builder;
   mutable TEveElementList*                   m_product; //owned by builder
   mutable bool m_mustRebuild;
   
};

struct FWRPZ2DModelProxy : public FWRPZModelProxyBase
{
public:
   FWRPZ2DModelProxy():m_rhoPhiProduct(0), m_rhoZProduct(0),
   m_mustRebuildRhoPhi(true),m_mustRebuildRhoZ(true){}
   FWRPZ2DModelProxy(boost::shared_ptr<FWRPZ2DDataProxyBuilder> iBuilder):
   m_builder(iBuilder),m_rhoPhiProduct(0), m_rhoZProduct(0),
   m_mustRebuildRhoPhi(true),m_mustRebuildRhoZ(true){}

   TEveElementList* getRhoPhiProduct() const;
   TEveElementList* getRhoZProduct() const;
   void setRhoPhiProj(TEveElement*);
   void setRhoZProj(TEveElement*);
private:
   void itemChangedImp(const FWEventItem*) ;
   boost::shared_ptr<FWRPZ2DDataProxyBuilder>   m_builder;
   mutable TEveElementList*                     m_rhoPhiProduct; //owned by builder
   mutable TEveElementList*                     m_rhoZProduct; //owned by builder
   mutable bool m_mustRebuildRhoPhi;
   mutable bool m_mustRebuildRhoZ;
};

class TGeoHMatrix;
class TGeoShape;
class TEveGeoShapeExtract;

class FWRhoPhiZViewManager : public FWViewManagerBase
{

   public:
      FWRhoPhiZViewManager();
      //virtual ~FWRhoPhiZViewManager();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual void newEventAvailable();

      virtual void newItem(const FWEventItem*);

      void registerProxyBuilder(const std::string&, 
				const std::string&);

   protected:
      virtual void modelChangesComing() ;
      virtual void modelChangesDone() ;

   private:
      FWRhoPhiZViewManager(const FWRhoPhiZViewManager&); // stop default

      const FWRhoPhiZViewManager& operator=(const FWRhoPhiZViewManager&); // stop default

      void itemChanged(const FWEventItem*);
      void addElements();
   
      void setupGeometry();
      void makeMuonGeometryRhoPhi();
      void makeMuonGeometryRhoZ();
      void makeMuonGeometryRhoZAdvance();
      void estimateProjectionSizeDT( const TGeoHMatrix*, const TGeoShape*, double&, double&, double&, double& );
      void estimateProjectionSizeCSC( const TGeoHMatrix*, const TGeoShape*, double&, double&, double&, double& );
      void estimateProjectionSize( const Double_t*, double&, double&, double&, double& );
      TEveGeoShapeExtract* makeShapeExtract( const char*, double, double, double, double );

      // ---------- member data --------------------------------
      typedef  std::map<std::string,std::pair<std::string,bool> > TypeToBuilder;
      TypeToBuilder m_typeToBuilder;
      std::vector<boost::shared_ptr<FWRPZModelProxyBase> > m_modelProxies;

      TEveElement* m_geom;
      TEveProjectionManager* m_rhoPhiProjMgr;
      TEveProjectionManager* m_rhoZProjMgr;
      std::vector<TEveElement*> m_rhoPhiGeom;
      std::vector<TEveElement*> m_rhoZGeom;
   
      bool m_itemChanged;
};


#endif
