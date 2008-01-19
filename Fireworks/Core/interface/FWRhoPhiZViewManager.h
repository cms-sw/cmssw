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
// $Id: FWRhoPhiZViewManager.h,v 1.2 2008/01/12 17:23:50 chrjones Exp $
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

struct FWRPZ3DModelProxy
{
   boost::shared_ptr<FWRPZDataProxyBuilder>   builder;
   TEveElementList*                           product; //owned by builder
   FWRPZ3DModelProxy():product(0){}
   FWRPZ3DModelProxy(boost::shared_ptr<FWRPZDataProxyBuilder> iBuilder):
    builder(iBuilder),product(0) {}
};

struct FWRPZ2DModelProxy
{
  boost::shared_ptr<FWRPZ2DDataProxyBuilder>   builder;
  TEveElementList*                             rhoPhiProduct; //owned by builder
  TEveElementList*                             rhoZProduct; //owned by builder
  FWRPZ2DModelProxy():rhoPhiProduct(0), rhoZProduct(0){}
  FWRPZ2DModelProxy(boost::shared_ptr<FWRPZ2DDataProxyBuilder> iBuilder):
  builder(iBuilder),rhoPhiProduct(0), rhoZProduct(0) {}
};


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

   private:
      FWRhoPhiZViewManager(const FWRhoPhiZViewManager&); // stop default

      const FWRhoPhiZViewManager& operator=(const FWRhoPhiZViewManager&); // stop default

      void addElements();
      // ---------- member data --------------------------------
      typedef  std::map<std::string,std::pair<std::string,bool> > TypeToBuilder;
      TypeToBuilder m_typeToBuilder;
      std::vector<FWRPZ3DModelProxy> m_3dmodelProxies;
      std::vector<FWRPZ2DModelProxy> m_2dmodelProxies;

      TEveElement* m_geom;
      TEveProjectionManager* m_rhoPhiProjMgr;
      TEveProjectionManager* m_rhoZProjMgr;
      std::vector<TEveElement*> m_rhoPhiGeom;
      std::vector<TEveElement*> m_rhoZGeom;
};


#endif
