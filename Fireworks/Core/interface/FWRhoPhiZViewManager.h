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
// $Id$
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

struct FWRPZModelProxy
{
   boost::shared_ptr<FWRPZDataProxyBuilder>   builder;
   TEveElementList*                           product; //owned by builder
   FWRPZModelProxy():product(0){}
   FWRPZModelProxy(boost::shared_ptr<FWRPZDataProxyBuilder> iBuilder):
    builder(iBuilder),product(0) {}
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

      // ---------- member data --------------------------------
      typedef  std::map<std::string,std::string> TypeToBuilder;
      TypeToBuilder m_typeToBuilder;
      std::vector<FWRPZModelProxy> m_modelProxies;

      TEveElement* m_geom;
      TEveProjectionManager* m_rhoPhiProjMgr;

};


#endif
