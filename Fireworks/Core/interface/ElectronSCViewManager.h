#ifndef Fireworks_ElectronSCViewManager_h
#define Fireworks_ElectronSCViewManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     ElectronSCViewManager
//
/**\class ElectronSCViewManager ElectronSCViewManager.h Fireworks/Core/interface/ElectronSCViewManager.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 22:01:21 EST 2008
// $Id: ElectronSCViewManager.h,v 1.3 2008/06/09 19:59:33 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/ElectronsProxySCBuilder.h"

// forward declarations
class FWEventItem;
class TEveProjectionManager;

/*
struct ElectronSCModelProxy {
     boost::shared_ptr<ElectronsProxySCBuilder>   builder;
     TEveElementList                           *product;
     ElectronSCModelProxy () : product(0) { }
     ElectronSCModelProxy (
	  boost::shared_ptr<ElectronsProxySCBuilder> iBuilder) :
	  builder(iBuilder), product(0) { }
};

class ElectronSCViewManager : public FWViewManagerBase {

public:
     ElectronSCViewManager();
     virtual ~ElectronSCViewManager();

     // ---------- const member functions ---------------------

     // ---------- static member functions --------------------

     // ---------- member functions ---------------------------
     virtual void newEventAvailable();
     virtual void newItem(const FWEventItem*);
     void registerProxyBuilder(const std::string&,
			       const std::string&);
     void addElements ();

protected:
     virtual void modelChangesComing();
     virtual void modelChangesDone();

private:
     ElectronSCViewManager(const ElectronSCViewManager&); // stop default
     const ElectronSCViewManager& operator=(const ElectronSCViewManager&); // stop default

     // ---------- member data --------------------------------
     typedef  std::map<std::string,std::string> TypeToBuilder;
     TypeToBuilder m_typeToBuilder;
     std::vector<ElectronSCModelProxy> m_modelProxies;
     TEveProjectionManager* m_projMgr;
     TEveScene *ns;
     TEveViewer* nv;
};

*/
#endif
