#ifndef Fireworks_MuonPUViewManager_h
#define Fireworks_MuonPUViewManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     MuonPUViewManager
//
/**\class MuonPUViewManager MuonPUViewManager.h Fireworks/Core/interface/MuonPUViewManager.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 22:01:21 EST 2008
// $Id: MuonPUViewManager.h,v 1.2 2008/06/09 19:59:52 chrjones Exp $
//

// system include files
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

// user include files
#include "Fireworks/Core/interface/FWViewManagerBase.h"
#include "Fireworks/Core/interface/MuonsProxyPUBuilder.h"

// forward declarations
class FWEventItem;
class TEveProjectionManager;
class TEveScene;
class TEveViewer;

struct MuonPUModelProxy {
  boost::shared_ptr<MuonsProxyPUBuilder>   builder;
  const FWEventItem                        *iItem;
  TEveElementList                          *product;
  MuonPUModelProxy () : product(0) { }
  MuonPUModelProxy (
		    boost::shared_ptr<MuonsProxyPUBuilder> iBuilder) :
    builder(iBuilder), product(0) { }
};

/*
class MuonPUViewManager : public FWViewManagerBase {

public:
     MuonPUViewManager();
     virtual ~MuonPUViewManager();

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
     MuonPUViewManager(const MuonPUViewManager&); // stop default
     const MuonPUViewManager& operator=(const MuonPUViewManager&); // stop default

     // ---------- member data --------------------------------
     typedef  std::map<std::string,std::string> TypeToBuilder;
     TypeToBuilder m_typeToBuilder;
     std::vector<MuonPUModelProxy> m_modelProxies;
     TEveProjectionManager* m_projMgr;
     TEveScene *ns;
     TEveViewer* nv;
};

*/
#endif
