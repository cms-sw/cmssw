#ifndef CONDCORE_PLUGINSYSTEM_PROXYFACTORY_H
#define CONDCORE_PLUGINSYSTEM_PROXYFACTORY_H
// -*- C++ -*-
//
// Package:     PluginSystem
// Class  :     ProxyFactory
// 
/**\class ProxyFactory ProxyFactory.h CondCore/PluginSystem/interface/ProxyFactory.h
   
Description: <one line class summary>

Usage:
<usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 19:14:06 EDT 2005
// $Id$
//

// system include files

// user include files
#include "PluginManager/PluginFactory.h"

// forward declarations
namespace pool{
  class IDataSvc;
}

namespace edm {
  namespace eventsetup {
    class DataProxy;
  }
}

namespace cond{
  class ProxyFactory : public seal::PluginFactory< edm::eventsetup::DataProxy* ( pool::IDataSvc*, std::map<std::string,std::string>::iterator& ) >
  {
    
  public:
    ProxyFactory();
    virtual ~ProxyFactory();
    
    // ---------- const member functions ---------------------
    
    // ---------- static member functions --------------------
    static ProxyFactory* get();
    
    // ---------- member functions ---------------------------

  private:
    ProxyFactory( const ProxyFactory& ); // stop default
    
    const ProxyFactory& operator=( const ProxyFactory& ); // stop default
    
    // ---------- member data --------------------------------
    
  };
}

#endif /* CONDCORE_PLUGINSYSTEM_PROXYFACTORY_H */
