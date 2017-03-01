#ifndef CondCore_CondDB_CoralServiceFactory_h
#define CondCore_CondDB_CoralServiceFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include <string>
//
// Package:     CondCore/CondDB
// Class  :     CoralServiceFactory
// 
/**\class CoralServiceFactory CoralServiceFactory.h CondCore/CondDB/interface/CoralServiceFactory.h

 Description: A special edm plugin factory that creates coral::Service 

 Usage: used internally by CoralServiceManager to create coral::Service as edm plugin
*/
//
// Original Author:  Zhen Xie 
//         Created:  Wed Nov 12 10:57:47 CET 2008
// $Id $
//
namespace coral{
  class Service;
}
namespace cond{
  typedef edmplugin::PluginFactory< coral::Service*(const std::string&) > CoralServicePluginFactory;
  
  class CoralServiceFactory{
  public:
    ~CoralServiceFactory();
    static CoralServiceFactory* get();
    coral::Service* create( const std::string& componentname) const;
 private:
    CoralServiceFactory();
  };
}
#endif
