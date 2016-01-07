#ifndef CondCore_CondDB_CoralServiceManager_h
#define CondCore_CondDB_CoralServiceManager_h
#include "CoralKernel/IPluginManager.h"
#include <set>
#include <string>
//
// Package:     CondCore/CondDB
// Class  :     CoralServiceManager
// 
/**\class CoralServiceManager CoralServiceManager.h CondCore/CondDB/interface/CoralServiceManager.h

 Description: This is a bridge that implement coral::IPluginManager interface and internally uses edm plugin factory to create plugins of type coral::Service. 

 Usage: the plugin managed by this class must inherit from coral::Service interface. The pointer of CoralServiceManager should be passed to the coral::Context
*/
//
// Original Author:  Zhen Xie 
//         Created:  Wed Nov 12 10:48:50 CET 2008
// $Id $
//
namespace coral{
  class ILoadableComponent;
}
namespace cond{
  class CoralServiceManager : public coral::IPluginManager{
  public:
    virtual coral::ILoadableComponent* newComponent( const std::string& componentName );
    /// Returns the list of known components
    virtual std::set<std::string> knownPlugins() const;
    virtual ~CoralServiceManager(){}
  };
}
#endif
