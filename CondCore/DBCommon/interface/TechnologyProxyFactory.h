#ifndef CondCore_DBCommon_TechnologyProxyFactory_h
#define CondCore_DBCommon_TechnologyProxyFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <memory>
#include <string>
namespace cond{
  typedef edmplugin::PluginFactory< cond::TechnologyProxy*() > TechnologyProxyFactory;

  inline std::auto_ptr<cond::TechnologyProxy> buildTechnologyProxy(const std::string&userconnect, 
								   const DbConnection& connection){
    std::string protocol;
    std::size_t pos=userconnect.find_first_of(':');
    if( pos!=std::string::npos ){
      protocol=userconnect.substr(0,pos);
      std::size_t p=protocol.find_first_of('_');
      if(p!=std::string::npos){
	 protocol=protocol.substr(0,p);
      }
    }else{
      throw cond::Exception(userconnect +":connection string format error");
    }
    //std::cout<<"userconnect "<<userconnect<<std::endl;
    //std::cout<<"protocol "<<protocol<<std::endl;  
    std::auto_ptr<cond::TechnologyProxy> ptr(cond::TechnologyProxyFactory::get()->create(protocol));
    (*ptr).initialize(userconnect,connection);
    return ptr;
  }
}
#endif
