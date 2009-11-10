#ifndef CondCore_DBCommon_TechnologyProxyFactory_h
#define CondCore_DBCommon_TechnologyProxyFactory_h
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include <memory>
#include <string>
namespace cond{
  typedef edmplugin::PluginFactory< cond::TechnologyProxy*(const std::string&) > TechnologyProxyFactory;

  std::auto_ptr<cond::TechnologyProxy> buildTechnologyProxy (cond::DBSession& session){
    const std::string & userconnect = session.connectionString(); 
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
    std::auto_ptr<cond::TechnologyProxy> ptr(cond::TechnologyProxyFactory::get()->create(protocol,session));
    return ptr;
  }



#endif
