#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include <string>
int main(){
  cond::ServiceLoader* loader=new cond::ServiceLoader;
  loader->loadAuthenticationService(cond::XML);
  try{
    cond::MetaData metadata_svc("sqlite_file:pippo.db", *loader);
    std::string t1("token1");
    metadata_svc.addMapping("mytest1",t1);
    std::string t2("token2");
    metadata_svc.addMapping("mytest2",t2);
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
  delete loader;
}
