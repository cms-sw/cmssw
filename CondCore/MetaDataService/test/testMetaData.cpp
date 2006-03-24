#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "SealKernel/Exception.h"
#include <string>
int main(){
  cond::ServiceLoader* loader=new cond::ServiceLoader;
  ::putenv("CORAL_AUTH_USER=cms_xiezhen_dev");
  ::putenv("CORAL_AUTH_PASSWORD=xiezhen123");
  loader->loadAuthenticationService(cond::Env);
  loader->loadMessageService(cond::Error);
  try{
    cond::MetaData metadata_svc("sqlite_file:pippo.db", *loader);
    //cond::MetaData metadata_svc("oracle://devdb10/cms_xiezhen_dev", *loader);
    metadata_svc.connect();
    //metadata_svc.getToken("mytest2");
    std::string t1("token1");
    metadata_svc.addMapping("mytest1",t1);
    std::string t2("token2");
    metadata_svc.addMapping("mytest2",t2);
    std::string tok1=metadata_svc.getToken("mytest2");
    std::cout<<"got token1 "<<tok1<<std::endl;
    std::string tok2=metadata_svc.getToken("mytest2");
    std::cout<<"got token2 "<<tok2<<std::endl;
    metadata_svc.disconnect();
  }catch(seal::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
  delete loader;
}


