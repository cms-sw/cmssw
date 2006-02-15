#include "CondCore/MetaDataService/interface/MetaData.h"
#include "SealKernel/Exception.h"
#include <string>
int main(){
  try{
  cond::MetaData metadata_svc("sqlite_file:pippo.db");
  std::string t1("token1");
  metadata_svc.addMapping("mytest1",t1);
  std::string t2("token2");
  metadata_svc.addMapping("mytest2",t2);
  }catch(seal::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(...){
    std::cout<<"Funny error"<<std::endl;
  }
}
