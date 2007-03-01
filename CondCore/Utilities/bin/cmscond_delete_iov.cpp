#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include <boost/program_options.hpp>
#include <iostream>
int main( int argc, char** argv ){
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_delete_iov [options] \n");
  visible.add_options()
    ("connect,c",boost::program_options::value<std::string>(),"connection string(required)")
    ("user,u",boost::program_options::value<std::string>(),"user name (default \"\")")
    ("pass,p",boost::program_options::value<std::string>(),"password (default \"\")")
    ("catalog,f",boost::program_options::value<std::string>(),"file catalog contact string (default $POOL_CATALOG)")
    ("all,a","delete all tags")
    ("tag,t",boost::program_options::value<std::string>(),"delete the specified tag and IOV")
    ("debug,d","switch on debug mode")
    ("help,h", "help message")
    ;
  desc.add(visible);
  boost::program_options::variables_map vm;
  try{
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    boost::program_options::notify(vm);
  }catch(const boost::program_options::error& er) {
    std::cerr << er.what()<<std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout << visible <<std::endl;;
    return 0;
  }
  std::string connect;
  std::string catalog("file:PoolFileCatalog.xml");
  std::string user("");
  std::string pass("");
  bool deleteAll=true;
  bool debug=false;
  std::string tag;
  if(!vm.count("connect")){
    std::cerr <<"[Error] no connect[c] option given \n";
    std::cerr<<" please do "<<argv[0]<<" --help \n";
    return 1;
  }else{
    connect=vm["connect"].as<std::string>();
  }
  if(vm.count("user")){
    user=vm["user"].as<std::string>();
  }
  if(vm.count("pass")){
    pass=vm["pass"].as<std::string>();
  }
  if(vm.count("catalog")){
    catalog=vm["catalog"].as<std::string>();
  }
  if(vm.count("tag")){
    tag=vm["tag"].as<std::string>();
    deleteAll=false;
  }
  if(vm.count("debug")){
    debug=true;
  }
  cond::DBSession* session=new cond::DBSession(true);
  session->sessionConfiguration().setAuthenticationMethod( cond::Env );
  if(debug){
    session->sessionConfiguration().setMessageLevel( cond::Debug );
  }else{
    session->sessionConfiguration().setMessageLevel( cond::Error );
  }
  session->connectionConfiguration().setConnectionRetrialTimeOut( 600 );
  session->connectionConfiguration().enableConnectionSharing();
  session->connectionConfiguration().enableReadOnlySessionOnUpdateConnections(); 
  std::string userenv(std::string("CORAL_AUTH_USER=")+user);
  std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
  ::putenv(const_cast<char*>(userenv.c_str()));
  ::putenv(const_cast<char*>(passenv.c_str()));
  if( deleteAll ){
    try{
      session->open();
      cond::PoolStorageManager pooldb(connect,catalog,session);
      cond::IOVService iovservice(pooldb);
      pooldb.connect();
      pooldb.startTransaction(false);
      iovservice.deleteAll();
      pooldb.commit();
      pooldb.disconnect();
      cond::RelationalStorageManager coraldb(connect,session);
      cond::MetaData metadata_svc(coraldb);
      coraldb.connect(cond::ReadWrite);
      coraldb.startTransaction(false);
      metadata_svc.deleteAllEntries();
      coraldb.commit();
      coraldb.disconnect();
      session->close();
      return 0;
    }catch(cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(std::exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(...){
      std::cout<<"Unknown error"<<std::endl;
    }
  }else{
    try{
      session->open();
      cond::RelationalStorageManager coraldb(connect,session);
      cond::MetaData metadata_svc(coraldb);
      coraldb.connect(cond::ReadOnly);
      coraldb.startTransaction(true);
      std::string token=metadata_svc.getToken(tag);
      if( token.empty() ) {
	std::cout<<"non-existing tag "<<tag<<std::endl;
	return 11;
      }
      coraldb.commit();
      coraldb.disconnect();
      cond::PoolStorageManager pooldb(connect,catalog,session);
      cond::IOVService iovservice(pooldb);
      cond::IOVEditor* ioveditor=iovservice.newIOVEditor(token);
      pooldb.connect();
      pooldb.startTransaction(false);
      ioveditor->deleteEntries();
      pooldb.commit();
      pooldb.disconnect();
      coraldb.connect(cond::ReadWrite);
      coraldb.startTransaction(false);
      metadata_svc.deleteEntryByTag(tag);
      coraldb.commit();
      coraldb.disconnect();
      delete ioveditor;
      session->close();
    }catch(cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(std::exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(...){
      std::cout<<"Unknown error"<<std::endl;
    }
  }
  delete session;
  return 0;
}
