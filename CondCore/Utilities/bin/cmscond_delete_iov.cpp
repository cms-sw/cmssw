#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include <boost/program_options.hpp>
#include <iostream>
int main( int argc, char** argv ){
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_delete_iov [options] \n");
  visible.add_options()
    ("connect,c",boost::program_options::value<std::string>(),"connection string(required)")
    ("user,u",boost::program_options::value<std::string>(),"user name (default \"\")")
    ("pass,p",boost::program_options::value<std::string>(),"password (default \"\")")
    //("catalog,f",boost::program_options::value<std::string>(),"file catalog contact string (default $POOL_CATALOG)")
    ("authPath,P",boost::program_options::value<std::string>(),"path to authentication.xml")
    ("all,a","delete all tags")
    ("tag,t",boost::program_options::value<std::string>(),"delete the specified tag and IOV")
    ("withPayload","delete payload data associated with the specified tag (default off)")
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
  std::string authPath("");
  std::string user("");
  std::string pass("");
  bool deleteAll=true;
  bool debug=false;
  bool withPayload=false;
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
  if( vm.count("authPath") ){
    authPath=vm["authPath"].as<std::string>();
  }
  /*if(vm.count("catalog")){
    catalog=vm["catalog"].as<std::string>();
    }*/
  if(vm.count("tag")){
    tag=vm["tag"].as<std::string>();
    deleteAll=false;
  }
  if(vm.count("withPayload")){
    withPayload=true;
  }
  if(vm.count("debug")){
    debug=true;
  }
  
  cond::DBSession* session=new cond::DBSession;
  if( !authPath.empty() ){
    session->configuration().setAuthenticationMethod( cond::XML );
  }else{
    session->configuration().setAuthenticationMethod( cond::Env );
  }
  if(debug){
    session->configuration().setMessageLevel( cond::Debug );
  }else{
    session->configuration().setMessageLevel( cond::Error );
  }
  session->configuration().connectionConfiguration()->setConnectionRetrialTimeOut( 600 );
  session->configuration().connectionConfiguration()->enableConnectionSharing();
  session->configuration().connectionConfiguration()->enableReadOnlySessionOnUpdateConnections(); 
  std::string userenv(std::string("CORAL_AUTH_USER=")+user);
  std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
  std::string authenv(std::string("CORAL_AUTH_PATH=")+authPath);
  ::putenv(const_cast<char*>(userenv.c_str()));
  ::putenv(const_cast<char*>(passenv.c_str()));
  ::putenv(const_cast<char*>(authenv.c_str()));
  //std::string catalog("pfncatalog_memory://POOL_RDBMS?");
  //catalog.append(connect);
  static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
  conHandler.registerConnection("mydb",connect,0);
  session->open();
  if( deleteAll ){
    try{
      cond::PoolTransaction& pooldb=conHandler.getConnection("mydb")->poolTransaction(false);
      cond::IOVService iovservice(pooldb);
      pooldb.start();
      iovservice.deleteAll(withPayload);
      pooldb.commit();
      cond::CoralTransaction& coraldb=conHandler.getConnection("mydb")->coralTransaction(false);
      cond::MetaData metadata_svc(coraldb);
      coraldb.start();
      metadata_svc.deleteAllEntries();
      coraldb.commit();
      return 0;
    }catch(cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(std::exception& er){
      std::cout<<er.what()<<std::endl;
    }
  }else{
    try{
      cond::CoralTransaction& coraldb=conHandler.getConnection("mydb")->coralTransaction(true);
      cond::MetaData metadata_svc(coraldb);
      coraldb.start();
      std::string token=metadata_svc.getToken(tag);
      if( token.empty() ) {
	std::cout<<"non-existing tag "<<tag<<std::endl;
	return 11;
      }
      coraldb.commit();      
      cond::PoolTransaction& pooldb=conHandler.getConnection("mydb")->poolTransaction(false);
      cond::IOVService iovservice(pooldb);
      cond::IOVEditor* ioveditor=iovservice.newIOVEditor(token);
      pooldb.start();
      ioveditor->deleteEntries(withPayload);
      pooldb.commit();
      coraldb.start();
      metadata_svc.deleteEntryByTag(tag);
      coraldb.commit();
      delete ioveditor;
    }catch(cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(std::exception& er){
      std::cout<<er.what()<<std::endl;
    }
  }
  delete session;
  return 0;
}
