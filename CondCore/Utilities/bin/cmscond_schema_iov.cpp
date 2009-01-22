#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/Utilities/interface/CommonOptions.h"
#include <cstdlib>
//#include <boost/program_options.hpp>
#include <iostream>
int main( int argc, char** argv ){
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::CommonOptions myopt("cmscond_schema_iov");
  myopt.addConnect();
  myopt.addAuthentication(true);
  myopt.visibles().add_options()
    ("create","create iov schema")
    ("drop","drop iov schema")
    ("truncate","truncate iov schema")
    ;
  myopt.description().add( myopt.visibles() );
  boost::program_options::variables_map vm;
  try{
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(myopt.description()).run(), vm);
    boost::program_options::notify(vm);
  }catch(const boost::program_options::error& er) {
    std::cerr << er.what()<<std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout <<myopt.visibles() <<std::endl;;
    return 0;
  }
  std::string connect;
  std::string authPath("");
  std::string user("");
  std::string pass("");
  bool debug=false;
  bool dropSchema=false;
  bool createSchema=false;
  bool truncateSchema=false;
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
  if(vm.count("create")){
    createSchema=true;
  }
  if(vm.count("drop")){
    dropSchema=true;
  }
  if(vm.count("debug")){
    debug=true;
  }
  cond::DBSession* session=new cond::DBSession;
  std::string userenv(std::string("CORAL_AUTH_USER=")+user);
  std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
  ::putenv(const_cast<char*>(userenv.c_str()));
  ::putenv(const_cast<char*>(passenv.c_str()));
  if( !authPath.empty() ){
    session->configuration().setAuthenticationMethod( cond::XML );
    session->configuration().setAuthenticationPath(authPath);
  }else{
    session->configuration().setAuthenticationMethod( cond::Env );    
  }
  if(debug){
    session->configuration().setMessageLevel( cond::Debug );
  }else{
    session->configuration().setMessageLevel( cond::Error );
  }
  cond::Connection con(connect,500);
  session->open();
  con.connect(session);
  if( createSchema ){
    try{
      cond::CoralTransaction& coraldb=con.coralTransaction();
      coraldb.start(false);
      cond::IOVSchemaUtility ut(coraldb);
      ut.create();
      coraldb.commit();
      return 0;
    }catch(cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(std::exception& er){
      std::cout<<er.what()<<std::endl;
    }
  }
  if( dropSchema ){
    try{
      cond::CoralTransaction& coraldb=con.coralTransaction();
      coraldb.start(false);
      cond::IOVSchemaUtility ut(coraldb);
      ut.drop();
      coraldb.commit();      
      return 0;
    }catch(cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(std::exception& er){
      std::cout<<er.what()<<std::endl;
    }
  }
  if( truncateSchema ){
    try{
      cond::CoralTransaction& coraldb=con.coralTransaction();
      coraldb.start(false);
      cond::IOVSchemaUtility ut(coraldb);
      ut.truncate();
      coraldb.commit();      
      return 0;
    }catch(cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(std::exception& er){
      std::cout<<er.what()<<std::endl;
    }
  }
  con.disconnect();
  delete session;
  return 0;
}
