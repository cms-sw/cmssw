#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/SharedLibraryName.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/Utilities/interface/CommonOptions.h"
#include <boost/program_options.hpp>
#include <iostream>

int main( int argc, char** argv ){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::CommonOptions myopt("cmscond_truncate_iov");
  myopt.addConnect();
  myopt.addAuthentication(true);
  myopt.addDictionary();
  myopt.visibles().add_options()
     ("tag,t",boost::program_options::value<std::string>(),"remove last entry from the specified tag")
    ("withPayload","delete payload data associated with the entry removed (default off)")
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
    std::cout << myopt.visibles() <<std::endl;;
    return 0;
  }
  std::string connect;
  std::string authPath("");
  std::string user("");
  std::string pass("");
  bool deleteAll=true;
  bool debug=false;
  bool withPayload=false;
  std::string dictionary;
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
  if(vm.count("tag")){
    tag=vm["tag"].as<std::string>();
   }
  if(vm.count("withPayload")){
    withPayload=true;
    if(vm.count("dictionary")){
      dictionary=vm["dictionary"].as<std::string>();
    }
  }
  if(vm.count("debug")){
    debug=true;
  }
  
  cond::SharedLibraryName s;
  if (!dictionary.empty())
    try {
      edmplugin::SharedLibrary( s(dictionary) );
    }catch ( const cms::Exception& er ) {
      throw std::runtime_error( er.what() );
    }
  cond::DBSession* session=new cond::DBSession;
  session->configuration().connectionConfiguration()->disablePoolAutomaticCleanUp();
  session->configuration().connectionConfiguration()->setConnectionTimeOut(0);

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
  cond::Connection con(connect,-1);
  session->open();
  con.connect(session);
  try{
    cond::CoralTransaction& coraldb=con.coralTransaction();
    cond::MetaData metadata_svc(coraldb);
    coraldb.start(true);
    std::string token=metadata_svc.getToken(tag);
    if( token.empty() ) {
      std::cout<<"non-existing tag "<<tag<<std::endl;
      return 11;
    }
    coraldb.commit();      
    cond::PoolTransaction& pooldb=con.poolTransaction();
    cond::IOVService iovservice(pooldb);
    cond::IOVEditor* ioveditor=iovservice.newIOVEditor(token);
    pooldb.start(false);
    ioveditor->truncate(withPayload);
    pooldb.commit();    
    delete ioveditor;
  }catch(cond::Exception& er){
    std::cout<<er.what()<<std::endl;
  }catch(std::exception& er){
    std::cout<<er.what()<<std::endl;
  }
  con.disconnect();
  delete session;
  return 0;
}
