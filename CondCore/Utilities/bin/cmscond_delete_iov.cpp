#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
//#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include <boost/program_options.hpp>
#include <iostream>

#include "SealBase/SharedLibrary.h"
#include "SealBase/SharedLibraryError.h"



int main( int argc, char** argv ){
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_delete_iov [options] \n");
  visible.add_options()
    ("connect,c",boost::program_options::value<std::string>(),"connection string(required)")
    ("user,u",boost::program_options::value<std::string>(),"user name (default \"\")")
    ("pass,p",boost::program_options::value<std::string>(),"password (default \"\")")
    ("authPath,P",boost::program_options::value<std::string>(),"path to authentication.xml")
    ("all,a","delete all tags")
    ("tag,t",boost::program_options::value<std::string>(),"delete the specified tag and IOV")
    ("withPayload","delete payload data associated with the specified tag (default off)")
    ("dictionary,D",boost::program_options::value<std::string>(),"data dictionary(required if withPayload)")
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
    deleteAll=false;
  }
  if(vm.count("withPayload")){
    withPayload=true;
    if(!vm.count("dictionary")){
      std::cerr <<"[Error] no dictionary[D] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      dictionary=vm["dictionary"].as<std::string>();
    }
  }
  if(vm.count("debug")){
    debug=true;
  }


  if (!dictionary.empty()) {
    std::string dictlibrary=seal::SharedLibrary::libname( dictionary );
    try {
      seal::SharedLibrary::load( dictlibrary );
    }catch ( seal::SharedLibraryError *error) {
      throw std::runtime_error( error->explainSelf().c_str() );
    }
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
  //rely on default
  //session->configuration().connectionConfiguration()->setConnectionRetrialTimeOut( 600 );
  //session->configuration().connectionConfiguration()->enableConnectionSharing();
  //session->configuration().connectionConfiguration()->enableReadOnlySessionOnUpdateConnections(); 
  std::string userenv(std::string("CORAL_AUTH_USER=")+user);
  std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
  std::string authenv(std::string("CORAL_AUTH_PATH=")+authPath);
  ::putenv(const_cast<char*>(userenv.c_str()));
  ::putenv(const_cast<char*>(passenv.c_str()));
  ::putenv(const_cast<char*>(authenv.c_str()));
  //std::string catalog("pfncatalog_memory://POOL_RDBMS?");
  //catalog.append(connect);
  cond::Connection con(connect,-1);
  session->open();
  con.connect(session);
  if( deleteAll ){
    try{
      cond::PoolTransaction& pooldb=con.poolTransaction();
      // irrelevant which tymestamp
      cond::IOVService iovservice(pooldb);
      pooldb.start(false);
      iovservice.deleteAll(withPayload);
      pooldb.commit();
      cond::CoralTransaction& coraldb=con.coralTransaction();
      cond::MetaData metadata_svc(coraldb);
      coraldb.start(false);
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
      ioveditor->deleteEntries(withPayload);
      pooldb.commit();
      coraldb.start(false);
      metadata_svc.deleteEntryByTag(tag);
      coraldb.commit();
      delete ioveditor;
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
