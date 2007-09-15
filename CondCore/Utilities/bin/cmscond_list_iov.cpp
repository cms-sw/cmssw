#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include <boost/program_options.hpp>
#include <iterator>
#include <iostream>
int main( int argc, char** argv ){
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_list_iov [options] \n");
  visible.add_options()
    ("connect,c",boost::program_options::value<std::string>(),"connection string(required)")
    ("user,u",boost::program_options::value<std::string>(),"user name (default \"\")")
    ("pass,p",boost::program_options::value<std::string>(),"password (default \"\")")
    ("authPath,P",boost::program_options::value<std::string>(),"path to authentication.xml")
    ("all,a","list all tags(default mode)")
    ("tag,t",boost::program_options::value<std::string>(),"list info of the specified tag")
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
  bool listAll=true;
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
  if( vm.count("authPath") ){
      authPath=vm["authPath"].as<std::string>();
  }
  if(vm.count("tag")){
    tag=vm["tag"].as<std::string>();
    listAll=false;
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
  static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
  if( connect.find("sqlite_fip:") != std::string::npos ){
    cond::FipProtocolParser p;
    connect=p.getRealConnect(connect);
  }
  conHandler.registerConnection("mydb",connect,0);  
  if( listAll ){
    try{
      session->open();
      conHandler.connect(session);
      cond::Connection* myconnection=conHandler.getConnection("mydb");
      cond::CoralTransaction& coraldb=myconnection->coralTransaction(true);
      cond::MetaData metadata_svc(coraldb);
      std::vector<std::string> alltags;
      coraldb.start();
      metadata_svc.listAllTags(alltags);
      coraldb.commit();
      std::copy (alltags.begin(),
		 alltags.end(),
		 std::ostream_iterator<std::string>(std::cout,"\n")
		 );
      return 0;
    }catch(cond::Exception& er){
      std::cout<<er.what()<<std::endl;
    }catch(std::exception& er){
      std::cout<<er.what()<<std::endl;
    }
   }else{ 
     try{
       session->open();
       conHandler.connect(session);
       cond::Connection* myconnection=conHandler.getConnection("mydb");
       cond::CoralTransaction& coraldb=myconnection->coralTransaction(true);
       cond::MetaData metadata_svc(coraldb);
       std::string token;
       coraldb.start();
       token=metadata_svc.getToken(tag);
       coraldb.commit();
       cond::PoolTransaction& pooldb = myconnection->poolTransaction(true);
       cond::IOVService iovservice(pooldb);
       cond::IOVIterator* ioviterator=iovservice.newIOVIterator(token);
       pooldb.start();
       unsigned int counter=0;
       std::string payloadContainer=iovservice.payloadContainerName(token);
       std::cout<<"Tag "<<tag<<"\n";
       std::cout<<"PayloadContainerName "<<payloadContainer<<"\n";
       std::cout<<"since \t till \t payloadToken"<<std::endl;
       while( ioviterator->next() ){
	 std::cout<<ioviterator->validity().first<<" \t "<<ioviterator->validity().second<<" \t "<<ioviterator->payloadToken()<<std::endl;	
	 ++counter;
       }
       std::cout<<"Total # of payload objects: "<<counter<<std::endl;
       pooldb.commit();
       delete ioviterator;
     }catch(cond::Exception& er){
       std::cout<<er.what()<<std::endl;
     }catch(std::exception& er){
       std::cout<<er.what()<<std::endl;
     }
   }
  delete session;
  return 0;
}
