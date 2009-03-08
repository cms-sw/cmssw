#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/Utilities/interface/CommonOptions.h"


#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondFormats/Common/interface/PayloadWrapper.h"
#include "CondCore/DBCommon/interface/TypedRef.h"


#include <boost/program_options.hpp>
#include <iterator>
#include <iostream>
int main( int argc, char** argv ){
  edmplugin::PluginManager::configure(edmplugin::standard::config());
  cond::CommonOptions myopt("cmscond_list_iov");
  myopt.addConnect();
  myopt.addAuthentication(true);
  myopt.visibles().add_options()
    ("all,a","list all tags(default mode)")
    ("tag,t",boost::program_options::value<std::string>(),"list info of the specified tag")
    ("summary,s","print also the summary for each payload")
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
  bool listAll=true;
  bool debug=false;
  bool details=false;

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
    if(vm.count("summary"))
      details=true;
  }
  
  if(vm.count("debug")){
    debug=true;
  }

  std::vector<edm::ParameterSet> psets;

  edm::ParameterSet pSet;
  pSet.addParameter("@service_type",std::string("SiteLocalConfigService"));
  psets.push_back(pSet);

  edm::ServiceToken services(edm::ServiceRegistry::createSet(psets));
  edm::ServiceRegistry::Operate operate(services);


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
  //rely on default
  //session->configuration().connectionConfiguration()->setConnectionRetrialTimeOut( 600 );
  //session->configuration().connectionConfiguration()->enableConnectionSharing();
  //session->configuration().connectionConfiguration()->enableReadOnlySessionOnUpdateConnections();
  //session->connectionService().configuration().disablePoolAutomaticCleanUp();
  //session->connectionService().configuration().setConnectionTimeOut(0);
  
  if( connect.find("sqlite_fip:") != std::string::npos ){
    cond::FipProtocolParser p;
    connect=p.getRealConnect(connect);
  }
  // cond::Connection myconnection(connect,-1);  
  session->open();

  cond::ConnectionHandler::Instance().registerConnection(connect,*session,-1);
  cond::Connection & myconnection = *cond::ConnectionHandler::Instance().getConnection(connect);
  
  if( listAll ){
    try{
      myconnection.connect(session);
      cond::CoralTransaction& coraldb=myconnection.coralTransaction();
      cond::MetaData metadata_svc(coraldb);
      std::vector<std::string> alltags;
      coraldb.start(true);
      metadata_svc.listAllTags(alltags);
      coraldb.commit();
      myconnection.disconnect();
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
       myconnection.connect(session);
       cond::CoralTransaction& coraldb=myconnection.coralTransaction();
       cond::MetaData metadata_svc(coraldb);
       std::string token;
       coraldb.start(true);
       token=metadata_svc.getToken(tag);
       coraldb.commit();
       cond::PoolTransaction& pooldb = myconnection.poolTransaction();
       {
	 cond::IOVProxy iov( pooldb, token, !details);
	 cond::IOVService iovservice(pooldb);
	 unsigned int counter=0;
	 std::string payloadContainer=iovservice.payloadContainerName(token);
	 std::cout<<"Tag "<<tag
		  <<"\nTimeType " << cond::timeTypeSpecs[iov.timetype()].name
		  <<"\nPayloadContainerName "<<payloadContainer<<"\n"
		  <<"since \t till \t payloadToken"<<std::endl;
	 for (cond::IOVProxy::const_iterator ioviterator=iov.begin(); ioviterator!=iov.end(); ioviterator++) {
	   std::cout<<ioviterator->since() << " \t "<<ioviterator->till() <<" \t "<<ioviterator->wrapperToken();
	   if (details) {
	     cond::TypedRef<cond::PayloadWrapper> wrapper(pooldb,ioviterator->wrapperToken());
	     if (wrapper.ptr()) std::cout << " \t "<< wrapper->summary();
	   }
	   std::cout<<std::endl;	
	   ++counter;
	 }
	 std::cout<<"Total # of payload objects: "<<counter<<std::endl;
       }
       myconnection.disconnect();
     }catch(cond::Exception& er){
       std::cout<<er.what()<<std::endl;
     }catch(std::exception& er){
       std::cout<<er.what()<<std::endl;
     }
   }
  delete session;
  return 0;
}
