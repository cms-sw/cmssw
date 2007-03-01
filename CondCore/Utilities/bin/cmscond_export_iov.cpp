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
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "SealBase/SharedLibrary.h"
#include "SealBase/SharedLibraryError.h"
#include <boost/program_options.hpp>
#include <iterator>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
int main( int argc, char** argv ){
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_export_iov [options] \n");
  visible.add_options()
    ("sourceConnect,s",boost::program_options::value<std::string>(),"source connection string(required)")
    ("destConnect,d",boost::program_options::value<std::string>(),"destionation connection string(required)")
    ("dictionary,D",boost::program_options::value<std::string>(),"data dictionary(required)")
    ("inputCatalog,i",boost::program_options::value<std::string>(),"input catalog contact string(required)")
    ("outputCatalog,o",boost::program_options::value<std::string>(),"output catalog contact string(required)")
    ("tag,t",boost::program_options::value<std::string>(),"tag to export(required)")
    ("payloadName,n",boost::program_options::value<std::string>(),"payload object name(required)")
    ("authPath,p",boost::program_options::value<std::string>(),"path to authentication xml(default .)")
    ("configFile,f",boost::program_options::value<std::string>(),"configuration file(optional)")
    ("withBlob","with blob streaming capability")
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  desc.add(visible);
  std::string sourceConnect, destConnect;
  std::string inputCatalog, outputCatalog;
  std::string dictionary;
  std::string tag;
  std::string payloadName;
  std::string authPath(".");
  std::string configuration_filename;
  bool debug=false;
  bool withBlob=false;
  boost::program_options::variables_map vm;
  try{
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
      std::cout << visible <<std::endl;;
      return 0;
    }
    if( vm.count("configFile") ){
      configuration_filename=vm["configFile"].as<std::string>();
      if (! configuration_filename.empty()){
	std::fstream configuration_file;
	configuration_file.open(configuration_filename.c_str(), std::fstream::in);
	boost::program_options::store(boost::program_options::parse_config_file(configuration_file,desc), vm);
	configuration_file.close();
      }
    }
    if(!vm.count("sourceConnect")){
      std::cerr <<"[Error] no sourceConnect[s] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      sourceConnect=vm["sourceConnect"].as<std::string>();
    }
    if(!vm.count("destConnect")){
      std::cerr <<"[Error] no destConnect[s] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      destConnect=vm["destConnect"].as<std::string>();
    }
    if(!vm.count("inputCatalog")){
      std::cerr <<"[Error] no inputCatalog[i] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      inputCatalog=vm["inputCatalog"].as<std::string>();
    }
    if(!vm.count("outputCatalog")){
      std::cerr <<"[Error] no outputCatalog[o] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      outputCatalog=vm["outputCatalog"].as<std::string>();
    }
    if(!vm.count("dictionary")){
      std::cerr <<"[Error] no dictionary[D] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      dictionary=vm["dictionary"].as<std::string>();
    }
    if(!vm.count("tag")){
      std::cerr <<"[Error] no tag[t] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      tag=vm["tag"].as<std::string>();
    }
    if(!vm.count("payloadName")){
      std::cerr <<"[Error] no payloadName[n] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }else{
      payloadName=vm["payloadName"].as<std::string>();
    }
    if( vm.count("authPath") ){
      authPath=vm["authPath"].as<std::string>();
    }
    if(vm.count("debug")){
      debug=true;
    }
    if(vm.count("withBlob")){
      withBlob=true;
    }
    boost::program_options::notify(vm);
  }catch(const boost::program_options::error& er) {
    std::cerr << er.what()<<std::endl;
    return 1;
  }
  std::string dictlibrary=seal::SharedLibrary::libname( dictionary );
  if(debug){
    std::cout<<"sourceConnect:\t"<<sourceConnect<<'\n';
    std::cout<<"inputCatalog:\t"<<inputCatalog<<'\n';
    std::cout<<"destConnect:\t"<<destConnect<<'\n';
    std::cout<<"outputCatalog:\t"<<outputCatalog<<'\n';
    std::cout<<"dictionary:\t"<<dictlibrary<<'\n';
    std::cout<<"payloadName:\t"<<payloadName<<'\n';
    std::cout<<"tag:\t"<<tag<<'\n';
    std::cout<<"authPath:\t"<<authPath<<'\n';
    if(withBlob) std::cout<<"with Blob streamer"<<authPath<<'\n';
    std::cout<<"configFile:\t"<<configuration_filename<<std::endl;
  }
  //
  try {
    seal::SharedLibrary::load( dictlibrary );
  }catch ( seal::SharedLibraryError *error) {
    throw std::runtime_error( error->explainSelf().c_str() );
  }
  cond::DBSession* session=new cond::DBSession(true);
  if(!debug){
    session->sessionConfiguration().setMessageLevel(cond::Error);
  }else{
    session->sessionConfiguration().setMessageLevel(cond::Debug);
  }
  session->sessionConfiguration().setAuthenticationMethod(cond::XML);
  session->sessionConfiguration().setBlobStreamer("");
  std::string pathval("CORAL_AUTH_PATH=");
  pathval+=authPath;
  ::putenv(const_cast<char*>(pathval.c_str()));
  session->open();
  try{
    std::string sourceiovtoken;
    std::string destiovtoken;
    cond::RelationalStorageManager* sourceCoralDB=new cond::RelationalStorageManager(sourceConnect,session);
    sourceCoralDB->connect(cond::ReadOnly);
    sourceCoralDB->startTransaction(true);
    cond::MetaData* sourceMetadata=new cond::MetaData(*sourceCoralDB);
    if( !sourceMetadata->hasTag(tag) ){
      throw std::runtime_error(std::string("tag ")+tag+std::string(" not found") );
    }
    sourceiovtoken=sourceMetadata->getToken(tag);
    sourceCoralDB->commit();
    sourceCoralDB->disconnect();
    if(debug){
      std::cout<<"source iov token "<<sourceiovtoken<<std::endl;
    }
    delete sourceMetadata;
    delete sourceCoralDB;
    cond::PoolStorageManager sourcedb(sourceConnect,inputCatalog,session);
    sourcedb.connect();
    cond::IOVService iovmanager(sourcedb);
    cond::IOVEditor* editor=iovmanager.newIOVEditor();
    cond::PoolStorageManager destdb(destConnect,outputCatalog,session);
    destdb.connect();
    sourcedb.startTransaction(true);
    destdb.startTransaction(false);
    destiovtoken=iovmanager.exportIOVWithPayload( destdb,
						  sourceiovtoken,
						  payloadName );
    sourcedb.commit();
    destdb.commit();
    sourcedb.disconnect();
    destdb.disconnect();
    cond::RelationalStorageManager* destCoralDB=new cond::RelationalStorageManager(destConnect,session);
    cond::MetaData* destMetadata=new cond::MetaData(*destCoralDB);
    destCoralDB->connect(cond::ReadWriteCreate);
    destCoralDB->startTransaction(false);
    destMetadata->addMapping(tag,destiovtoken);
    if(debug){
      std::cout<<"source iov token "<<sourceiovtoken<<std::endl;
    }
    destCoralDB->commit();
    destCoralDB->disconnect();
    delete destMetadata;
    delete destCoralDB;
    session->close();
    delete editor;
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
  return 0;
}
