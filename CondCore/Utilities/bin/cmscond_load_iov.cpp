#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
//#include "CondCore/DBCommon/interface/ConnectionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"

#include "CondCore/DBCommon/interface/ObjectRelationalMappingUtility.h"
#include "CondCore/IOVService/interface/IOVNames.h"


#include <boost/program_options.hpp>
#include <iterator>
#include <iostream>
#include <fstream>

namespace{

  struct Parser {
    typedef std::pair<cond::Time_t, std::string> Item;

    std::string tag;
    std::string contName;
    std::vector<Item> values;
    cond::Time_t firstSince;


    void parseInputFile(std::fstream& file){
      
      
      std::string dummy;
      
      cond::Time_t since, till;
      std::string token;
      
      file >> dummy >> tag;
      file >> dummy >> contName;
      char buff[1024];
      file.getline(buff,1024);
      file.getline(buff,1024);
      char p;
      bool first=true;
      while(file) {
	file.get(p); if (p=='T') break;
	file.putback(p);
	file >> since >> till >> token;  file.getline(buff,1024);
	values.push_back(Item(till,token));
	if (first) {
	  first=false;
	  firstSince=since;
	}
      }
      
    }

  };

}
    
int main( int argc, char** argv ){
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_load_iov [options] inputFile \n");
  visible.add_options()
    ("connect,c",boost::program_options::value<std::string>(),"connection string(required)")
    ("user,u",boost::program_options::value<std::string>(),"user name (default \"\")")
    ("pass,p",boost::program_options::value<std::string>(),"password (default \"\")")
    ("authPath,P",boost::program_options::value<std::string>(),"path to authentication.xml")
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  boost::program_options::options_description invisible;
  invisible.add_options()
    ("inputFile",boost::program_options::value<std::string>(), "input file")
    ;
  desc.add(visible);
  desc.add(invisible);
  boost::program_options::positional_options_description posdesc;
  posdesc.add("inputFile", -1);

  boost::program_options::variables_map vm;
  try{
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(posdesc).run(), vm);
    boost::program_options::notify(vm);
  }catch(const boost::program_options::error& er) {
    std::cerr << er.what()<<std::endl;
    return 1;
  }
  if (vm.count("help")) {
    std::cout << visible <<std::endl;;
    return 0;
  }

  Parser parser;
  std::string connect;
  std::string user("");
  std::string pass("");
  std::string authPath("");
  std::string inputFileName;
  std::fstream inputFile;

  bool debug=false;


  if( !vm.count("inputFile") ){
    std::cerr <<"[Error] no input file given \n";
    std::cerr<<" please do "<<argv[0]<<" --help \n";
    return 1;
  }else{
    inputFileName=vm["inputFile"].as<std::string>();
    inputFile.open(inputFileName.c_str(), std::fstream::in);
    parser.parseInputFile(inputFile);
    inputFile.close();
  }
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
  if(vm.count("debug")){
    debug=true;
  }
  if(debug){
    std::cout<<"inputFile:\t"<<inputFileName<<std::endl;
    std::cout<<"connect:\t"<<connect<<'\n';
    std::cout<<"user:\t"<<user<<'\n';
    std::cout<<"pass:\t"<<pass<<'\n';
    std::cout<<"authPath:\t"<<authPath<<'\n';
  }
  std::string iovtoken;
  cond::DBSession* session=new cond::DBSession;
  if(!debug){
    session->configuration().setMessageLevel(cond::Error);
  }else{
    session->configuration().setMessageLevel(cond::Debug);
  }
  std::string userenv(std::string("CORAL_AUTH_USER=")+user);
  std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
  std::string authenv(std::string("CORAL_AUTH_PATH=")+authPath);
  ::putenv(const_cast<char*>(userenv.c_str()));
  ::putenv(const_cast<char*>(passenv.c_str()));
  ::putenv(const_cast<char*>(authenv.c_str()));
  cond::Connection myconnection(connect,-1);
  session->open();
  try{
    myconnection.connect(session);
    cond::PoolTransaction& pooldb=myconnection.poolTransaction();
    {
      cond::CoralTransaction& coraldb=myconnection.coralTransaction();
      coraldb.start(false); 

      // we need to clean this
      cond::ObjectRelationalMappingUtility mappingUtil(&(coraldb.coralSessionProxy()) );
      if( !mappingUtil.existsMapping(cond::IOVNames::iovMappingVersion()) ){
	mappingUtil.buildAndStoreMappingFromBuffer(cond::IOVNames::iovMappingXML());
      }

      coraldb.commit();
    }

    // FIXME need to get timetype from input!!!!
    cond::IOVService iovmanager(pooldb,cond::runnumber);
    cond::IOVEditor* editor=iovmanager.newIOVEditor("");
    pooldb.start(false);
    editor->create(parser.firstSince,cond::runnumber);
    editor->bulkInsert(parser.values);
    iovtoken=editor->token();
    pooldb.commit();
    cond::CoralTransaction& coraldb=myconnection.coralTransaction();
    cond::MetaData metadata(coraldb);
    coraldb.start(false);
    metadata.addMapping(parser.tag,iovtoken);
    coraldb.commit();
    if(debug){
      std::cout<<"source iov token "<<iovtoken<<std::endl;
    }
    myconnection.disconnect();
    delete editor;
    delete session;
  }catch(const cond::Exception& er){
    std::cout<<"error "<<er.what()<<std::endl;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
  }
  return 0;
}
