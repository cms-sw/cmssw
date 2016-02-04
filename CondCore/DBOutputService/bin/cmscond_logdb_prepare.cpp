#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Logger.h"
#include <boost/program_options.hpp>
#include <iostream>
int main( int argc, char** argv ){
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_list_iov [options] \n");
  visible.add_options()
    ("connect,c",boost::program_options::value<std::string>(),"connection string(required)")
    ("user,u",boost::program_options::value<std::string>(),"user name (default \"\")")
    ("pass,p",boost::program_options::value<std::string>(),"password (default \"\")")
    ("authPath,P",boost::program_options::value<std::string>(),"path to authentication.xml")
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
  bool debug=false;
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
  cond::DbConnection connection;
  if( !authPath.empty() ){
    connection.configuration().setAuthenticationPath( authPath );
  }else{
    std::string userenv(std::string("CORAL_AUTH_USER=")+user);
    std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
    ::putenv(const_cast<char*>(userenv.c_str()));
    ::putenv(const_cast<char*>(passenv.c_str()));
  }
  if(debug){
    connection.configuration().setMessageLevel( coral::Debug );
  }else{
    connection.configuration().setMessageLevel( coral::Error );
  }

  connection.configure();
  cond::DbSession session = connection.createSession();
  session.open( connect );
  
  try{
    cond::Logger mylogger(session);
    mylogger.getWriteLock();
    mylogger.createLogDBIfNonExist();
    mylogger.releaseWriteLock();
  }catch(std::exception& er){
      std::cout<<er.what()<<std::endl;
  }
  return 0;
}
