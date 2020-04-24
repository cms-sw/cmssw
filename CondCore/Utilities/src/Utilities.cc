#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//local includes
#include "CondCore/Utilities/interface/Utilities.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
#include <boost/foreach.hpp>                   
#include <fstream>
#include <iostream>

#include "CondCore/CondDB/interface/Auth.h"


cond::UtilitiesError::UtilitiesError(const std::string& message ):Exception(message){
}
cond::UtilitiesError::~UtilitiesError() throw(){}

cond::Utilities::Utilities( const std::string& commandName,
                            std::string positionalParameter):m_name(commandName),
							     m_options(std::string("Usage: ")+m_name+
								       std::string(" [options] ")+positionalParameter
								       +std::string(" \n")),
							     m_positionalOptions(),
							     m_values(){
  m_options.add_options()
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  if(!positionalParameter.empty()) {
    m_positionalOptions.add( positionalParameter.c_str(), -1);
    addOption<std::string>(positionalParameter,"",positionalParameter);
  }
}


cond::Utilities::~Utilities(){
}

int cond::Utilities::execute(){
  return 0;
}

int cond::Utilities::run( int argc, char** argv ){
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  std::vector<edm::ParameterSet> psets;
  edm::ParameterSet pSet;
  pSet.addParameter("@service_type",std::string("SiteLocalConfigService"));
  psets.push_back(pSet);
  edm::ServiceToken servToken(edm::ServiceRegistry::createSet(psets));
  m_currentToken = &servToken;
  edm::ServiceRegistry::Operate operate(servToken);

  int ret = 0;
  try{
    parseCommand( argc, argv );
    if (m_values.count("help")) {
      std::cout << m_options <<std::endl;;
      return 0;
    }
    ret = execute();
  }catch( cond::Exception& err ){
    std::cout << err.what() << std::endl;
    ret = 1;
  }catch( const std::exception& exc ){
    std::cout << exc.what() << std::endl;
    ret = 1;
  }
  m_currentToken = nullptr;
  return ret;
}

void
cond::Utilities::addConnectOption(const std::string& connectionOptionName,
                                  const std::string& shortName,
                                  const std::string& helpEntry ){
  addOption<std::string>(connectionOptionName,shortName,helpEntry);
}

void 
cond::Utilities::addAuthenticationOptions(){
  addOption<std::string>("authPath","P","path to the authentication key");
  addOption<std::string>("user","u","user name");
  addOption<std::string>("pass","p","password");
}

void 
cond::Utilities::addConfigFileOption(){
  addOption<std::string>("configFile","f","configuration file(optional)");
}

void cond::Utilities::parseCommand( int argc, char** argv ){
  boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(m_options).positional(m_positionalOptions).run(), m_values);
  if(m_options.find_nothrow("configFile",false)){
    std::string configFileName = getValueIfExists("configFile");
    if (! configFileName.empty()){
      std::fstream configFile;
      configFile.open(configFileName.c_str(), std::fstream::in);
      boost::program_options::store(boost::program_options::parse_config_file(configFile,m_options), m_values);
      configFile.close();
    }
  }
  boost::program_options::notify(m_values);
}

std::string cond::Utilities::getAuthenticationPathValue(){
  return getOptionValue<std::string>("authPath");
}

std::string cond::Utilities::getUserValue(){
  return getOptionValue<std::string>("user");  
}

std::string cond::Utilities::getPasswordValue(){
  return getOptionValue<std::string>("pass");  
}

std::string cond::Utilities::getConnectValue(){
  return getOptionValue<std::string>("connect");  
}

std::string cond::Utilities::getLogDBValue(){
  return getOptionValue<std::string>("logDB");  
}

std::string cond::Utilities::getDictionaryValue(){
  return getOptionValue<std::string>("dictionary");
}

std::string cond::Utilities::getConfigFileValue(){
  return getOptionValue<std::string>("configFile");
}


bool cond::Utilities::hasOptionValue(const std::string& fullName){
  const void* found = m_options.find_nothrow(fullName, false);
  if(!found){
    std::stringstream message;
    message << "Utilities::hasOptionValue: option \"" << fullName << "\" is not known by the command.";
    sendException(message.str());
  }
  return m_values.count(fullName);
}

bool cond::Utilities::hasDebug(){
  return m_values.count("debug");  
}

void cond::Utilities::initializePluginManager(){
  // dummy, to avoid to adapt non-CondCore clients
}

std::string cond::Utilities::getValueIfExists(const std::string& fullName){
  std::string val("");
  if(m_values.count(fullName)){
    val = m_values[fullName].as<std::string>();
  }
  return val;
}

void cond::Utilities::sendError( const std::string& message ){
  throw cond::UtilitiesError(message);
}

void cond::Utilities::sendException( const std::string& message ){
  throw cond::Exception(message);
}

