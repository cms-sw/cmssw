#ifndef Utilities_Utilities_h
#define Utilities_Utilities_h

#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include <boost/program_options.hpp>
#include <sstream>
#include <set>

namespace edm {
  class ServiceToken;
}

namespace cond {
  class DbConnection;

  class UtilitiesError : public Exception {
    public:
    UtilitiesError(const std::string& message );
    virtual ~UtilitiesError() throw();    
  };

  class Utilities {
    public:
    Utilities( const std::string& commandName, std::string positionalParameter=std::string("") );

    virtual ~Utilities();

    virtual int execute();
    int run( int argc, char** argv );
    
    void addAuthenticationOptions();
    void addConnectOption();
    void addConnectOption(const std::string& connectionOptionName,
                          const std::string& shortName,
                          const std::string& helpEntry );
    void addLogDBOption();
    void addDictionaryOption();
    void addConfigFileOption();
    void addSQLOutputOption();

    template <typename T> void addOption(const std::string& fullName,
                                         const std::string& shortName,
                                         const std::string& helpEntry );
    
    void parseCommand( int argc, char** argv );

    std::string getAuthenticationPathValue();
    std::string getUserValue();
    std::string getPasswordValue();
    std::string getConnectValue();
    std::string getLogDBValue();
    std::string getDictionaryValue();
    std::string getConfigFileValue();
    template <typename T> T getOptionValue(const std::string& fullName);
    bool hasOptionValue(const std::string& fullName);
    bool hasDebug();
    void initializePluginManager();
    cond::DbSession openDbSession( const std::string& connectionParameterName, bool readOnly=false );
    cond::DbSession openDbSession( const std::string& connectionParameterName, const std::string& role, bool readOnly=false );

    protected:
    cond::DbSession newDbSession(  const std::string& connectionString, bool readOnly=false );
    cond::DbSession newDbSession(  const std::string& connectionString, const std::string& role, bool readOnly=false );
    void initializeForDbConnection();
  
    private:

    std::string getValueIfExists(const std::string& fullName);
    void sendException( const std::string& message );
    void sendError( const std::string& message );

    protected:
    edm::ServiceToken* m_currentToken = nullptr;
    
    private:

    std::string m_name;
    //boost::program_options::options_description m_description;
    boost::program_options::options_description m_options;
    boost::program_options::positional_options_description m_positionalOptions;
    boost::program_options::variables_map m_values;
    cond::DbConnection* m_dbConnection;
    std::set<std::string> m_dbSessions;
  };
  

template <typename T> inline void Utilities::addOption(const std::string& fullName,
                                                      const std::string& shortName,
                                                      const std::string& helpEntry){
  std::stringstream optInfo;
  optInfo << fullName;
  if(!shortName.empty()) optInfo << ","<<shortName;
  m_options.add_options()
    (optInfo.str().c_str(),boost::program_options::value<T>(),helpEntry.c_str());
}

template <> inline void Utilities::addOption<bool>(const std::string& fullName,
                                                  const std::string& shortName,
                                                  const std::string& helpEntry){
  std::stringstream optInfo;
  optInfo << fullName;
  if(!shortName.empty()) optInfo << ","<<shortName;
  m_options.add_options()
    (optInfo.str().c_str(),helpEntry.c_str());
}

}


template <typename T> inline T cond::Utilities::getOptionValue(const std::string& fullName){
  const void* found = m_options.find_nothrow(fullName, false);
  if(!found){
    std::stringstream message;
    message << "Utilities::getOptionValue: option \"" << fullName << "\" is not known by the command.";
    sendException(message.str());
  } 
  
  if (!m_values.count(fullName)) {
    std::stringstream message;
    message << "Error: Option \"" << fullName << "\" has not been provided.";
    sendError(message.str());
  }
  return m_values[fullName].as<T>();
}

#endif


