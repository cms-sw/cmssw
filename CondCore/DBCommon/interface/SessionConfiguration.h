#ifndef COND_DBCommon_SessionConfiguration_h
#define COND_DBCommon_SessionConfiguration_h
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include <string>
namespace cond{
  class ConnectionConfiguration;
  class SessionConfiguration{
  public:
    SessionConfiguration();
    ~SessionConfiguration();
    ConnectionConfiguration* connectionConfiguration();
    void setAuthenticationMethod( cond::AuthenticationMethod m );
    void setAuthenticationPath( const std::string& p );
    void setBlobStreamer( const std::string& name );
    void setMessageLevel( cond::MessageLevel l );
    cond::AuthenticationMethod authenticationMethod() const;
    bool hasBlobStreamService() const;
    std::string blobStreamerName() const;
    cond::MessageLevel messageLevel() const;
    std::string authName() const;
  private:
    cond::AuthenticationMethod m_authMethod;
    std::string m_authPath;
    bool m_hasBlobstreamer;
    std::string m_blobstreamerName;
    cond::MessageLevel m_messageLevel;
    cond::ConnectionConfiguration* m_conConfig;
  };
}
#endif
