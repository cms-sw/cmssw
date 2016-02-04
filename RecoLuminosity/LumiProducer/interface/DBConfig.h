#ifndef RecoLuminosity_LumiProducer_DBConfig_H
#define RecoLuminosity_LumiProducer_DBConfig_H
#include <string>
namespace coral{
  class ConnectionService;
}
namespace lumi{
  class DBConfig{
  public:
    explicit DBConfig(coral::ConnectionService& svc);
    ~DBConfig();
    void setAuthentication( const std::string& authPath );
    std::string trueConnectStr( const std::string& usercon );
    
  private:
    coral::ConnectionService* m_svc;
  };//cls DBConfig
}//ns lumi
#endif
