#ifndef RecoLuminosity_LumiProducer_DataPipe_H
#define RecoLuminosity_LumiProducer_DataPipe_H
#include <string>
namespace edm{
  class ParameterSet;
}
namespace lumi{
  class DataPipe{
  public:
    explicit DataPipe( const std::string& );
    virtual void retrieveRun( unsigned int ) = 0;
    virtual const std::string dataType() const = 0;
    virtual const std::string sourceType() const = 0;
    virtual ~DataPipe(){}
    void setSource( const std::string& source );
    void setAuthPath( const std::string& authpath );
  protected:
    std::string m_dest;
    std::string m_source;
    std::string m_authpath;
  private:
    DataPipe( const DataPipe& );
    const DataPipe& operator=( const DataPipe& );
  };//class DataPipe
}//ns lumi
#endif
