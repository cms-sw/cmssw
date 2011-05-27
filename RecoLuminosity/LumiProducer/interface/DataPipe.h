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
    virtual void retrieveData( unsigned int ) = 0;
    virtual const std::string dataType() const = 0;
    virtual const std::string sourceType() const = 0;
    virtual ~DataPipe(){}
    void setSource( const std::string& source );
    void setAuthPath( const std::string& authpath );
    void setMode( const std::string& mode );
    std::string getSource() const;
    std::string getMode() const;
    std::string getAuthPath() const;
    
  protected:
    std::string m_dest;
    std::string m_source;
    std::string m_authpath;
    std::string m_mode;
  private:
    DataPipe( const DataPipe& );
    const DataPipe& operator=( const DataPipe& );
  };//class DataPipe
}//ns lumi
#endif
