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
    virtual unsigned long long retrieveData( unsigned int ) = 0;
    virtual const std::string dataType() const = 0;
    virtual const std::string sourceType() const = 0;
    virtual ~DataPipe(){}
    void setNoValidate();
    void setNoCheckingStableBeam();
    void setSource( const std::string& source );
    void setAuthPath( const std::string& authpath );
    void setMode( const std::string& mode );
    void setNorm( float norm );
    std::string getSource() const;
    std::string getMode() const;
    std::string getAuthPath() const;
    float getNorm() const;

  protected:
    std::string m_dest;
    std::string m_source;
    std::string m_authpath;
    std::string m_mode;
    bool m_novalidate;
    float m_norm; //Lumi2DB specific
    bool m_nocheckingstablebeam; //Lumi2DB specific
  private:
    DataPipe( const DataPipe& );
    const DataPipe& operator=( const DataPipe& );
  };//class DataPipe
}//ns lumi
#endif
