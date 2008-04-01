#ifndef CondCore_DBCommon_TechnologyProxy_h
#define CondCore_DBCommon_TechnologyProxy_h
#include <string>
namespace cond{
  class TechnologyProxy{
  public:
    TechnologyProxy(){}
    virtual ~TechnologyProxy(){}
    virtual std::string getRealConnectString( const std::string& iValue ) const=0;
    virtual void setupSession()=0;
    virtual void prepareConnection()=0;
    virtual void prepareTransaction()=0;
  private:
    TechnologyProxy( const TechnologyProxy& );
    const TechnologyProxy& operator=(const TechnologyProxy&); 
  };
}//ns cond
#endif
