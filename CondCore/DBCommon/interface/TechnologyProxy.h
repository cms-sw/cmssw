#ifndef CondCore_DBCommon_TechnologyProxy_h
#define CondCore_DBCommon_TechnologyProxy_h
#include <string>
namespace cond{
  class DBSession;
  class TechnologyProxy{
  public:
    explicit TechnologyProxy( const std::string& userconnect ):m_userconnect(userconnect){}
    virtual ~TechnologyProxy(){}
    virtual std::string getRealConnectString() const=0;
    virtual void setupSession( DBSession& session )=0;
  protected:
    std::string m_userconnect;
  private:
    TechnologyProxy( const TechnologyProxy& );
    const TechnologyProxy& operator=(const TechnologyProxy&); 
  };
}//ns cond
#endif
