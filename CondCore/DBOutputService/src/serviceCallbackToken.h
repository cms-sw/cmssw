#ifndef CondCore_DBOutputService_serviceCallbackToken_h
#define CondCore_DBOutputService_serviceCallbackToken_h
#include <string>
namespace cond{
  namespace service{
    class serviceCallbackToken{
      friend class PoolDBOutputService;
    public:
      ~serviceCallbackToken(){}
    protected:
      serviceCallbackToken(){}
      static size_t build(const std::string& str1,
			  const std::string& str2 );
    };
  }//ns service
}//ns cond
#endif
