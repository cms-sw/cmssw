#ifndef CondCore_DBOutputService_serviceCallbackRecord_h
#define CondCore_DBOutputService_serviceCallbackRecord_h
#include <string>
namespace cond{
  namespace service{
    struct serviceCallbackRecord{
      serviceCallbackRecord():m_tag(""),m_isNewTag(false),m_containerName(""),m_iovtoken(""){}
      ~serviceCallbackRecord(){
      }
      std::string m_tag;
      bool m_isNewTag;
      std::string m_containerName;
      std::string m_iovtoken;
    };
  }//ns serviceCallbackRecord
}//ns cond
#endif
