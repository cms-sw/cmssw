#ifndef CondCore_DBOutputService_serviceCallbackRecord_h
#define CondCore_DBOutputService_serviceCallbackRecord_h
#include <string>
namespace cond{
  class IOVEditor;
  namespace service{
    struct serviceCallbackRecord{
      serviceCallbackRecord():m_tag(""),m_isNewTag(false),m_containerName(""),m_iovEditor(0){}
      ~serviceCallbackRecord(){
      }
      std::string m_tag;
      bool m_isNewTag;
      std::string m_containerName;
      IOVEditor* m_iovEditor;
    };
  }//ns serviceCallbackRecord
}//ns cond
#endif
