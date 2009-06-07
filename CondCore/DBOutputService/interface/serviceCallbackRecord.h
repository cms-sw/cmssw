#ifndef CondCore_DBOutputService_serviceCallbackRecord_h
#define CondCore_DBOutputService_serviceCallbackRecord_h
#include "CondFormats/Common/interface/Time.h"
#include <string>
namespace cond{
  namespace service{
    struct serviceCallbackRecord{
      serviceCallbackRecord():m_tag(""),m_isNewTag(false),
			      m_idName(""),m_iovtoken(""),
			      m_freeInsert(false),
			      m_withWrapper(false)
{}
      ~serviceCallbackRecord(){
      }

      std::string timetypestr() const { return cond::timeTypeSpecs[m_timetype].name;}
      std::string m_tag;
      bool m_isNewTag;
      std::string m_idName;
      std::string m_iovtoken;
      cond::TimeType m_timetype;
      bool m_freeInsert;
      bool m_withWrapper;


    };
  }//ns serviceCallbackRecord
}//ns cond
#endif
