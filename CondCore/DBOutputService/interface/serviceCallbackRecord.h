#ifndef CondCore_DBOutputService_serviceCallbackRecord_h
#define CondCore_DBOutputService_serviceCallbackRecord_h
#include <string>
namespace cond{
  class DBWriter;
  class IOV;
  namespace service{
    struct serviceCallbackRecord{
      serviceCallbackRecord():m_tag(""),m_containerName("")/*,m_timetype("")*/,m_iovToken(""),m_iov(0),m_appendIOV(false),m_payloadWriter(0){}
      ~serviceCallbackRecord(){
	if(m_payloadWriter) {
	  delete m_payloadWriter;
	  m_payloadWriter=0;
	}
      }
      std::string m_tag;
      std::string m_containerName;
      //std::string m_timetype;
      std::string m_iovToken;
      cond::IOV* m_iov;
      bool m_appendIOV;
      cond::DBWriter* m_payloadWriter;
    };
  }//ns serviceCallbackRecord
}//ns cond
#endif
