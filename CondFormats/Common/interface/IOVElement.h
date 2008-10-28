#ifndef Cond_IOVElement_h
#define Cond_IOVElement_h
#include "CondFormats/Common/interface/Time.h"
#include <string>

namespace cond {

  /** Element of an IOV Sequence
      include the till time, token to the payload, token to the metadata
   */
  class IOVElement {
  public:
    IOVElement(){}
    // for comparisons
    IOVElement(cond::Time_t it) : m_tillTime(it){}

    IOVElement(cond::Time_t it,
	       std::string const& ipayload,
	       std::string const& imetadata):
      m_tillTime(it),m_payload(ipayload),m_metadata(imetadata){}

    cond::Time_t tillTime() const {return m_tillTime;}
    std::string const & payloadToken()  const {return m_payload;}
    std::string const & metadataToken() const { return m_metadata;}


  private:
    cond::Time_t m_tillTime;
    std::string m_payload;
    std::string m_metadata;

  };


} // ns cond

bool operator==(cond::IOVElement const & lh, cond::IOVElement const & rh) {
  return lh.tillTime()==rh.tillTime()
    &&  lh.hpayloadToken()==rh.payloadToken()
    &&  lh.metadataToken()==rh.metadataToken();
}


#endif // IOVElement_h
