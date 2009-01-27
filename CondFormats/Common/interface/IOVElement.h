#ifndef Cond_IOVElement_h
#define Cond_IOVElement_h
#include "CondFormats/Common/interface/Time.h"
#include <string>

namespace cond {

  /** Element of an IOV Sequence
      include the since time, token to the payload, token to the metadata
      FIXME: the metadata shall be moved in a payload wrapper using pool:Ptr
   */
  class IOVElement {
  public:
    IOVElement(){}
    // for comparisons
    IOVElement(cond::Time_t it) : m_sinceTime(it){}

    IOVElement(cond::Time_t it,
	       std::string const& ipayload,
	       std::string const& imetadata):
      m_sinceTime(it),m_payload(ipayload),m_metadata(imetadata){}

    cond::Time_t sinceTime() const {return m_sinceTime;}
    std::string const & payloadToken()  const {return m_payload;}
    std::string const & metadataToken() const { return m_metadata;}

    bool operator==(IOVElement const & rh) const {
      return sinceTime()==rh.sinceTime()
	&&  payloadToken()==rh.payloadToken()
	&&   metadataToken()==rh.metadataToken();
    }

    
  private:
    cond::Time_t m_sinceTime;
    std::string m_payload;
    std::string m_metadata;
    
  };
  
  
} // ns cond

/*  ???? why is not found in template code
inline bool operator==(cond::IOVElement const & lh, cond::IOVElement const & rh) {
  return lh.tillTime()==rh.tillTime()
    &&  lh.payloadToken()==rh.payloadToken()
    &&  lh.metadataToken()==rh.metadataToken();
}
*/

#endif // IOVElement_h
