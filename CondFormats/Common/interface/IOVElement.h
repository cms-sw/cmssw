#ifndef Cond_IOVElement_h
#define Cond_IOVElement_h
#include "CondFormats/Common/interface/Time.h"
#include "CondCore/ORA/interface/OId.h"
#include <string>

namespace cond {

  /** Element of an IOV Sequence
      include the since time, token to the wrapper
   */
  class IOVElement {
    public:
    IOVElement():
      m_sinceTime(0),
      m_wrapper(""),
      m_oid(){}

    explicit IOVElement(cond::Time_t it) : 
      m_sinceTime(it),
      m_wrapper(""),
      m_oid(){}

    IOVElement(cond::Time_t it,
	       std::string const& itoken):
      m_sinceTime(it),
      m_wrapper(""),
      m_oid(){
      m_oid.fromString( itoken );
    }

    cond::Time_t sinceTime() const {return m_sinceTime;}

    std::string token() const {
      return m_oid.toString();
    }

    bool operator==(IOVElement const & rh) const {
      return sinceTime()==rh.sinceTime()
	&&  token()==rh.token();
    }
    
    void swapToken( ora::ITokenParser& parser ) const {
      if( !m_wrapper.empty() ){
        const_cast<IOVElement*>(this)->m_oid = parser.parse( m_wrapper );
        const_cast<IOVElement*>(this)->m_wrapper.clear();
      }
    }

    void swapOId( ora::ITokenWriter& writer ) const {
      if( !m_oid.isInvalid() ){
        const_cast<IOVElement*>(this)->m_wrapper = writer.write( m_oid );
      }
    }

    private:
    cond::Time_t m_sinceTime;
    std::string m_wrapper;
    ora::OId m_oid;
    
  };
  
  
} // ns cond


#endif
