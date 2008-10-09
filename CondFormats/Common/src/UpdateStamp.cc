#include "CondFormats/Common/interface/UpdateStamp.h"
#include "CondFormats/Common/interface/TimeConversions.h"


namespace cond {

  UpdateStamp::UpdateStamp() : 
    m_revision(-1), m_timestamp(0), m_comment("not stamped"){}
    
    
  virtual UpdateStamp::~UpdateStamp(){}
    
    // stamp and return current revision number;
  int UpdateStamp::stamp(std::string const & icomment) {
    m_revision++;
    m_timestamp = cond::time::now();
    m_comment = icomment;
    return m_revision;
  }
    
}
