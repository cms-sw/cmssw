#include "CondFormats/Common/interface/UpdateStamp.h"
#include "CondFormats/Common/interface/TimeConversions.h"

namespace {
  const std::string sep(". ");
}

namespace cond {

  UpdateStamp::UpdateStamp() : 
    m_revision(-1), m_timestamp(0), m_comment("not stamped"){}
    
  UpdateStamp::UpdateStamp(UpdateStamp const & rhs) {
    m_revision = rhs.m_revision;
    m_timestamp = rhs.m_timestamp;
    m_comment = rhs.m_comment;
  }
    
  UpdateStamp::~UpdateStamp(){}
    
    // stamp and return current revision number;
  int UpdateStamp::stamp(std::string const & icomment, bool append) {
    m_revision++;
    m_timestamp = cond::time::now();
    if (append && !icomment.empty()) m_comment += sep + icomment;
    else m_comment = icomment;
    return m_revision;
  }

}
