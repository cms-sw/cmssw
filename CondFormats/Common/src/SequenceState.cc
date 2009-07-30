#include "CondFormats/Common/interface/SequenceState.h"
#include "CondFormats/Common/interface/IOVSequence.h"

namespace cond {


  SequenceState::SequenceState() : 
    m_size(0), m_revision(-1), m_timestamp(0), m_comment("not stamped"){}
  
  SequenceState::SequenceState(IOVSequence const & seq) :
    m_size(seq.size()), m_revision(seq.revision()), 
    m_timestamp(seq.timestamp()), 
    m_comment(seq.comment()){}
  
}
