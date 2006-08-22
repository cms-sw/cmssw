#ifndef Common_ParameterSetBlob_h
#define Common_ParameterSetBlob_h

/*----------------------------------------------------------------------
  
ParameterSetBlob: A string in which to store a parameter set so that it can be made persistent.

The ParameterSetBlob is a concatenation of the names and values of the
tracked parameters within a ParameterSet,

$Id: ParameterSetBlob.h,v 1.1 2006/03/10 17:54:19 wmtan Exp $

----------------------------------------------------------------------*/

#include <ostream>
#include <string>

namespace edm {
  struct ParameterSetBlob {
    typedef std::string value_t;
    ParameterSetBlob() : pset_() {}
    explicit ParameterSetBlob(value_t const& v) : pset_(v) {}
    value_t pset_;
  };
  inline
  std::ostream&
  operator<<(std::ostream& os, ParameterSetBlob const& blob) {
    os << blob.pset_;
    return os;
  }
}
#endif
