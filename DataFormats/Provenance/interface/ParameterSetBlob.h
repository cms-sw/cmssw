#ifndef DataFormats_Provenance_ParameterSetBlob_h
#define DataFormats_Provenance_ParameterSetBlob_h

/*----------------------------------------------------------------------
  
ParameterSetBlob: A string in which to store a parameter set so that it can be made persistent.

The ParameterSetBlob is a concatenation of the names and values of the
tracked parameters within a ParameterSet,

----------------------------------------------------------------------*/

#include <iosfwd>
#include <string>

namespace edm {
  class ParameterSetBlob {
  public:
    typedef std::string value_t;
    ParameterSetBlob() : pset_() {}
    explicit ParameterSetBlob(value_t const& v) : pset_(v) {}
    value_t const& pset() const {return pset_;}
    value_t& pset() {return pset_;}
  private:
    value_t pset_;
  };
  std::ostream&
  operator<<(std::ostream& os, ParameterSetBlob const& blob);
}
#endif
