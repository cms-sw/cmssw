/*----------------------------------------------------------------------
  
$Id: ParameterSetID.cc,v 1.2 2005/09/02 19:47:32 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/ParameterSetID.h"

namespace edm
{

  ParameterSetID::ParameterSetID() :
    value_()
  { }

  ParameterSetID::ParameterSetID(const value_t& v) :
    value_(v) 
  { }

  bool
  ParameterSetID::isValid() const
  {
    // For now, just check the most basic: a default constructed
    // ParameterSetID is not valid. This is very crude: we are
    // assuming that nobody created a ParameterSetID from an empty
    // string, nor from any string that is not a valid string
    // representation of an MD5 checksum.
    return ! value_.empty();
  }

  // A fundamental operator
  bool
  ParameterSetID::operator==(ParameterSetID const& rh) const 
  {
    return value_ == rh.value_;
  }

  // The following operator is defined in terms of the fundamental
  // operator above.
  bool
  ParameterSetID::operator!=(ParameterSetID const& rh) const 
  {
    return !(*this == rh);
  }

  // Another fundamental operator
  bool 
  ParameterSetID::operator<(ParameterSetID const& rh) const
  {
    return value_ < rh.value_;
  }

  // The following operator is defined in terms of the fundamental
  // operator above.

  bool 
  ParameterSetID::operator>(ParameterSetID const& rh) const 
  {
    return rh < *this;
  }

  std::ostream& 
  ParameterSetID::print(std::ostream& os) const 
  {
    os << value_;
    return os;
  }

}; // namespace edm
