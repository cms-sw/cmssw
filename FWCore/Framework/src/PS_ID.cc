/*----------------------------------------------------------------------

$Id: PS_ID.cc,v 1.2 2005/03/23 20:29:43 paterno Exp $

----------------------------------------------------------------------*/

#include <algorithm>
#include <istream>
#include <ostream>

#include "FWCore/CoreFramework/interface/PS_ID.h"

namespace edm
{
  PS_ID::PS_ID() :
    value_()
  { }

  PS_ID::PS_ID(const std::string& hash_data) :
    value_()
  { }

  bool
  PS_ID::operator< (const PS_ID& id) const
  {
    return value_ < id.value_;
  }

  bool
  PS_ID::operator== (const PS_ID& id) const
  {
    return value_ == id.value_;
  }


  void
  PS_ID::print(std::ostream& os) const
  {
    os << value_ ;
  }

  void
  PS_ID::restore(std::istream& is) 
  {
    unsigned int temp;
    is >> temp;
    if ( is ) value_ = temp;
  }

  void
  PS_ID::swap(PS_ID& other)
  {
    std::swap(value_, other.value_);
  }

  inline std::ostream& operator<<(std::ostream& s, const PS_ID& id)
    {
      id.print(s);
      return s;
    }

  inline std::istream& operator<<(std::istream& s, PS_ID& id)
    {
      id.restore(s);
      return s;
    }

}

