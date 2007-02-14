#include "Reflex/Type.h"

#include "FWCore/Framework/src/ReflexTools.h"

namespace ROOT
{
  namespace Reflex
  {
    std::ostream& operator<< (std::ostream& os, Type const& t)
    {
      os << t.Name();
      return os;
    }
  }
}

using ROOT::Reflex::Type;
using ROOT::Reflex::Type_Iterator;


namespace edm
{
  
  bool 
  find_nested_type_named(std::string const& nested_type,
			 Type const& type_to_search,
			 Type& found_type)
  {
    // Look for a sub-type named 'nested_type'
    for (Type_Iterator
	   i = type_to_search.SubType_Begin(),
	   e = type_to_search.SubType_End();
	 i != e;
	 ++i)
      {
	if (i->Name() == nested_type)
	  {
	    found_type = i->ToType();
	    return true;
	  }
      }
    return false;
  }

  bool
  is_sequence_wrapper(Type const& possible_sequence_wrapper,
		      Type& found_sequence_value_type)
  {
    Type possible_sequence;
    return 
      wrapper_type_of(possible_sequence_wrapper, possible_sequence) &&
      value_type_of(possible_sequence, found_sequence_value_type);
  }
}
