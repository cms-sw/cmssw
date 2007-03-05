#ifndef FWCore_Utilities_ReflexTools_h
#define FWCore_Utilities_ReflexTools_h

/*----------------------------------------------------------------------

ReflexTools provides a small number of Reflex-based tools, used in
the CMS event model.  


$Id: ReflexTools.h,v 1.1 2007/03/04 04:39:42 wmtan Exp $

----------------------------------------------------------------------*/

#include <ostream>
#include <vector>

namespace ROOT
{
  namespace Reflex
  {
    class Type;
    class TypeTemplate;
    std::ostream& operator<< (std::ostream& os, Type const& t);  
    std::ostream& operator<< (std::ostream& os, TypeTemplate const& tt);
  }
}

namespace edm
{
  
  bool 
  find_nested_type_named(std::string const& nested_type,
			 ROOT::Reflex::Type const& type_to_search,
			 ROOT::Reflex::Type& found_type);

  inline
  bool 
  value_type_of(ROOT::Reflex::Type const& t, ROOT::Reflex::Type& found_type)
  {
    return find_nested_type_named("value_type", t, found_type);
  }


  inline
  bool 
  wrapper_type_of(ROOT::Reflex::Type const& possible_wrapper,
		  ROOT::Reflex::Type& found_wrapped_type)
  {
    return find_nested_type_named("wrapped_type",
				  possible_wrapper,
				  found_wrapped_type);
  }

  bool 
  is_sequence_wrapper(ROOT::Reflex::Type const& possible_sequence_wrapper,
		      ROOT::Reflex::Type& found_sequence_value_type);

  void 
    if_edm_ref_get_value_type(ROOT::Reflex::Type const& possible_ref,
			      ROOT::Reflex::Type & value_type);

  void checkDictionaries(std::string const& name, bool transient = false);
  void checkAllDictionaries();

  void public_base_classes(const ROOT::Reflex::Type& type,
                           std::vector<ROOT::Reflex::Type>& baseTypes);
}

#endif
