#ifndef FWCoreFramework_ReflexTools_h
#define FWCoreFramework_ReflexTools_h

/*----------------------------------------------------------------------

ReflexTools provides a small number of Reflex-based tools, used in
the CMS event model.  


$Id: Group.h,v 1.14 2006/08/10 23:24:42 wmtan Exp $

----------------------------------------------------------------------*/

namespace ROOT
{
  namespace Reflex
  {
    class Type;
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

}

#endif
