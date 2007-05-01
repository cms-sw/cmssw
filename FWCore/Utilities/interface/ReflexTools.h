#ifndef FWCore_Utilities_ReflexTools_h
#define FWCore_Utilities_ReflexTools_h

/*----------------------------------------------------------------------

ReflexTools provides a small number of Reflex-based tools, used in
the CMS event model.  


$Id: ReflexTools.h,v 1.4 2007/05/01 16:45:20 wmtan Exp $

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

  typedef std::set<std::string> StringSet;
  
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

  // is_sequence_wrapper is used to determine whether the Type
  // 'possible_sequence_wrapper' represents
  //   edm::Wrapper<Seq<X> >,
  // where Seq<X> is anything that is a sequence of X.
  // Note there is special support of edm::RefVector<Seq<X> >, which
  // will be recognized as a sequence of X.
  bool 
  is_sequence_wrapper(ROOT::Reflex::Type const& possible_sequence_wrapper,
		      ROOT::Reflex::Type& found_sequence_value_type);

  void 
  if_edm_ref_get_value_type(ROOT::Reflex::Type const& possible_ref,
			    ROOT::Reflex::Type & value_type);

  bool
  is_RefVector(ROOT::Reflex::Type const& possible_ref_vector,
	       ROOT::Reflex::Type& value_type);

  void checkDictionaries(std::string const& name, bool transient = false);
  void checkAllDictionaries();
  StringSet & missingTypes();

  void public_base_classes(const ROOT::Reflex::Type& type,
                           std::vector<ROOT::Reflex::Type>& baseTypes);
}

#endif
