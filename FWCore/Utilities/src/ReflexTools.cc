#include "FWCore/Utilities/interface/ReflexTools.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "Api.h" // for G__ClassInfo
#include "Reflex/Base.h"
#include "Reflex/Member.h"
#include "Reflex/TypeTemplate.h"

#include "boost/algorithm/string.hpp"
#include "boost/thread/tss.hpp"

#include <algorithm>
#include <sstream>

using Reflex::Base;
using Reflex::FINAL;
using Reflex::Member;
using Reflex::Object;
using Reflex::SCOPED;
using Reflex::Type;
using Reflex::TypeTemplate;
using Reflex::Type_Iterator;

namespace edm {

  Type get_final_type(Type t) {
    while(t.IsTypedef()) t = t.ToType();
    return t;
  }

  bool
  find_nested_type_named(std::string const& nested_type,
                         Type const& type_to_search,
                         Type& found_type) {
    // Look for a sub-type named 'nested_type'
    for(Type_Iterator
           i = type_to_search.SubType_Begin(),
           e = type_to_search.SubType_End();
           i != e;
           ++i) {
      if(i->Name() == nested_type) {
        found_type = get_final_type(*i);
        return true;
      }
    }
    return false;
  }

  bool
  is_RefVector(Reflex::Type const& possible_ref_vector,
               Reflex::Type& value_type) {

    static TypeTemplate ref_vector_template_id(TypeTemplate::ByName("edm::RefVector", 3));
    static std::string member_type("member_type");
    TypeTemplate primary_template_id(possible_ref_vector.TemplateFamily());
    if(primary_template_id == ref_vector_template_id) {
      return find_nested_type_named(member_type, possible_ref_vector, value_type);
    }
    return false;
  }

  bool
  is_PtrVector(Reflex::Type const& possible_ref_vector,
               Reflex::Type& value_type) {

    static TypeTemplate ref_vector_template_id(TypeTemplate::ByName("edm::PtrVector", 1));
    static std::string member_type("member_type");
    static std::string val_type("value_type");
    TypeTemplate primary_template_id(possible_ref_vector.TemplateFamily());
    if(primary_template_id == ref_vector_template_id) {
      Reflex::Type ptrType;
      if(find_nested_type_named(val_type, possible_ref_vector, ptrType)) {
        return find_nested_type_named(val_type, ptrType, value_type);
      }
    }
    return false;
  }

  bool
  is_RefToBaseVector(Reflex::Type const& possible_ref_vector,
                     Reflex::Type& value_type) {

    static TypeTemplate ref_vector_template_id(TypeTemplate::ByName("edm::RefToBaseVector", 1));
    static std::string member_type("member_type");
    TypeTemplate primary_template_id(possible_ref_vector.TemplateFamily());
    if(primary_template_id == ref_vector_template_id) {
      return find_nested_type_named(member_type, possible_ref_vector, value_type);
    }
    return false;
  }

  namespace {

    int const oneParamArraySize = 6;
    std::string const oneParam[oneParamArraySize] = {
      "vector",
      "basic_string",
      "set",
      "list",
      "deque",
      "multiset"
    };
    int const twoParamArraySize = 3;
    std::string const twoParam[twoParamArraySize] = {
      "map",
      "pair",
      "multimap"
    };


    bool
    hasCintDictionary(std::string const& name) {
      std::auto_ptr<G__ClassInfo> ci(new G__ClassInfo(name.c_str()));
        return(ci.get() && ci->IsLoaded());
    }

    // Checks if there is a Reflex dictionary for the Type t.
    // If noComponents is false, checks members and base classes recursively.
    // If noComponents is true, checks Type t only.
    void
    checkType(Type t, bool noComponents = false) {

      // ToType strips const, volatile, array, pointer, reference, etc.,
      // and also translates typedefs.
      // To be safe, we do this recursively until we either get a null type
      // or the same type.
      Type null;
      for(Type x = t.ToType(); x != null && x != t; t = x, x = t.ToType()) {}

      std::string name = t.Name(SCOPED);
      boost::trim(name);

      if(foundTypes().end() != foundTypes().find(name) || missingTypes().end() != missingTypes().find(name)) {
        // Already been processed. Prevents infinite loop.
        return;
      }

      if(name.empty() || t.IsFundamental() || t.IsEnum()) {
        foundTypes().insert(name);
        return;
      }

      if(!bool(t)) {
        if(hasCintDictionary(name)) {
          foundTypes().insert(name);
        } else {
          missingTypes().insert(name);
        }
        return;
      }

      foundTypes().insert(name);
      if(noComponents) return;

      if(name.find("std::") == 0) {
        if(t.IsTemplateInstance()) {
          std::string::size_type n = name.find('<');
          int cnt = 0;
          if(std::find(oneParam, oneParam + oneParamArraySize, name.substr(5, n - 5)) != oneParam + oneParamArraySize) {
            cnt = 1;
          } else if(std::find(twoParam, twoParam + twoParamArraySize, name.substr(5, n - 5)) != twoParam + twoParamArraySize) {
            cnt = 2;
          }
          for(int i = 0; i < cnt; ++i) {
            checkType(t.TemplateArgumentAt(i));
          }
        }
      } else {
        int mcnt = t.DataMemberSize();
        for(int i = 0; i < mcnt; ++i) {
          Member m = t.DataMemberAt(i);
          if(m.IsTransient() || m.IsStatic()) continue;
          checkType(m.TypeOf());
        }
        int cnt = t.BaseSize();
        for(int i = 0; i < cnt; ++i) {
          checkType(t.BaseAt(i).ToType());
        }
      }
    }
  } // end unnamed namespace

  StringSet& missingTypes() {
    static boost::thread_specific_ptr<StringSet> missingTypes_;
    if(0 == missingTypes_.get()) {
      missingTypes_.reset(new StringSet);
    }
    return *missingTypes_.get();
  }

  StringSet& foundTypes() {
    // The only purpose of this cache is to stop infinite recursion.
    // Reflex maintains its own internal cache.
    static boost::thread_specific_ptr<StringSet> foundTypes_;
    if(0 == foundTypes_.get()) {
      foundTypes_.reset(new StringSet);
    }
    return *foundTypes_.get();
  }

  void checkDictionaries(std::string const& name, bool noComponents) {
    Type null;
    Type t = Type::ByName(name);
    if(t == null) {
      missingTypes().insert(name);
      return;
    }
    checkType(Type::ByName(name), noComponents);
  }

  void throwMissingDictionariesException() {
    if(!missingTypes().empty()) {
      std::ostringstream ostr;
      for (StringSet::const_iterator it = missingTypes().begin(), itEnd = missingTypes().end();
           it != itEnd; ++it) {
        ostr << *it << "\n\n";
      }
      throw Exception(errors::DictionaryNotFound)
        << "No REFLEX data dictionary found for the following classes:\n\n"
        << ostr.str()
        << "Most likely each dictionary was never generated,\n"
        << "but it may be that it was generated in the wrong package.\n"
        << "Please add (or move) the specification\n"
        << "<class name=\"whatever\"/>\n"
        << "to the appropriate classes_def.xml file.\n"
        << "If the class is a template instance, you may need\n"
        << "to define a dummy variable of this type in classes.h.\n"
        << "Also, if this class has any transient members,\n"
        << "you need to specify them in classes_def.xml.";
    }
  }

  void public_base_classes(Type const& type,
                           std::vector<Type>& baseTypes) {

    if(type.IsClass() || type.IsStruct()) {

      int nBase = type.BaseSize();
      for(int i = 0; i < nBase; ++i) {

        Base base = type.BaseAt(i);
        if(base.IsPublic()) {

          Type baseType = type.BaseAt(i).ToType();
          if(bool(baseType)) {

            while(baseType.IsTypedef() == true) {
              baseType = baseType.ToType();
            }

            // Check to make sure this base appears only once in the
            // inheritance heirarchy.
            if(!search_all(baseTypes, baseType)) {
              // Save the type and recursive look for its base types
              baseTypes.push_back(baseType);
              public_base_classes(baseType, baseTypes);
            }
            // For now just ignore it if the class appears twice,
            // After some more testing we may decide to uncomment the following
            // exception.
            /*
            else {
              throw Exception(errors::UnimplementedFeature)
                << "DataFormats/Common/src/ReflexTools.cc in function public_base_classes.\n"
                << "Encountered class that has a public base class that appears\n"
                << "multiple times in its inheritance heirarchy.\n"
                << "Please contact the EDM Framework group with details about\n"
                << "this exception. It was our hope that this complicated situation\n"
                << "would not occur. There are three possible solutions. 1. Change\n"
                << "the class design so the public base class does not appear multiple\n"
                << "times in the inheritance heirarchy. In many cases, this is a\n"
                << "sign of bad design. 2. Modify the code that supports Views to\n"
                << "ignore these base classes, but not supply support for creating a\n"
                << "View of this base class. 3. Improve the View infrastructure to\n"
                << "deal with this case. Class name of base class: " << baseType.Name() << "\n\n";
            }
            */
          }
        }
      }
    }
  }

  void const*
  reflex_pointer_adjust(void* raw,
                        Type const& dynamicType,
                        std::type_info const& toType) {
    Object obj(dynamicType, raw);
    return obj.CastObject(Type::ByTypeInfo(toType)).Address();
  }
}
