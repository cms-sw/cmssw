#ifndef Parser_MethodMap_h
#define Parser_MethodMap_h
/* \class reco::MethodMap
 *
 * Maps method names to Reflex mehods
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 */
#include "PhysicsTools/Utilities/src/TypeCode.h"
#include <Reflex/Type.h>
#include <Reflex/Member.h>
#include <typeinfo>
#include <string>
#include <map>

namespace reco {
  class MethodMap {
  public:
    typedef std::pair<ROOT::Reflex::Member, method::TypeCode> method_t;
    typedef std::map<std::string, method_t> container;
    typedef container::const_iterator const_iterator;
    
    template<typename T>
    const static MethodMap & methods() {
      static MethodMap map;
      if(map.size() == 0) map.fill(ROOT::Reflex::Type::ByTypeInfo(typeid(T)));
      return map;
    }
    
    size_t size() { return map_.size(); }
    const ROOT::Reflex::Type & type() const { return type_; }
    const_iterator find( const container::key_type & k ) const { return map_.find( k ); }
    const_iterator begin() const { return map_.begin(); }
    const_iterator end() const { return map_.end(); }

  private:
    MethodMap() { }
    void fill( const ROOT::Reflex::Type &, bool fillBase = false );
    ROOT::Reflex::Type type_;
    container map_;
  };
}

#endif
