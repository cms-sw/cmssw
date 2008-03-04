#ifndef Parser_MethodMap_h
#define Parser_MethodMap_h
/* \class reco::MethodMap
 *
 * Maps method names to Reflex mehods
 *
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.3 $
 *
 */
#include <Reflex/Type.h>
#include <Reflex/Member.h>
#include <typeinfo>
#include <iosfwd>
#include <string>
#include <map>

namespace reco {
  namespace method {
    enum retType { 
      doubleType = 0, floatType,
      intType, uIntType,
      charType, uCharType,
      shortType, uShortType, 
      longType, uLongType, 
      boolType
    };
  }

  class MethodMap {
  public:
    typedef std::pair<ROOT::Reflex::Member, method::retType> method_t;
    typedef std::map<std::string, method_t> container;
    typedef container::const_iterator const_iterator;
    
    template<typename T>
    const static MethodMap & methods() {
      static MethodMap map;
      if( map.size() == 0 ) map.fill( ROOT::Reflex::Type::ByTypeInfo( typeid( T ) ) );
      return map;
    }

    size_t size() { return map_.size(); }
    const ROOT::Reflex::Type & type() const { return type_; }
    const_iterator find( const container::key_type & k ) const { return map_.find( k ); }
    const_iterator begin() const { return map_.begin(); }
    const_iterator end() const { return map_.end(); }
    void print( std::ostream & ) const;

  private:
    MethodMap() { }
    void fill( const ROOT::Reflex::Type &, bool fillBase = false );
    static std::string memberReturnType( const ROOT::Reflex::Member & mem );
    ROOT::Reflex::Type type_;
    container map_;
  };
}

inline std::ostream & operator << ( std::ostream & out, const reco::MethodMap & m ) {
  m.print( out );
  return out;
}

#endif
