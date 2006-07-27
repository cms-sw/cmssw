#ifndef UtilAlgos_methods_h
#define UtilAlgos_methods_h
/* \class ObjectMethodSet
 *
 * \author Luca Lista, INFN
 *
 * $Id: ObjectMethodSet.h,v 1.1 2006/07/27 10:45:23 llista Exp $
 *
 */
#include <Reflex/Type.h>
#include <Reflex/Base.h>
#include <Reflex/Member.h>
#include <typeinfo>
#include <map>
#include <ostream>
#include <utility>

namespace reco {
  namespace methods {
    enum retType { 
      doubleType = 0, floatType,
      intType, unsignedIntType,
      charType, unsignedCharType,
      boolType
    };
    typedef std::map<std::string, std::pair<ROOT::Reflex::Member, retType> > methodMap;
    
    template<typename T>
    const methodMap & methods();

    template<typename T>
    void printMethods( std::ostream & );

    void fill( methodMap &, const ROOT::Reflex::Type & );
    
    template<typename T>
    const methodMap & methods() {
      using namespace ROOT::Reflex;
      static methodMap map;
      if( map.size() == 0 ) {
	Type t = Type::ByTypeInfo( typeid( T ) );
	fill( map, t );
	for( Base_Iterator b = t.Base_Begin(); b != t.Base_End(); ++ b ) {
	  fill( map, b->ToType() );
	}
      }
      return map;
    }

    template<typename T>
    void printMethods( std::ostream & out ) {
      const methodMap & map = methods<T>();
      ROOT::Reflex::Type t = ROOT::Reflex::Type::ByTypeInfo( typeid( T ) );
      for( methodMap::const_iterator i = map.begin(); i != map.end(); ++ i ) {
	std::string n = i->second.first.TypeOf().Name();
	size_t p = n.find( '(' );
	std::string retName = n.substr( 0, p - 1 );
	out << retName << " " << t.Name() << "::" << i->second.first.Name() << "()" << std::endl;
      }
    }
  }
}

#endif
