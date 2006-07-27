#ifndef UtilAlgos_ObjectMethodSet_h
#define UtilAlgos_PbjectMethodSwt_h
/* \class ObjectMethodSet
 *
 * \author Luca Lista, INFN
 *
 * $Id$
 *
 */
#include "FWCore/Utilities/interface/EDMException.h"
#include <Reflex/Type.h>
#include <Reflex/Base.h>
#include <Reflex/Member.h>
#include <typeinfo>
#include <map>
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <functional>
#include <ostream>

namespace reco {
  template<typename T>
  struct ObjectMethodSet {
    typedef boost::function<double ( T const & )> method;
    typedef std::map<std::string, ROOT::Reflex::Member> methodMap;
    static const methodMap & methods();
    static void print( std::ostream & );
  private:
    static void fill( methodMap &, const ROOT::Reflex::Type & );
  };

  template<typename T>
  void ObjectMethodSet<T>::fill( typename ObjectMethodSet<T>::methodMap & map, const ROOT::Reflex::Type & t) {
    using namespace ROOT::Reflex;
    if ( ! t )  
      throw edm::Exception( edm::errors::ProductNotFound, "NoMatch" )
	<< "TypeID::className: No dictionary for class " << typeid( T ).name() << '\n';
    Member mem;
    for( size_t i = 0; i < t.FunctionMemberSize(); ++ i ) {
      mem = t.FunctionMemberAt( i );
      if ( mem.FunctionParameterSize( true ) != 0 ) continue;
      if ( mem.IsConstructor() ) continue;
      if ( mem.IsDestructor() ) continue;
      if ( mem.IsOperator() ) continue;
      if ( ! mem.IsPublic() ) continue;
      if ( mem.IsStatic() ) continue;
      // how to check it is const?
      std::string methodName = mem.Name();
      if ( methodName.substr( 0, 2 ) == "__" ) continue;
      map[ methodName ] = mem;
    }
  }
  
  template<typename T>
    const typename ObjectMethodSet<T>::methodMap & ObjectMethodSet<T>::methods() {
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
    void ObjectMethodSet<T>::print( std::ostream & out ) {
    const methodMap & map = methods();
    ROOT::Reflex::Type t = ROOT::Reflex::Type::ByTypeInfo( typeid( T ) );
    out << "methods available for class " << t.Name() << ":" << std::endl;
    for( methodMap::const_iterator i = map.begin(); i != map.end(); ++ i ) {
      out << i->first << std::endl;
    }
  }
}

#endif
