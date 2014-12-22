#include "CondCore/ORA/interface/Exception.h"
#include "ArrayHandlerFactory.h"
#include "STLContainerHandler.h"
#include "CArrayHandler.h"
#include "PVectorHandler.h"
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

ora::IArrayHandler*
ora::ArrayHandlerFactory::newArrayHandler( const edm::TypeWithDict& arrayType )
{
  if(arrayType.isArray()){
    return new CArrayHandler( arrayType );
  } else {  
    if ( arrayType.isTemplateInstance() ) {
      std::string contName = arrayType.templateName(); 
      if(  contName == "std::vector"              ||
           contName == "std::list"                ||
           contName == "std::set"                 ||
           contName == "std::multiset"            ||
           contName == "std::deque"               ||
           contName == "__gnu_cxx::hash_set"      ||
           contName == "__gnu_cxx::hash_multiset" ||
           contName == "std::map"                 ||
           contName == "std::multimap"            ||
           contName == "__gnu_cxx::hash_map"      ||
           contName == "__gnu_cxx::hash_multimap" ){
        return new STLContainerHandler( arrayType );
      } else if (  contName == "std::stack"      ||
                   contName == "std::queue"      ){
        return new SpecialSTLContainerHandler( arrayType );
      } else if (  contName == "ora::PVector" ) {
        return new PVectorHandler( arrayType );
      }
      
    }
  }
  throwException( "No Array Handler available for class \""+arrayType.templateName()+"\"",
                  "ArrayHandlerFactory::newArrayHandler");
  return 0;
}
