#ifndef INCLUDE_ORA_ARRAYHANDLERFACTORY_H
#define INCLUDE_ORA_ARRAYHANDLERFACTORY_H

namespace edm {
  class TypeWithDict;
}

namespace ora {

    class IArrayHandler;

    /**
       @class ArrayHandlerFactory ArrayHandlerFactory.h
       Factory class for IArrayHandler objects.
     */

    class ArrayHandlerFactory {
    public:
      static IArrayHandler* newArrayHandler( const edm::TypeWithDict& arrayType );
    };

}

#endif
