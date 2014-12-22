#ifndef INCLUDE_ORA_IARRAYHANDLER_H
#define INCLUDE_ORA_IARRAYHANDLER_H

#include <cstddef>

namespace edm {
  class TypeWithDict;
}

namespace ora {

    class IArrayIteratorHandler {
    public:
      /// Destructor
      virtual ~IArrayIteratorHandler() {}

      /// Increments itself
      virtual void increment() = 0;

      /// Returns the current object
      virtual void* object() = 0;

      /// Returns the return type of the iterator dereference method
      virtual edm::TypeWithDict& returnType() = 0;
    };


    class IArrayHandler {

    public:
      /// Destructor
      virtual ~IArrayHandler() {}

      /// Returns the size of the container
      virtual size_t size( const void* address ) = 0;

      /// Returns the index of the first element
      virtual size_t startElementIndex( const void* ){ return 0; };

      /// Returns an initialized iterator
      virtual IArrayIteratorHandler* iterate( const void* address ) = 0;

      /// Appends a new element and returns its address
      virtual void appendNewElement( void* address, void* data ) = 0;

      /// Clear the content of the container
      virtual void clear( const void* address ) = 0;

      /// Returns the associativeness of the container
      virtual bool isAssociative() const { return false; }

      /// Returns the size of the container. Only differs in the PVector. 
      virtual size_t* persistentSize( const void* address ){ return 0; }

      /// execute the ending procedure for the container
      virtual void finalize( void* ){ }

      /// Returns the iterator return type
      virtual edm::TypeWithDict& iteratorReturnType() = 0;
    };

}


#endif
