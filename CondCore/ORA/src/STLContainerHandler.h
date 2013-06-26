#ifndef INCLUDE_ORA_STLCONTAINERHANDLER_H
#define INCLUDE_ORA_STLCONTAINERHANDLER_H

#include "IArrayHandler.h"
//
#include <memory>
// externals
#include "Reflex/Reflex.h"
#include "Reflex/Builder/CollectionProxy.h"

namespace ora {

  class STLContainerIteratorHandler : virtual public IArrayIteratorHandler {
    public:
      /// Constructor
    STLContainerIteratorHandler( const Reflex::Environ<long>& collEnv,
                                 Reflex::CollFuncTable& collProxy,
                                 const Reflex::Type& iteratorReturnType );

      /// Destructor
      ~STLContainerIteratorHandler();

      /// Increments itself
      void increment();

      /// Returns the current object
      void* object();

      /// Returns the return type of the iterator dereference method
      Reflex::Type& returnType();

    private:

      /// The return type of the iterator dereference method
      Reflex::Type m_returnType;
      
      /// Structure containing parameters of the collection instance  
      Reflex::Environ<long> m_collEnv;

      /// Proxy of the generic collection
      Reflex::CollFuncTable& m_collProxy;

      /// Current element object pointer
      void* m_currentElement;
    };

 
    class STLContainerHandler : virtual public IArrayHandler {

    public:
      /// Constructor
      explicit STLContainerHandler( const Reflex::Type& dictionary );

      /// Destructor
      ~STLContainerHandler();

      /// Returns the size of the container
      size_t size( const void* address );

      /// Returns an initialized iterator
      IArrayIteratorHandler* iterate( const void* address );

      /// Appends a new element and returns its address of the object reference
      void appendNewElement( void* address, void* data );

      /// Clear the content of the container
      void clear( const void* address );
      
      /// Returns the iterator return type
      Reflex::Type& iteratorReturnType();

      /// Returns the associativeness of the container
      bool isAssociative() const { return m_isAssociative; }
      
    private:
      /// The dictionary information
      Reflex::Type m_type;

      /// The iterator return type
      Reflex::Type m_iteratorReturnType;

      /// Flag indicating whether the container is associative
      bool m_isAssociative;

      /// Structure containing parameters of the collection instance  
      Reflex::Environ<long> m_collEnv;

      /// Proxy of the generic collection
      std::auto_ptr<Reflex::CollFuncTable> m_collProxy;

    };


    class SpecialSTLContainerHandler : virtual public IArrayHandler {

    public:
      /// Constructor
      explicit SpecialSTLContainerHandler( const Reflex::Type& dictionary );

      /// Destructor
      ~SpecialSTLContainerHandler();

      /// Returns the size of the container
      size_t size( const void* address );

      /// Returns an initialized iterator
      IArrayIteratorHandler* iterate( const void* address );

      /// Appends a new element and returns its address of the object reference
      void appendNewElement( void* address, void* data );

      /// Clear the content of the container
      void clear( const void* address );

      /// Returns the iterator return type
      Reflex::Type& iteratorReturnType();

    private:
      /// The handler of the unserlying container
      std::auto_ptr<IArrayHandler> m_containerHandler;

      /// The offset of the underlying container
      int m_containerOffset;
    };


}

#endif
