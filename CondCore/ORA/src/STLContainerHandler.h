#ifndef INCLUDE_ORA_STLCONTAINERHANDLER_H
#define INCLUDE_ORA_STLCONTAINERHANDLER_H

#include "IArrayHandler.h"
//
#include <memory>
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/FunctionWithDict.h"

#include "TVirtualCollectionIterators.h"
#include "TVirtualCollectionProxy.h"

namespace ora {

  class STLContainerIteratorHandler : virtual public IArrayIteratorHandler {
    public:
      /// Constructor
    STLContainerIteratorHandler( void* address,
                                 TVirtualCollectionProxy& collProxy,
                                 const edm::TypeWithDict& iteratorReturnType );

      /// Destructor
      ~STLContainerIteratorHandler();

      /// Increments itself
      void increment();

      /// Returns the current object
      void* object();

      /// Returns the return type of the iterator dereference method
      edm::TypeWithDict& returnType();

    private:

      /// The return type of the iterator dereference method
      edm::TypeWithDict m_returnType;

      /// Proxy of the generic collection
      TVirtualCollectionProxy& m_collProxy;

      /// Current element object pointer
      void* m_currentElement;

      // holds the iterator when the branch is of fType==4.
      TGenericCollectionIterator *m_Iterators;

    };


    class STLContainerHandler : virtual public IArrayHandler {

    public:
      /// Constructor
      explicit STLContainerHandler( const edm::TypeWithDict& dictionary );

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
      edm::TypeWithDict& iteratorReturnType();

      /// Returns the associativeness of the container
      bool isAssociative() const { return m_isAssociative; }

    private:
      /// The dictionary information
      edm::TypeWithDict m_type;

      /// The iterator return type
      edm::TypeWithDict m_iteratorReturnType;

      /// Flag indicating whether the container is associative
      bool m_isAssociative;

      /// Proxy of the generic collection
      TVirtualCollectionProxy* m_collProxy;

    };


    class SpecialSTLContainerHandler : virtual public IArrayHandler {

    public:
      /// Constructor
      explicit SpecialSTLContainerHandler( const edm::TypeWithDict& dictionary );

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
      edm::TypeWithDict& iteratorReturnType();

    private:
      /// The handler of the unserlying container
      std::auto_ptr<IArrayHandler> m_containerHandler;

      /// The offset of the underlying container
      int m_containerOffset;
    };


}

#endif
