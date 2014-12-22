#ifndef INCLUDE_ORA_PVECTORHANDLER_H
#define INCLUDE_ORA_PVECTORHANDLER_H

#include "IArrayHandler.h"
//
#include <memory>
// externals
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include "TVirtualCollectionIterators.h"
#include "TVirtualCollectionProxy.h"

namespace ora {

  class PVectorIteratorHandler : virtual public IArrayIteratorHandler 
  {

    public:
    /// Constructor
    PVectorIteratorHandler( void* address,
			    TVirtualCollectionProxy& collProxy,
			    const edm::TypeWithDict& iteratorReturnType,
                            size_t startElement );

    /// Destructor
    virtual ~PVectorIteratorHandler();

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

    // holds the iterators when the branch is of fType==4.
    TGenericCollectionIterator *m_Iterators;
    
    size_t m_startElement;
    
  };

  class PVectorHandler : virtual public IArrayHandler {

    public:
      /// Constructor
      explicit PVectorHandler( const edm::TypeWithDict& dictionary );

      /// Destructor
      virtual ~PVectorHandler();

      /// Returns the size of the container
      size_t size( const void* address );

      /// Returns the index of the first element
      size_t startElementIndex( const void* address );

      /// Returns an initialized iterator
      IArrayIteratorHandler* iterate( const void* address );

      /// Appends a new element and returns its address of the object reference
      void appendNewElement( void* address, void* data );

      /// Clear the content of the container
      void clear( const void* address );

      /// Returns the iterator return type
      edm::TypeWithDict& iteratorReturnType();

      /// Returns the associativeness of the container
      bool isAssociative() const {
        return false;
      }

      /// Returns the persistent size of the container
      size_t* persistentSize( const void* address );

      /// execute the ending procedure for the container
      void finalize( void* address );

    private:
      /// The dictionary information
      edm::TypeWithDict m_type;

      /// The iterator return type
      edm::TypeWithDict m_iteratorReturnType;

      /// Proxy of the generic collection
      TVirtualCollectionProxy* m_collProxy;

      size_t m_persistentSizeAttributeOffset;

      size_t m_vecAttributeOffset;

  };

 }

#endif
