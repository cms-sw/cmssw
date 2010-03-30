#ifndef INCLUDE_ORA_PVECTORHANDLER_H
#define INCLUDE_ORA_PVECTORHANDLER_H

#include "IArrayHandler.h"
//
#include <memory>
// externals
#include "Reflex/Reflex.h"
#include "Reflex/Builder/CollectionProxy.h"

namespace ora {

  class PVectorIteratorHandler : virtual public IArrayIteratorHandler 
  {

    public:
    /// Constructor
    PVectorIteratorHandler( const Reflex::Environ<long>& collEnv,
                            Reflex::CollFuncTable& collProxy,
                            const Reflex::Type& iteratorReturnType,
                            size_t startElement );

    /// Destructor
    virtual ~PVectorIteratorHandler();

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

    size_t m_startElement;
    
  };

  class PVectorHandler : virtual public IArrayHandler {

    public:
      /// Constructor
      explicit PVectorHandler( const Reflex::Type& dictionary );

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
      Reflex::Type& iteratorReturnType();

      /// Returns the associativeness of the container
      bool isAssociative() const {
        return m_isAssociative;
      }

      /// Returns the persistent size of the container
      size_t persistentSize( const void* address );

      /// execute the ending procedure for the container
      void finalize( void* address );

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

      size_t m_persistentSizeAttributeOffset;

      size_t m_vecAttributeOffset;

  };

 }

#endif
