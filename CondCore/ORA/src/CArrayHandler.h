#ifndef INCLUDE_ORA_CARRAYHANDLER_H
#define INCLUDE_ORA_CARRAYHANDLER_H

#include "IArrayHandler.h"
// externals
#include "Reflex/Reflex.h"

namespace ora {

  class CArrayIteratorHandler : virtual public IArrayIteratorHandler {
    public:
    /// Constructor
    CArrayIteratorHandler( const void* startAddress, const Reflex::Type& iteratorReturnType );

    /// Destructor
    ~CArrayIteratorHandler();

    /// Increments itself
    void increment();

    /// Returns the current object
    void* object();

    /// Returns the return type of the iterator dereference method
    Reflex::Type& returnType();

    private:

    /// The return type of the iterator dereference method
    Reflex::Type m_returnType;
      
    /// Current element object pointer
    const void* m_currentElement;
  };


  class CArrayHandler : virtual public IArrayHandler {

    public:
      /// Constructor
    explicit CArrayHandler( const Reflex::Type& dictionary );

    /// Destructor
    ~CArrayHandler();

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
    bool isAssociative() const { return false; }
      
    private:
    /// The dictionary information
    Reflex::Type m_type;

    /// The iterator return type
    Reflex::Type m_elementType;

  };

}

#endif
