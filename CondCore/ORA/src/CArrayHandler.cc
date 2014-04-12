#include "CArrayHandler.h"
#include "ClassUtils.h"
// externals
#include "Reflex/Object.h"

ora::CArrayIteratorHandler::CArrayIteratorHandler( const void* startAddress,
                                                   const Reflex::Type& iteratorReturnType ):
  m_returnType(iteratorReturnType),
  m_currentElement(startAddress){

}

ora::CArrayIteratorHandler::~CArrayIteratorHandler(){}

void
ora::CArrayIteratorHandler::increment()
{
  m_currentElement = static_cast< const char* >( m_currentElement) + m_returnType.SizeOf();
}

void*
ora::CArrayIteratorHandler::object()
{
  return const_cast<void*>(m_currentElement);
}

Reflex::Type&
ora::CArrayIteratorHandler::returnType()
{
  return m_returnType;
}

ora::CArrayHandler::CArrayHandler( const Reflex::Type& dictionary ):
  m_type( dictionary ),
  m_elementType()
{

  // find the iterator return type 
  Reflex::Type elementType = m_type.ToType();
  m_elementType = ClassUtils::resolvedType( elementType );
  
}

ora::CArrayHandler::~CArrayHandler(){
}

size_t
ora::CArrayHandler::size( const void* )
{
  return m_type.ArrayLength();
}


ora::IArrayIteratorHandler*
ora::CArrayHandler::iterate( const void* address )
{
  return new ora::CArrayIteratorHandler( address, m_elementType );
}


void
ora::CArrayHandler::appendNewElement( void*, void* )
{
}

void
ora::CArrayHandler::clear( const void* )
{
}

Reflex::Type&
ora::CArrayHandler::iteratorReturnType()
{
  return m_elementType;
}
