#include "CArrayHandler.h"
#include "ClassUtils.h"
// externals
#include "FWCore/Utilities/interface/ObjectWithDict.h"

ora::CArrayIteratorHandler::CArrayIteratorHandler( const void* startAddress,
                                                   const edm::TypeWithDict& iteratorReturnType ):
  m_returnType(iteratorReturnType),
  m_currentElement(startAddress){

}

ora::CArrayIteratorHandler::~CArrayIteratorHandler(){}

void
ora::CArrayIteratorHandler::increment()
{
  m_currentElement = static_cast< const char* >( m_currentElement) + m_returnType.size();
}

void*
ora::CArrayIteratorHandler::object()
{
  return const_cast<void*>(m_currentElement);
}

edm::TypeWithDict&
ora::CArrayIteratorHandler::returnType()
{
  return m_returnType;
}

ora::CArrayHandler::CArrayHandler( const edm::TypeWithDict& dictionary ):
  m_type( dictionary ),
  m_elementType()
{

  // find the iterator return type 
  edm::TypeWithDict elementType = m_type.toType();
  m_elementType = ClassUtils::resolvedType( elementType );
  
}

ora::CArrayHandler::~CArrayHandler(){
}

size_t
ora::CArrayHandler::size( const void* )
{ 
  return ClassUtils::arrayLength( m_type );
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

edm::TypeWithDict&
ora::CArrayHandler::iteratorReturnType()
{
  return m_elementType;
}
