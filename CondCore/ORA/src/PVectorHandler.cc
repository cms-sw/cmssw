#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/PVector.h"
#include "CondFormats/Common/interface/IOVElement.h"
#include "FWCore/Utilities/interface/MemberWithDict.h"
#include "PVectorHandler.h"
#include "ClassUtils.h"
// externals
#include "RVersion.h"
#include <cassert>
#include <cstring>

ora::PVectorIteratorHandler::PVectorIteratorHandler( void* address,
						     TVirtualCollectionProxy& collProxy,
						     const edm::TypeWithDict& iteratorReturnType,
						     size_t startElement):
  m_returnType(iteratorReturnType),
  m_collProxy(collProxy),
  m_currentElement(0),
  m_Iterators(TGenericCollectionIterator::New(address, &collProxy)),
  m_startElement(startElement){

  m_currentElement = m_Iterators->Next(); 

  if(startElement){
    size_t i = 0;
    while(i<startElement){
      increment();
      i++;
    }
  }
}

ora::PVectorIteratorHandler::~PVectorIteratorHandler(){
}

void
ora::PVectorIteratorHandler::increment(){
  m_currentElement = m_Iterators->Next(); 
}

void*
ora::PVectorIteratorHandler::object(){
  return m_currentElement;
}

edm::TypeWithDict&
ora::PVectorIteratorHandler::returnType(){
  return m_returnType;
}

ora::PVectorHandler::PVectorHandler( const edm::TypeWithDict& dictionary ):
  m_type( dictionary ),
  m_iteratorReturnType(),
  m_collProxy(),
  m_persistentSizeAttributeOffset(0),
  m_vecAttributeOffset(0)
{
  edm::MemberWithDict privateVectorAttribute = m_type.dataMemberByName("m_vec");
  if(privateVectorAttribute){
    m_vecAttributeOffset = privateVectorAttribute.offset();
    TClass* cl = privateVectorAttribute.typeOf().getClass();
    m_collProxy = cl->GetCollectionProxy();
    if( !m_collProxy ){
      throwException( "Cannot create \"TVirtualCollectionProxy\" for type \""+m_type.cppName()+"\"",
		      "PVectorHandler::PVectorHandler");
    }
  }
    
  edm::MemberWithDict persistentSizeAttribute = m_type.dataMemberByName("m_persistentSize");
  if( persistentSizeAttribute ){
    m_persistentSizeAttributeOffset = persistentSizeAttribute.offset();
  }

  // find the iterator return type as the member type_value of the containers
  edm::TypeWithDict valueType = ClassUtils::containerValueType( m_type );
  m_iteratorReturnType = ClassUtils::resolvedType( valueType );
}

ora::PVectorHandler::~PVectorHandler(){
}

size_t
ora::PVectorHandler::size( const void* address ){
  TVirtualCollectionProxy::TPushPop helper(m_collProxy, static_cast<char*>(const_cast<void*>(address))+m_vecAttributeOffset );
  return m_collProxy->Size();
}

size_t
ora::PVectorHandler::startElementIndex( const void* address ){
  const void* persistentSizeAddress = static_cast<const char *>(address) + m_persistentSizeAttributeOffset;
  size_t persistentSize = *static_cast<const size_t*>(persistentSizeAddress);
    
  TVirtualCollectionProxy::TPushPop helper(m_collProxy, static_cast<char*>(const_cast<void*>(address))+m_vecAttributeOffset );
  size_t transientSize = m_collProxy->Size();

  size_t startElement = 0;
  if(persistentSize < transientSize) startElement = persistentSize;
  return startElement;
}

size_t* ora::PVectorHandler::persistentSize( const void* address ){
  void* persistentSizeAddress = static_cast<char*>(const_cast<void*>(address))+m_persistentSizeAttributeOffset;
  return static_cast<size_t*>(persistentSizeAddress);
}

ora::IArrayIteratorHandler*
ora::PVectorHandler::iterate( const void* address ){
  if ( ! m_iteratorReturnType ) {
    throwException( "Missing the dictionary information for the value_type member of the container \"" +
                    m_type.cppName() + "\"",
                    "PVectorHandler" );
  }
  void *addr = static_cast<char*>(const_cast<void*>(address))+m_vecAttributeOffset;

  return new PVectorIteratorHandler( addr,*m_collProxy,m_iteratorReturnType,startElementIndex(address) );
}

void
ora::PVectorHandler::appendNewElement( void* address, void* data ){

  void* addr = static_cast<char*>(const_cast<void*>(address))+m_vecAttributeOffset;
  m_collProxy->Insert(data, addr, 1);
}

void
ora::PVectorHandler::clear( const void* address ){
  void* addr = static_cast<char*>(const_cast<void*>(address))+m_vecAttributeOffset;
  TVirtualCollectionProxy::TPushPop helper(m_collProxy, const_cast<void*>(addr));
  m_collProxy->Clear();
}

edm::TypeWithDict&
ora::PVectorHandler::iteratorReturnType() {
  return m_iteratorReturnType;
}

void ora::PVectorHandler::finalize( void* address ){
  size_t transSize = size( address );
  void* persistentSizeAttributeAddress = static_cast<char*>(address)+m_persistentSizeAttributeOffset;
  *static_cast<size_t*>(persistentSizeAttributeAddress) = transSize;
}

                        
                        
                        
