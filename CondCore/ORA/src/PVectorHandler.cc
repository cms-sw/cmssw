#include "CondCore/ORA/interface/Exception.h"
#include "PVectorHandler.h"
#include "ClassUtils.h"

ora::PVectorIteratorHandler::PVectorIteratorHandler( const Reflex::Environ<long>& collEnv,
                                                     Reflex::CollFuncTable& collProxy,
                                                     const Reflex::Type& iteratorReturnType,
                                                     size_t startElement):
  m_returnType(iteratorReturnType),
  m_collEnv(collEnv),
  m_collProxy(collProxy),
  m_currentElement(0),
  m_startElement(startElement){
  // retrieve the first element
  m_currentElement = m_collProxy.first_func(&m_collEnv);

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
  // this is requiredd! It sets the number of memory slots (of size sizeof(Class)) to be used for the step
  m_collEnv.fIdx = 1;
  m_currentElement = m_collProxy.next_func(&m_collEnv);
}

void*
ora::PVectorIteratorHandler::object(){
  return m_currentElement;
}

Reflex::Type&
ora::PVectorIteratorHandler::returnType(){
  return m_returnType;
}

ora::PVectorHandler::PVectorHandler( const Reflex::Type& dictionary ):
  m_type( dictionary ),
  m_iteratorReturnType(),
  m_isAssociative( false ),
  m_collEnv(),
  m_collProxy(),
  m_persistentSizeAttributeOffset(0),
  m_vecAttributeOffset(0)
{
  Reflex::Member privateVectorAttribute = m_type.DataMemberByName("m_vec");
  if(privateVectorAttribute){
    m_vecAttributeOffset = privateVectorAttribute.Offset();
    Reflex::Member method = privateVectorAttribute.TypeOf().MemberByName("createCollFuncTable");
    if(method){
      Reflex::CollFuncTable* collProxyPtr;
      method.Invoke(collProxyPtr);
      m_collProxy.reset( collProxyPtr );
    }
    if(! m_collProxy.get() ){
      throwException( "Cannot find \"createCollFuncTable\" function for type \""+m_type.Name(Reflex::SCOPED)+"\"",
                      "PVectorHandler::PVectorHandler");
    }
  }

  Reflex::Member persistentSizeAttribute = m_type.DataMemberByName("m_persistentSize");
  if( persistentSizeAttribute ){
    m_persistentSizeAttributeOffset = persistentSizeAttribute.Offset();
  }

  // find the iterator return type as the member type_value of the containers
  Reflex::Type valueType = ClassUtils::containerValueType( m_type );
  m_iteratorReturnType = ClassUtils::resolvedType( valueType );
}

ora::PVectorHandler::~PVectorHandler(){
}

size_t
ora::PVectorHandler::size( const void* address ){
  m_collEnv.fObject = static_cast<char*>(const_cast<void*>(address))+m_vecAttributeOffset;
  size_t transientSize = *(static_cast<size_t*>(m_collProxy->size_func(&m_collEnv)));
  return transientSize;
}

size_t
ora::PVectorHandler::startElementIndex( const void* address ){
  void* persistentSizeAddress = static_cast<char*>(const_cast<void*>(address))+m_persistentSizeAttributeOffset;
  size_t persistentSize = *static_cast<size_t*>(persistentSizeAddress);
  size_t transientSize = *(static_cast<size_t*>(m_collProxy->size_func(&m_collEnv)));
  size_t startElement = 0;
  if(persistentSize < transientSize) startElement = persistentSize;
  return startElement;
}

size_t ora::PVectorHandler::persistentSize( const void* address ){
  void* persistentSizeAddress = static_cast<char*>(const_cast<void*>(address))+m_persistentSizeAttributeOffset;
  size_t persistentSize = *static_cast<size_t*>(persistentSizeAddress);
  return persistentSize;
}

ora::IArrayIteratorHandler*
ora::PVectorHandler::iterate( const void* address ){
  m_collEnv.fObject = static_cast<char*>(const_cast<void*>(address))+m_vecAttributeOffset;
  if ( ! m_iteratorReturnType ) {
    throwException( "Missing the dictionary information for the value_type member of the container \"" +
                    m_type.Name(Reflex::SCOPED|Reflex::FINAL) + "\"",
                    "PVectorHandler" );
  }
  return new PVectorIteratorHandler( m_collEnv,*m_collProxy,m_iteratorReturnType,startElementIndex(address) );
}

void
ora::PVectorHandler::appendNewElement( void* address, void* data ){
  m_collEnv.fObject = static_cast<char*>(address)+m_vecAttributeOffset;
  m_collEnv.fSize = 1;
  m_collEnv.fStart = data;
  m_collProxy->feed_func(&m_collEnv);
}

void
ora::PVectorHandler::clear( const void* address ){
  m_collEnv.fObject = static_cast<char*>(const_cast<void*>(address))+m_vecAttributeOffset;
  m_collProxy->clear_func(&m_collEnv);
}

Reflex::Type&
ora::PVectorHandler::iteratorReturnType() {
  return m_iteratorReturnType;
}

void ora::PVectorHandler::finalize( void* address ){
  size_t transSize = size( address );
  void* persistentSizeAttributeAddress = static_cast<char*>(address)+m_persistentSizeAttributeOffset;
  *static_cast<size_t*>(persistentSizeAttributeAddress) = transSize;
}

                        
                        
                        
