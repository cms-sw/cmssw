#include "CondCore/ORA/interface/Exception.h"
#include "STLContainerHandler.h"
#include "ClassUtils.h"
// externals
#include "RVersion.h"

ora::STLContainerIteratorHandler::STLContainerIteratorHandler( const Reflex::Environ<long>& collEnv,
                                                               Reflex::CollFuncTable& collProxy,
                                                               const Reflex::Type& iteratorReturnType ):
  m_returnType(iteratorReturnType),
  m_collEnv(collEnv),
  m_collProxy(collProxy),
  m_currentElement(0){

  // retrieve the first element
  m_currentElement = m_collProxy.first_func(&m_collEnv);
}

ora::STLContainerIteratorHandler::~STLContainerIteratorHandler(){}

void
ora::STLContainerIteratorHandler::increment(){
  // this is required! It sets the number of memory slots (of size sizeof(Class)) to be used for the step
  m_collEnv.fIdx = 1;
  m_currentElement = m_collProxy.next_func(&m_collEnv);
}

void*
ora::STLContainerIteratorHandler::object()
{
  return m_currentElement;
}


Reflex::Type&
ora::STLContainerIteratorHandler::returnType()
{
  return m_returnType;
}

ora::STLContainerHandler::STLContainerHandler( const Reflex::Type& dictionary ):
  m_type( dictionary ),
  m_iteratorReturnType(),
  m_isAssociative( false ),
  m_collEnv(),
  m_collProxy(){
  m_isAssociative = ClassUtils::isTypeKeyedContainer( m_type );

  Reflex::Member method = m_type.MemberByName("createCollFuncTable");
  if(method){
    Reflex::CollFuncTable* collProxyPtr;
    method.Invoke( collProxyPtr );
    m_collProxy.reset( collProxyPtr );
  }
  if( !m_collProxy.get() ){
    throwException( "Cannot find \"createCollFuncTable\" function for type \""+m_type.Name(Reflex::SCOPED)+"\"",
                    "STLContainerHandler::STLContainerHandler");
  }

  // find the iterator return type as the member type_value of the containers
  Reflex::Type valueType = ClassUtils::containerValueType( m_type );
  m_iteratorReturnType = ClassUtils::resolvedType( valueType );
  
}

ora::STLContainerHandler::~STLContainerHandler(){
}

size_t
ora::STLContainerHandler::size( const void* address ){
  m_collEnv.fObject = const_cast<void*>(address);
  return *(static_cast<size_t*>(m_collProxy->size_func(&m_collEnv)));
}


ora::IArrayIteratorHandler*
ora::STLContainerHandler::iterate( const void* address ){
  if ( ! m_iteratorReturnType ) {
    throwException( "Missing the dictionary information for the value_type member of the container \"" +
                    m_type.Name(Reflex::SCOPED|Reflex::FINAL) + "\"",
                    "STLContainerHandler::iterate" );
  }
  m_collEnv.fObject = const_cast<void*>(address);
  return new STLContainerIteratorHandler( m_collEnv,*m_collProxy,m_iteratorReturnType );
}


void
ora::STLContainerHandler::appendNewElement( void* address, void* data ){
#if ROOT_VERSION_CODE < ROOT_VERSION(5,28,0)
  m_collEnv.fObject = address;
  m_collEnv.fSize = 1;
  m_collEnv.fStart = data;
  m_collProxy->feed_func(&m_collEnv);
#else
  m_collProxy->feed_func(data,address,1);
#endif
}

void
ora::STLContainerHandler::clear( const void* address ){
  m_collEnv.fObject = const_cast<void*>(address);
  m_collProxy->clear_func(&m_collEnv);
}

Reflex::Type&
ora::STLContainerHandler::iteratorReturnType(){
  return m_iteratorReturnType;
}

ora::SpecialSTLContainerHandler::SpecialSTLContainerHandler( const Reflex::Type& dictionary ):
  m_containerHandler(),
  m_containerOffset( 0 )
{
  // update dictionary to include base classes members
  dictionary.UpdateMembers(); 
  for ( unsigned int i=0;i<dictionary.DataMemberSize();i++){

    Reflex::Member field = dictionary.DataMemberAt(i);    
    Reflex::Type fieldType = field.TypeOf();
    if ( ! fieldType ) {
      throwException( "The dictionary of the underlying container of \"" +
                      dictionary.Name(Reflex::SCOPED|Reflex::FINAL) + "\" is not available",
                      "SpecialSTLContainerHandler" );
    }
    if ( ClassUtils::isTypeContainer(fieldType) ) {
      m_containerHandler.reset( new STLContainerHandler( fieldType ) );
      m_containerOffset = field.Offset();
      break;
    }
  }
  if ( !m_containerHandler.get() ) {
    throwException( "Could not retrieve the underlying container of \"" +
                    dictionary.Name(Reflex::SCOPED|Reflex::FINAL) + "\" is not available",
                    "SpecialSTLContainerHandler" );
  }
}


ora::SpecialSTLContainerHandler::~SpecialSTLContainerHandler()
{
}


size_t
ora::SpecialSTLContainerHandler::size( const void* address )
{
  return m_containerHandler->size( static_cast< const char* >( address ) + m_containerOffset );
}


ora::IArrayIteratorHandler*
ora::SpecialSTLContainerHandler::iterate( const void* address )
{
  return m_containerHandler->iterate( static_cast< const char* >( address ) + m_containerOffset );
}


void
ora::SpecialSTLContainerHandler::appendNewElement( void* address, void* data )
{
  m_containerHandler->appendNewElement( static_cast< char* >( address ) + m_containerOffset, data );
}

void 
ora::SpecialSTLContainerHandler::clear( const void* address )
{
  m_containerHandler->clear( static_cast< const char* >( address ) + m_containerOffset );
}

Reflex::Type&
ora::SpecialSTLContainerHandler::iteratorReturnType()
{
  return m_containerHandler->iteratorReturnType();
}
