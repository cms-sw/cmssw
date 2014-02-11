#include "CondCore/ORA/interface/Exception.h"

#include "ClassUtils.h"
#include "STLContainerHandler.h"
// externals
#include "RVersion.h"

#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "oraHelper.h"

ora::STLContainerIteratorHandler::STLContainerIteratorHandler( void* address,
                                                               TVirtualCollectionProxy& collProxy,
                                                               const edm::TypeWithDict& iteratorReturnType ):
  m_returnType(iteratorReturnType),
  m_collProxy(collProxy),
  m_currentElement(0),
  m_Iterators(nullptr),
  m_PtrIterators(nullptr),
  m_Next()
{

  if (m_collProxy.HasPointers()) {
    m_PtrIterators = new TVirtualCollectionPtrIterators(&m_collProxy);
  } else {
    m_Iterators = new TVirtualCollectionIterators(&m_collProxy);
  }
  m_Next = m_collProxy.GetFunctionNext();

  if (m_collProxy.HasPointers()) {
    m_PtrIterators->CreateIterators(address, &m_collProxy);
  } else {
    m_Iterators->CreateIterators(address, &m_collProxy);
  }

  if (m_collProxy.HasPointers()) {
    m_currentElement = m_Next(m_PtrIterators->fBegin,m_PtrIterators->fEnd);
  } else {
    m_currentElement = m_Next(m_Iterators->fBegin,m_Iterators->fEnd); 
  }
  //m_currentElement = m_collProxy.first_func(&m_collEnv);
}

ora::STLContainerIteratorHandler::~STLContainerIteratorHandler(){}

void
ora::STLContainerIteratorHandler::increment(){
  if (m_collProxy.HasPointers())  {
    m_currentElement = m_Next(m_PtrIterators->fBegin,m_PtrIterators->fEnd);
  } else {
    m_currentElement = m_Next(m_Iterators->fBegin,m_Iterators->fEnd); 
  }
  //m_currentElement = m_collProxy.next_func(&m_collEnv);
}

void*
ora::STLContainerIteratorHandler::object()
{
  return m_currentElement;
}


edm::TypeWithDict&
ora::STLContainerIteratorHandler::returnType()
{
  return m_returnType;
}

ora::STLContainerHandler::STLContainerHandler( const edm::TypeWithDict& dictionary ):
  m_type( dictionary ),
  m_iteratorReturnType(),
  m_isAssociative( false ),
  m_collProxy(){
  m_isAssociative = ClassUtils::isTypeKeyedContainer( m_type );

  TClass* cl = dictionary.getClass();
  m_collProxy = cl->GetCollectionProxy();
  if( !m_collProxy ){
    throwException( "Cannot create \"TVirtualCollectionProxy\" for type \""+m_type.qualifiedName()+"\"",
                    "STLContainerHandler::STLContainerHandler");
  }

  // find the iterator return type as the member type_value of the containers
  edm::TypeWithDict valueType = ClassUtils::containerValueType( m_type );
  m_iteratorReturnType = ClassUtils::resolvedType( valueType );

}

ora::STLContainerHandler::~STLContainerHandler(){
}

size_t
ora::STLContainerHandler::size( const void* address ){
  //m_collEnv.fObject = const_cast<void*>(address);
  //return *(static_cast<size_t*>(m_collProxy->size_func(&m_collEnv)));
  TVirtualCollectionProxy::TPushPop helper(m_collProxy, const_cast<void*>(address));
  return m_collProxy->Size();
}


ora::IArrayIteratorHandler*
ora::STLContainerHandler::iterate( const void* address ){
  if ( ! m_iteratorReturnType ) {
    throwException( "Missing the dictionary information for the value_type member of the container \"" +
                    m_type.qualifiedName() + "\"",
                    "STLContainerHandler::iterate" );
  }
  void *addr = const_cast<void*>(address);
  return new STLContainerIteratorHandler( addr,*m_collProxy,m_iteratorReturnType );
}


void
ora::STLContainerHandler::appendNewElement( void* address, void* data ){
  // m_collProxy->feed_func(data,address,1);
  m_collProxy->Insert(data, address, 1);
}

void
ora::STLContainerHandler::clear( const void* address ){
  //m_collEnv.fObject = const_cast<void*>(address);
  //m_collProxy->clear_func(&m_collEnv);
  TVirtualCollectionProxy::TPushPop helper(m_collProxy, const_cast<void*>(address));
  return m_collProxy->Clear();
}

edm::TypeWithDict&
ora::STLContainerHandler::iteratorReturnType(){
  return m_iteratorReturnType;
}

ora::SpecialSTLContainerHandler::SpecialSTLContainerHandler( const edm::TypeWithDict& dictionary ):
  m_containerHandler(),
  m_containerOffset( 0 )
{
  // update dictionary to include base classes members
  //-ap ignore for now:  dictionary.UpdateMembers();
  for ( unsigned int i=0;i<dictionary.dataMemberSize();i++){

    edm::MemberWithDict field = ora::helper::DataMemberAt(dictionary, i);
    edm::TypeWithDict fieldType = field.typeOf();
    if ( ! fieldType ) {
      throwException( "The dictionary of the underlying container of \"" +
                      dictionary.qualifiedName() + "\" is not available",
                      "SpecialSTLContainerHandler" );
    }
    if ( ClassUtils::isTypeContainer(fieldType) ) {
      m_containerHandler.reset( new STLContainerHandler( fieldType ) );
      m_containerOffset = field.offset();
      break;
    }
  }
  if ( !m_containerHandler.get() ) {
    throwException( "Could not retrieve the underlying container of \"" +
                    dictionary.qualifiedName() + "\" is not available",
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

edm::TypeWithDict&
ora::SpecialSTLContainerHandler::iteratorReturnType()
{
  return m_containerHandler->iteratorReturnType();
}
