#include "CondCore/ORA/interface/Exception.h"

#include "ClassUtils.h"
#include "STLContainerHandler.h"
// externals
#include "RVersion.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

ora::STLContainerIteratorHandler::STLContainerIteratorHandler( void* address,
                                                               TVirtualCollectionProxy& collProxy,
                                                               const edm::TypeWithDict& iteratorReturnType ):
  m_returnType(iteratorReturnType),
  m_collProxy(collProxy),
  m_currentElement(nullptr),
  m_Iterators(TGenericCollectionIterator::New(address, &collProxy))
{
    m_currentElement = m_Iterators->Next();
}

ora::STLContainerIteratorHandler::~STLContainerIteratorHandler(){}

void
ora::STLContainerIteratorHandler::increment(){
  m_currentElement = m_Iterators->Next();
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
    throwException( "Cannot create \"TVirtualCollectionProxy\" for type \""+m_type.cppName()+"\"",
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
                    m_type.cppName() + "\"",
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
  edm::TypeDataMembers members(dictionary);
  for (auto const & member : members) {
    edm::MemberWithDict field(member);
    edm::TypeWithDict fieldType = field.typeOf();
    if ( ! fieldType ) {
      throwException( "The dictionary of the underlying container of \"" +
                      dictionary.cppName() + "\" is not available",
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
                    dictionary.cppName() + "\" is not available",
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
