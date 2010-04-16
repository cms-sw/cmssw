#include "CondCore/ORA/interface/Container.h"
#include "DatabaseContainer.h"
#include "ClassUtils.h"

ora::ContainerIterator::ContainerIterator( Handle<IteratorBuffer>& iteratorBuffer ):
  m_buffer( iteratorBuffer ){
}

ora::ContainerIterator::~ContainerIterator(){
}

void ora::ContainerIterator::reset(){
  m_buffer->reset();
}

bool ora::ContainerIterator::next(){
  return m_buffer->next();
}

ora::Object ora::ContainerIterator::getItem(){
  return Object( m_buffer->getItem(), m_buffer->type() );
}

boost::shared_ptr<void> ora::ContainerIterator::getItemAsType( const std::type_info& asTypeInfo ){
  Reflex::Type castType = ClassUtils::lookupDictionary( asTypeInfo );
  void* ptr = m_buffer->getItemAsType( castType );
  return boost::shared_ptr<void>( ptr, RflxDeleter( m_buffer->type() ) );
}

ora::Container::Container( Handle<DatabaseContainer>& dbContainer ):
  m_dbContainer( dbContainer ){
}

ora::Container::~Container(){
}

int ora::Container::id(){
  return m_dbContainer->id();
}

const std::string& ora::Container::name(){
  return m_dbContainer->name();
}

const std::string& ora::Container::className(){
  return m_dbContainer->className();
}

const std::string& ora::Container::mappingVersion(){
  return m_dbContainer->mappingVersion();
}

size_t ora::Container::size(){
  return m_dbContainer->size();
}

ora::ContainerIterator ora::Container::iterator(){
  Handle<IteratorBuffer> buff = m_dbContainer->iteratorBuffer();
  return ContainerIterator( buff );
}

void ora::Container::extendSchema( const std::type_info& typeInfo ){
  Reflex::Type type = ClassUtils::lookupDictionary( typeInfo );
  m_dbContainer->extendSchema( type );
}

ora::Object ora::Container::fetchItem(int itemId){
  return Object( m_dbContainer->fetchItem(itemId), m_dbContainer->type() );
}

boost::shared_ptr<void> ora::Container::fetchItemAsType(int itemId,
                                                        const std::type_info& asTypeInfo){
  Reflex::Type asType = ClassUtils::lookupDictionary( asTypeInfo );
  void* ptr = m_dbContainer->fetchItemAsType(itemId, asType );
  return boost::shared_ptr<void>( ptr, RflxDeleter( m_dbContainer->type() ) );
}

int ora::Container::insertItem( const Object& data ){
  const Reflex::Type& objType = data.type();
  if(!objType){
    throwException("Object class has not been found in the dictionary.",
                   "Container::insertItem");
  }
  return m_dbContainer->insertItem( data.address(), objType );
}

int ora::Container::insertItem( const void* data,
                                         const std::type_info& typeInfo ){
  Reflex::Type type = ClassUtils::lookupDictionary( typeInfo );
  return m_dbContainer->insertItem( data, type );
}

void ora::Container::updateItem( int itemId,
                                 const Object& data ){
  const Reflex::Type& objType = data.type();
  if(!objType){
    throwException("Object class has not been found in the dictionary.",
                   "Container::updateItem");
  }
  return m_dbContainer->updateItem( itemId, data.address(), objType );
}

void ora::Container::updateItem( int itemId,
                                 const void* data,
                                 const std::type_info& typeInfo ){
  Reflex::Type type = ClassUtils::lookupDictionary( typeInfo );
  m_dbContainer->updateItem( itemId, data, type );
}

void ora::Container::erase( int itemId ){
  m_dbContainer->erase(itemId );
}

void ora::Container::flush(){
  m_dbContainer->flush();
}



