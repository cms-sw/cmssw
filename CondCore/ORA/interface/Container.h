#ifndef INCLUDE_ORA_CONTAINER_H
#define INCLUDE_ORA_CONTAINER_H

#include "Object.h"
#include "Handle.h"
//
#include <typeinfo>
#include <boost/shared_ptr.hpp>

namespace ora {

  class IteratorBuffer;
  class DatabaseContainer;
  
  class ContainerIterator {
    
    public:
    ContainerIterator( Handle<IteratorBuffer>& iteratorBuffer );
    
    ContainerIterator( const ContainerIterator& rhs );
    
    virtual ~ContainerIterator();

    ContainerIterator& operator=( const ContainerIterator& rhs );

    Object getItem();

    template <typename T> boost::shared_ptr<T> get();

    int itemId();

    bool next();

    void reset();

    private:
    boost::shared_ptr<void> getItemAsType( const std::type_info& asTypeInfo );

    private:
    Handle<IteratorBuffer> m_buffer;
  };
  
  class Container {
    
    public:
    Container( Handle<DatabaseContainer>& dbContainer );

    Container( const Container& rhs );

    virtual ~Container();

    Container& operator=( const Container& rhs );
    
    int id();

    const std::string& name();

    const std::string& className();

    const std::string& mappingVersion();

    size_t size();

    void extendSchema( const std::type_info& typeInfo );

    template <typename T> void extendSchema();

    ContainerIterator iterator();

    Object fetchItem( int itemId );

    template <typename T> boost::shared_ptr<T> fetch( int itemId );

    int insertItem( const Object& data );

    template <typename T> int insert( const T& data );

    void updateItem( int itemId, const Object& data );

    template <typename T> void update( int itemId, const T& data );

    void erase( int itemId );
    
    void flush();

    private:
    boost::shared_ptr<void> fetchItemAsType(int itemId, const std::type_info& asTypeInfo);
    int insertItem( const void* data, const std::type_info& typeInfo );
    void updateItem( int itemId, const void* data, const std::type_info& typeInfo );



    private:
    Handle<DatabaseContainer> m_dbContainer;
  };

}

template <typename T>
inline
boost::shared_ptr<T> ora::ContainerIterator::get(){
  return boost::static_pointer_cast<T>( getItemAsType( typeid(T) ));
}

template <typename T>
inline
void ora::Container::extendSchema(){
  extendSchema( typeid(T) );
}

template <typename T>
inline
boost::shared_ptr<T> ora::Container::fetch( int itemId){
  return boost::static_pointer_cast<T>( fetchItemAsType( itemId, typeid(T) ));
}

template <typename T>
inline
int ora::Container::insert( const T& data ){
  return insertItem( &data, typeid( data ) );
}

template <typename T>
inline
void ora::Container::update( int itemId, const T& data ){
  return updateItem( itemId, &data, typeid( data ) );
}

#endif
