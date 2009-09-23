#ifndef CondCore_DBCommon_ContainerIterator_h
#define CondCore_DBCommon_ContainerIterator_h
#include "Collection/Collection.h"
#include "CondCore/DBCommon/interface/DbSession.h"

//#include <iostream>
namespace cond{
  class Connection;
  /*
   *wrapper around pool implicit collection
   **/
  template<typename DataT>
    class ContainerIterator{
    public:
    ContainerIterator( cond::DbSession& pooldb,
                       const std::string& containername):
      m_pooldb(pooldb), m_collection(new pool::Collection<DataT>( &(pooldb.poolCache()),"ImplicitCollection","PFN:" + pooldb.connectionString(),containername, pool::ICollection::READ )),m_it(m_collection->select()){
    }
    std::string dataToken(){
      return m_data.toString();
    }
    pool::Ref<DataT>& dataRef(){
      return m_data;
    }
    bool next(){ 
      if( m_it.next() ){
        m_data = m_it.ref();
        return true;
      }
      return false;
    }
    std::string name(){
      return m_collection->name();
    }
    virtual ~ContainerIterator(){
      delete m_collection;
    }
    private:
    mutable pool::Ref<DataT> m_data;
    cond::DbSession m_pooldb;
    pool::Collection<DataT>* m_collection;
    typename pool::Collection<DataT>::Iterator m_it;
  };
}//ns cond
#endif
