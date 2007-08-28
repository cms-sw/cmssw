#ifndef CondCore_DBCommon_ContainerIterator_h
#define CondCore_DBCommon_ContainerIterator_h
#include "Collection/Collection.h"
#include "CondCore/DBCommon/interface/Ref.h"
namespace cond{
  class PoolStorageManager;
  /*
   *wrapper around pool implicit collection
   **/
  template<typename DataT>
    class ContainerIterator{
    public:
    ContainerIterator( PoolStorageManager& pooldb, const std::string& containername):
      m_pooldb(&pooldb), m_collection(new pool::Collection<DataT>( &(pooldb.DataSvc()),"ImplicitCollection","PFN:" + pooldb.connectionString(),containername, pool::ICollection::READ )),m_it(m_collection->select()){
    }
    Ref<DataT> dataRef(){
      return m_data;
    }
    bool next(){ 
      if( m_it.next() ){
	m_data=cond::Ref<DataT>(*m_pooldb,m_it.ref());
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
    mutable cond::Ref<DataT> m_data;
    cond::PoolStorageManager* m_pooldb;
    pool::Collection<DataT>* m_collection;
    typename pool::Collection<DataT>::Iterator m_it;
  };
}//ns cond
#endif
