#ifndef CondCore_PluginSystem_DataProxy_H
#define CondCore_PluginSystem_DataProxy_H
//#include <iostream>
// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"
//#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Ref.h"

template< class RecordT, class DataT >
  class DataProxy : public edm::eventsetup::DataProxyTemplate<RecordT, DataT>{
  public:
  /*  DataProxy( pool::IDataSvc* svc, std::map<std::string,std::string>::iterator& pProxyToToken ): m_svc(svc), m_pProxyToToken(pProxyToToken) { 
  //NOTE: We do this so that the type 'DataT' will get registered
  // when the plugin is dynamically loaded
  edm::eventsetup::DataKey::makeTypeTag<DataT>(); 
  }
  */
  DataProxy( cond::PoolStorageManager* pooldb, std::map<std::string,std::string>::iterator& pProxyToToken ): m_pooldb(pooldb), m_pProxyToToken(pProxyToToken) { 
    //NOTE: We do this so that the type 'DataT' will get registered
    // when the plugin is dynamically loaded
    //std::cout<<"DataProxy constructor"<<std::endl;
    edm::eventsetup::DataKey::makeTypeTag<DataT>();
    //std::cout<<"about to get out of DataProxy constructor"<<std::endl;
  }
  //virtual ~DataProxy();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  
  protected:
  virtual const DataT* make(const RecordT&, const edm::eventsetup::DataKey&) {
    try{
      //std::cout<<"DataT make "<<std::endl;
      m_pooldb->startTransaction(true);
      m_data=cond::Ref<DataT>(*m_pooldb,m_pProxyToToken->second);
      *m_data;
      m_pooldb->commit();
    }catch( const cond::Exception& er ){
      throw er;
    }catch( const std::exception& er ){
      throw cond::Exception( er.what() );
    }catch( ... ){
      throw cond::Exception( "Unknown error" );
    }
    //std::cout<<"returning ptr "<<m_data.ptr()<<std::endl;
    return m_data.ptr();
  }
  virtual void invalidateCache() {
    //m_data.clear();
    //std::cout<<"end invalidateCache"<<std::endl;
  }
  private:
  //DataProxy(); // stop default
  const DataProxy& operator=( const DataProxy& ); // stop default
  // ---------- member data --------------------------------
  //pool::IDataSvc* m_svc;
  cond::PoolStorageManager* m_pooldb;
  std::map<std::string,std::string>::iterator m_pProxyToToken;
  cond::Ref<DataT> m_data;
};
#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
