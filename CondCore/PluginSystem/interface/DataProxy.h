#ifndef CondCore_PluginSystem_DataProxy_H
#define CondCore_PluginSystem_DataProxy_H
//#include <iostream>
#include <map>
#include <string>
// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"
//#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/TypedRef.h"

template< class RecordT, class DataT >
  class DataProxy : public edm::eventsetup::DataProxyTemplate<RecordT, DataT>{
  public:
  /*  DataProxy( pool::IDataSvc* svc, std::map<std::string,std::string>::iterator& pProxyToToken ): m_svc(svc), m_pProxyToToken(pProxyToToken) { 
  //NOTE: We do this so that the type 'DataT' will get registered
  // when the plugin is dynamically loaded
  edm::eventsetup::DataKey::makeTypeTag<DataT>(); 
  }
  */
  DataProxy( cond::Connection* connection, std::map<std::string,std::string>::iterator& pProxyToToken ): m_connection(connection), m_pProxyToToken(pProxyToToken) { 
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
    DataT* result=0;
    try{
      //std::cout<<"DataT make "<<std::endl;
      cond::PoolTransaction& pooldb=m_connection->poolTransaction(true);
      pooldb.start();
      cond::TypedRef<DataT> data(pooldb,m_pProxyToToken->second);
      *data;
      result=data.ptr();
      pooldb.commit();
    }catch( const cond::Exception& er ){
      throw er;
    }catch( const std::exception& er ){
      throw cond::Exception( er.what() );
    }catch( ... ){
      throw cond::Exception( "Unknown error" );
    }
    if(!result){
      throw cond::Exception("DataProxy::make: null result");
    }
    return result;
  }
  virtual void invalidateCache() {
    //m_data.clear();
    //std::cout<<"end invalidateCache"<<std::endl;
  }
  private:
  //DataProxy(); // stop default
  const DataProxy& operator=( const DataProxy& ); // stop default
  // ---------- member data --------------------------------
  cond::Connection* m_connection;
  std::map<std::string,std::string>::iterator m_pProxyToToken;
  //cond::TypedRef<DataT> m_data;
};
#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
