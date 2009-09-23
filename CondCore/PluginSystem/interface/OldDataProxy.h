#ifndef CondCore_PluginSystem_OldDataProxy_H
#define CondCore_PluginSystem_OldDataProxy_H
//#include <iostream>
#include <map>
#include <string>
// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "DataSvc/Ref.h"
#include "DataSvc/RefException.h"

#include "CondFormats/Common/interface/PayloadWrapper.h"


template< class RecordT, class DataT >
  class OldDataProxy : public edm::eventsetup::DataProxyTemplate<RecordT, DataT>{
  public:
  typedef cond::DataWrapper<DataT> DataWrapper;
  /*  DataProxy( pool::IDataSvc* svc, std::map<std::string,std::string>::iterator& pProxyToToken ): m_svc(svc), m_pProxyToToken(pProxyToToken) { 
  //NOTE: We do this so that the type 'DataT' will get registered
  // when the plugin is dynamically loaded
  edm::eventsetup::DataKey::makeTypeTag<DataT>(); 
  }
  */
  OldDataProxy( cond::DbSession session, std::map<std::string,std::string>::iterator& pDatumToToken ): m_session(session), m_pDatumToToken(pDatumToToken) { 
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
    DataT const * result=0;
    //std::cout<<"DataT make "<<std::endl;
    m_session.transaction().start(true);      
    // FIXME (clean this mess)
    try {
      pool::Ref<DataWrapper> mydata = m_session.getTypedObject<DataWrapper>(m_pDatumToToken->second);
      if (mydata) {
        try{
          result = &mydata->data();
        } catch( const pool::Exception& e) {
          throw cond::Exception("DataProxy::make: null result");
        }
        m_data.copyShallow(mydata);
        m_session.transaction().commit();
        return result;
      }
    } catch(const pool::Exception&){}

    // compatibility mode....
    pool::Ref<DataT> myodata = m_session.getTypedObject<DataT>(m_pDatumToToken->second);
    result = myodata.ptr();
    if (!result) throw cond::Exception("DataProxy::make: null result");
    m_OldData.copyShallow(myodata);

    m_session.transaction().commit();
    return result;
  }
  virtual void invalidateCache() {
    m_data.clear();
    m_OldData.clear();
  }
  private:
  //DataProxy(); // stop default
  const OldDataProxy& operator=( const OldDataProxy& ); // stop default
  // ---------- member data --------------------------------
  cond::DbSession m_session;
  std::map<std::string,std::string>::iterator m_pDatumToToken;

  pool::Ref<DataWrapper> m_data;
  // Backward compatibility
  pool::Ref<DataT> m_OldData;
};
#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
