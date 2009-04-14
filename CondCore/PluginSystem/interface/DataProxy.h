#ifndef CondCore_PluginSystem_DataProxy_H
#define CondCore_PluginSystem_DataProxy_H
//#include <iostream>
#include <map>
#include <string>
// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "DataSvc/Ref.h"
#include "DataSvc/RefException.h"

#include "CondFormats/Common/interface/PayloadWrapper.h"


template< class RecordT, class DataT >
  class DataProxy : public edm::eventsetup::DataProxyTemplate<RecordT, DataT>{
  public:
  typedef  boost::shared_ptr<cond::PayloadProxy<DataT> > DataP;

  */
  DataProxy(DatatP pdata) : m_data(pdata) { 
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
    m_data->make();
    return (*m_data)();
  }
  virtual void invalidateCache() {
    m_data.clear();
    m_OldData.clear();
  }
  private:
  //DataProxy(); // stop default
  const DataProxy& operator=( const DataProxy& ); // stop default
  // ---------- member data --------------------------------
  cond::Connection* m_connection;
  std::map<std::string,std::string>::iterator m_pDatumToToken;

  DatatP m_data;

};

class DataProxyWrapperBase {
public:
  typedef boost::shared_ptr<cond::BasePayloadProxy> ProxyP;
  typedef boost::shared_ptr<edm::eventsetup::DataProxy> edmProxyP;

  virtual ProxyP proxy() const=0;
  virtual edmProxyP emdProxy() const=0;

};

template< class RecordT, class DataT >
class DataProxyWrapper : public  DataProxyWrapperBase {
public:  
  typedef  boost::shared_ptr<cond::PayloadProxy<DataT> > DataP;

  
  DataProxyWrapper(cond::Connection& conn,
		   const std::string & token);

  virtual ProxyP proxy() const { return m_proxy;}
  virtual edmProxyP emdProxy() const { return m_edmProxy;}
 
private:

  DataP m_proxy;
  edmProxyP m_edmProxy;

};


#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
