#ifndef CondCore_PluginSystem_DataProxy_H
#define CondCore_PluginSystem_DataProxy_H
//#include <iostream>
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"

#include "CondCore/IOVService/interface/PayloadProxy.h"


template< class RecordT, class DataT >
  class DataProxy : public edm::eventsetup::DataProxyTemplate<RecordT, DataT>{
  public:
  typedef DataProxy<RecordT,DataT> self;
  typedef boost::shared_ptr<cond::PayloadProxy<DataT> > DataP;

  explicit DataProxy(boost::shared_ptr<cond::PayloadProxy<DataT> > pdata) : m_data(pdata) { 
 
  }
  //virtual ~DataProxy();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  
  protected:
  virtual const DataT* make(const RecordT&, const edm::eventsetup::DataKey&) {
    m_data->make();
    return &(*m_data)();
  }
  virtual void invalidateCache() {
    m_data->invalidateCache();
  }
  private:
  //DataProxy(); // stop default
  const DataProxy& operator=( const DataProxy& ); // stop default
  // ---------- member data --------------------------------

  boost::shared_ptr<cond::PayloadProxy<DataT> >  m_data;

};

namespace cond {
  class DataProxyWrapperBase {
  public:
    typedef boost::shared_ptr<cond::BasePayloadProxy> ProxyP;
    typedef boost::shared_ptr<edm::eventsetup::DataProxy> edmProxyP;
    

    virtual edm::eventsetup::TypeTag type() const=0;
    virtual ProxyP proxy() const=0;
    virtual edmProxyP emdProxy() const=0;


    DataProxyWrapperBase(std::string const & il) : m_label(il){}
    virtual ~DataProxyWrapperBase(){}
    std::string const & label() const { return m_label;}
    
  private:
    std::string m_label;

  };
}

template< class RecordT, class DataT >
class DataProxyWrapper : public  cond::DataProxyWrapperBase {
public:
  typedef cond::PayloadProxy<DataT> PayProxy;
  typedef boost::shared_ptr<PayProxy> DataP;

  
  DataProxyWrapper(cond::Connection& conn,
		   const std::string & token, std::string const & il) :
    cond::DataProxyWrapperBase(il),
    m_proxy(new PayProxy(conn,token)),
    m_edmProxy(m_proxy){
   //NOTE: We do this so that the type 'DataT' will get registered
    // when the plugin is dynamically loaded
    //std::cout<<"DataProxy constructor"<<std::endl;
    m_type = edm::eventsetup::DataKey::makeTypeTag<DataT>();
    //std::cout<<"about to get out of DataProxy constructor"<<std::endl;
  }
    
  virtual edm::eventsetup::TypeTag type() const { return m_type;}
  virtual ProxyP proxy() const { return m_proxy;}
  virtual edmProxyP emdProxy() const { return m_edmProxy;}
 
private:
  edm::eventsetup::TypeTag m_type;
  boost::shared_ptr<cond::PayloadProxy<DataT> >  m_proxy;
  edmProxyP m_edmProxy;

};


#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
