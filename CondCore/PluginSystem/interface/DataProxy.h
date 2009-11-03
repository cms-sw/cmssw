#ifndef CondCore_PluginSystem_DataProxy_H
#define CondCore_PluginSystem_DataProxy_H
//#include <iostream>
#include <string>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"

#include "CondCore/IOVService/interface/PayloadProxy.h"

// expose a cond::PayloadProxy as a eventsetup::DataProxy
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

  /* ABI bridging between the cond world and eventsetup world
   * keep them separated!
   */
  class DataProxyWrapperBase {
  public:
    typedef boost::shared_ptr<cond::BasePayloadProxy> ProxyP;
    typedef boost::shared_ptr<edm::eventsetup::DataProxy> edmProxyP;
    
    // limitation of plugin manager...
    typedef std::pair< std::string, std::string> Args;

    virtual edm::eventsetup::TypeTag type() const=0;
    virtual ProxyP proxy() const=0;
    virtual edmProxyP edmProxy() const=0;


    explicit DataProxyWrapperBase(std::string const & il) : m_label(il){}
    virtual ~DataProxyWrapperBase(){}
    std::string const & label() const { return m_label;}

    void addInfo(std::string const & cs, std::string const & tag) { 
      m_connString = cs; m_tag=tag;
    }
    
    std::string const & connString() const { return m_connString;}
    std::string const & tag() const { return m_tag;}

  private:
    std::string m_label;
    std::string m_connString;
    std::string m_tag;

  };
}

/* bridge between the cond world and eventsetup world
 * keep them separated!
 */
template< class RecordT, class DataT >
class DataProxyWrapper : public  cond::DataProxyWrapperBase {
public:
  typedef ::DataProxy<RecordT,DataT> DataProxy;
  typedef cond::PayloadProxy<DataT> PayProxy;
  typedef boost::shared_ptr<PayProxy> DataP;
  
  
  DataProxyWrapper(cond::DbSession& session,
		   const std::string & iovtoken, std::string const & ilabel, const char * source=0) :
    cond::DataProxyWrapperBase(ilabel),
    m_proxy(new PayProxy(session,iovtoken,false, source)),
    m_edmProxy(new DataProxy(m_proxy)){
   //NOTE: We do this so that the type 'DataT' will get registered
    // when the plugin is dynamically loaded
    //std::cout<<"DataProxy constructor"<<std::endl;
    m_type = edm::eventsetup::DataKey::makeTypeTag<DataT>();
    //std::cout<<"about to get out of DataProxy constructor"<<std::endl;
  }

  // constructor from plugin...
  DataProxyWrapper(cond::DbSession& session,
		   Args const & args) :
    cond::DataProxyWrapperBase(args.second),
    m_proxy(new PayProxy(session,args.first,false)),
    m_edmProxy(new DataProxy(m_proxy)){
   //NOTE: We do this so that the type 'DataT' will get registered
    // when the plugin is dynamically loaded
    //std::cout<<"DataProxy constructor"<<std::endl;
    m_type = edm::eventsetup::DataKey::makeTypeTag<DataT>();
    //std::cout<<"about to get out of DataProxy constructor"<<std::endl;
  }

    
  virtual edm::eventsetup::TypeTag type() const { return m_type;}
  virtual ProxyP proxy() const { return m_proxy;}
  virtual edmProxyP edmProxy() const { return m_edmProxy;}
 
private:
  edm::eventsetup::TypeTag m_type;
  boost::shared_ptr<cond::PayloadProxy<DataT> >  m_proxy;
  edmProxyP m_edmProxy;

};


#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
