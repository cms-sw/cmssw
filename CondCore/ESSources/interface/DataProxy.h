#ifndef CondCore_ESSources_DataProxy_H
#define CondCore_ESSources_DataProxy_H
//#include <iostream>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"

#include "CondCore/CondDB/interface/PayloadProxy.h"

// expose a cond::PayloadProxy as a eventsetup::DataProxy
namespace cond {
  template< typename DataT> struct DefaultInitializer {
    void operator()(DataT &){}
  };
}

template< class RecordT, class DataT , typename Initializer=cond::DefaultInitializer<DataT> >
class DataProxy : public edm::eventsetup::DataProxyTemplate<RecordT, DataT >{
  public:
  typedef DataProxy<RecordT,DataT> self;
    typedef std::shared_ptr<cond::persistency::PayloadProxy<DataT> > DataP;

    explicit DataProxy(std::shared_ptr<cond::persistency::PayloadProxy<DataT> > pdata) : m_data(pdata) { 
 
  }
  //virtual ~DataProxy();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  
  protected:
  const DataT* make(const RecordT&, const edm::eventsetup::DataKey&) override {
    m_data->make();
    m_initializer(const_cast<DataT&>((*m_data)()));
    return &(*m_data)();
  }
  void invalidateCache() override {
    // don't, preserve data for future access
    // m_data->invalidateCache();
  }
  void invalidateTransientCache() override {
    m_data->invalidateTransientCache();
  }
  private:
  //DataProxy(); // stop default
  const DataProxy& operator=( const DataProxy& ) = delete; // stop default
  // ---------- member data --------------------------------

  std::shared_ptr<cond::persistency::PayloadProxy<DataT> >  m_data;
  Initializer m_initializer;
};

namespace cond {

  /* ABI bridging between the cond world and eventsetup world
   * keep them separated!
   */
  class DataProxyWrapperBase {
  public:
    typedef std::shared_ptr<cond::persistency::BasePayloadProxy> ProxyP;
    typedef std::shared_ptr<edm::eventsetup::DataProxy> edmProxyP;
    
    // limitation of plugin manager...
    typedef std::pair< std::string, std::string> Args;

    virtual edm::eventsetup::TypeTag type() const=0;
    virtual ProxyP proxy() const=0;
    virtual edmProxyP edmProxy() const=0;


    DataProxyWrapperBase();
    explicit DataProxyWrapperBase(std::string const & il);
    // late initialize (to allow to load ALL library first)
    virtual void lateInit(cond::persistency::Session& session, const std::string & tag, const boost::posix_time::ptime& snapshotTime,
			  std::string const & il, std::string const & cs)=0;

    void addInfo(std::string const & il, std::string const & cs, std::string const & tag);
    

    virtual ~DataProxyWrapperBase();
    std::string const & label() const { return m_label;}

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
template< class RecordT, class DataT, typename Initializer=cond::DefaultInitializer<DataT> >
class DataProxyWrapper : public  cond::DataProxyWrapperBase {
public:
  typedef ::DataProxy<RecordT,DataT, Initializer> DataProxy;
  typedef cond::persistency::PayloadProxy<DataT> PayProxy;
  typedef std::shared_ptr<PayProxy> DataP;
  
  
  DataProxyWrapper(cond::persistency::Session& session,
		   const std::string& tag, const std::string& ilabel, const char * source=nullptr) :
    cond::DataProxyWrapperBase(ilabel),
    m_source( source ? source : "" ),
    m_proxy(new PayProxy( source)), //'errorPolicy set to true: PayloadProxy should catch and re-throw ORA exceptions' still needed?
    m_edmProxy(new DataProxy(m_proxy)){
    m_proxy->setUp( session );
    //NOTE: We do this so that the type 'DataT' will get registered
    // when the plugin is dynamically loaded
    m_type = edm::eventsetup::DataKey::makeTypeTag<DataT>();
  }

  // constructor from plugin...
  explicit DataProxyWrapper(const char * source=nullptr) : m_source (source ? source : "") {
    //NOTE: We do this so that the type 'DataT' will get registered
    // when the plugin is dynamically loaded
    m_type = edm::eventsetup::DataKey::makeTypeTag<DataT>();
  }

  // late initialize (to allow to load ALL library first)
  void lateInit(cond::persistency::Session& session, const std::string & tag, const boost::posix_time::ptime& snapshotTime,
			std::string const & il, std::string const & cs) override {
    m_proxy.reset(new PayProxy(m_source.empty() ?  (const char *)nullptr : m_source.c_str() ) );
    m_proxy->setUp( session );
    m_proxy->loadTag( tag, snapshotTime );
    m_edmProxy.reset(new DataProxy(m_proxy));
    addInfo(il, cs, tag);
  }
    
  edm::eventsetup::TypeTag type() const override { return m_type;}
  ProxyP proxy() const override { return m_proxy;}
  edmProxyP edmProxy() const override { return m_edmProxy;}
 
private:
  std::string m_source;
  edm::eventsetup::TypeTag m_type;
  std::shared_ptr<cond::persistency::PayloadProxy<DataT> >  m_proxy;
  edmProxyP m_edmProxy;

};


#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
