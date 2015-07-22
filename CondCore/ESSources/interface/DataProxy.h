#ifndef CondCore_ESSources_DataProxy_H
#define CondCore_ESSources_DataProxy_H
//#include <iostream>
#include <string>
#include "boost/shared_ptr.hpp"

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
    typedef boost::shared_ptr<cond::persistency::PayloadProxy<DataT> > DataP;

    explicit DataProxy(boost::shared_ptr<cond::persistency::PayloadProxy<DataT> > pdata) : m_data(pdata) { 
 
  }
  //virtual ~DataProxy();
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  
  // ---------- member functions ---------------------------
  
  protected:
  virtual const DataT* make(const RecordT&, const edm::eventsetup::DataKey&) {
    m_data->make();
    m_initializer(const_cast<DataT&>((*m_data)()));
    return &(*m_data)();
  }
  virtual void invalidateCache() {
    // don't, preserve data for future access
    // m_data->invalidateCache();
  }
  virtual void invalidateTransientCache() {
    m_data->invalidateTransientCache();
  }
  private:
  //DataProxy(); // stop default
  const DataProxy& operator=( const DataProxy& ); // stop default
  // ---------- member data --------------------------------

  boost::shared_ptr<cond::persistency::PayloadProxy<DataT> >  m_data;
  Initializer m_initializer;
};

namespace cond {

  /* ABI bridging between the cond world and eventsetup world
   * keep them separated!
   */
  class DataProxyWrapperBase {
  public:
    typedef boost::shared_ptr<cond::persistency::BasePayloadProxy> ProxyP;
    typedef boost::shared_ptr<edm::eventsetup::DataProxy> edmProxyP;
    
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
  typedef boost::shared_ptr<PayProxy> DataP;
  
  
  DataProxyWrapper(cond::persistency::Session& session,
		   const std::string& tag, const std::string& ilabel, const char * source=0) :
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
  explicit DataProxyWrapper(const char * source=0) : m_source (source ? source : "") {
    //NOTE: We do this so that the type 'DataT' will get registered
    // when the plugin is dynamically loaded
    m_type = edm::eventsetup::DataKey::makeTypeTag<DataT>();
  }

  // late initialize (to allow to load ALL library first)
  virtual void lateInit(cond::persistency::Session& session, const std::string & tag, const boost::posix_time::ptime& snapshotTime,
			std::string const & il, std::string const & cs) {
    m_proxy.reset(new PayProxy(m_source.empty() ?  (const char *)(0) : m_source.c_str() ) );
    m_proxy->setUp( session );
    m_proxy->loadTag( tag, snapshotTime );
    m_edmProxy.reset(new DataProxy(m_proxy));
    addInfo(il, cs, tag);
  }
    
  virtual edm::eventsetup::TypeTag type() const { return m_type;}
  virtual ProxyP proxy() const { return m_proxy;}
  virtual edmProxyP edmProxy() const { return m_edmProxy;}
 
private:
  std::string m_source;
  edm::eventsetup::TypeTag m_type;
  boost::shared_ptr<cond::persistency::PayloadProxy<DataT> >  m_proxy;
  edmProxyP m_edmProxy;

};


#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
