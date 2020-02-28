#ifndef CondCore_ESSources_DataProxy_H
#define CondCore_ESSources_DataProxy_H

#include <cassert>
//#include <iostream>
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/DataProxyTemplate.h"

#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"

// expose a cond::PayloadProxy as a eventsetup::DataProxy
namespace cond {
  template <typename DataT>
  struct DefaultInitializer {
    void operator()(DataT&) {}
  };
}  // namespace cond

template <class RecordT, class DataT, typename Initializer = cond::DefaultInitializer<DataT>>
class DataProxy : public edm::eventsetup::DataProxyTemplate<RecordT, DataT> {
public:
  explicit DataProxy(std::shared_ptr<cond::persistency::PayloadProxy<DataT>> pdata) : m_data(pdata) {}
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
  void invalidateTransientCache() override {}

private:
  //DataProxy(); // stop default
  const DataProxy& operator=(const DataProxy&) = delete;  // stop default

  void initializeForNewIOV() override { m_data->initializeForNewIOV(); }

  // ---------- member data --------------------------------

  std::shared_ptr<cond::persistency::PayloadProxy<DataT>> m_data;
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
    typedef std::pair<std::string, std::string> Args;

    virtual edm::eventsetup::TypeTag type() const = 0;
    virtual ProxyP proxy(unsigned int iovIndex) const = 0;
    virtual edmProxyP edmProxy(unsigned int iovIndex) const = 0;

    DataProxyWrapperBase();
    // late initialize (to allow to load ALL library first)
    virtual void lateInit(persistency::Session& session,
                          const std::string& tag,
                          const boost::posix_time::ptime& snapshotTime,
                          std::string const& il,
                          std::string const& cs) = 0;

    virtual void initConcurrentIOVs(unsigned int nConcurrentIOVs) = 0;

    void addInfo(std::string const& il, std::string const& cs, std::string const& tag);

    virtual ~DataProxyWrapperBase();

    std::string const& label() const { return m_label; }
    std::string const& connString() const { return m_connString; }
    std::string const& tag() const { return m_tag; }
    persistency::IOVProxy& iovProxy() { return m_iovProxy; }
    persistency::IOVProxy const& iovProxy() const { return m_iovProxy; }
    Iov_t const& currentIov() const { return m_currentIov; }
    persistency::Session& session() { return m_session; }
    persistency::Session const& session() const { return m_session; }
    std::shared_ptr<std::vector<Iov_t>> const& requests() const { return m_requests; }

    void setSession(persistency::Session const& v) { m_session = v; }

    void loadTag(std::string const& tag);
    void loadTag(std::string const& tag, boost::posix_time::ptime const& snapshotTime);
    void reload();

    ValidityInterval setIntervalFor(Time_t target);
    TimeType timeType() const { return m_iovProxy.timeType(); }

  private:
    std::string m_label;
    std::string m_connString;
    std::string m_tag;
    persistency::IOVProxy m_iovProxy;
    Iov_t m_currentIov;
    persistency::Session m_session;
    std::shared_ptr<std::vector<Iov_t>> m_requests;
  };
}  // namespace cond

/* bridge between the cond world and eventsetup world
 * keep them separated!
 */
template <class RecordT, class DataT, typename Initializer = cond::DefaultInitializer<DataT>>
class DataProxyWrapper : public cond::DataProxyWrapperBase {
public:
  typedef ::DataProxy<RecordT, DataT, Initializer> DataProxy;

  // constructor from plugin...
  explicit DataProxyWrapper(const char* source = nullptr) : m_source(source ? source : "") {
    //NOTE: We do this so that the type 'DataT' will get registered
    // when the plugin is dynamically loaded
    m_type = edm::eventsetup::DataKey::makeTypeTag<DataT>();
  }

  // late initialize (to allow to load ALL library first)
  void lateInit(cond::persistency::Session& iSession,
                const std::string& tag,
                const boost::posix_time::ptime& snapshotTime,
                std::string const& il,
                std::string const& cs) override {
    setSession(iSession);
    // set the IOVProxy
    loadTag(tag, snapshotTime);
    // Only make the first PayloadProxy object now because we don't know yet
    // how many we will need.
    m_proxies.push_back(std::make_shared<cond::persistency::PayloadProxy<DataT>>(
        &currentIov(), &session(), &requests(), m_source.empty() ? (const char*)nullptr : m_source.c_str()));
    m_edmProxies.push_back(std::make_shared<DataProxy>(m_proxies[0]));
    addInfo(il, cs, tag);
  }

  void initConcurrentIOVs(unsigned int nConcurrentIOVs) override {
    // Create additional PayloadProxy objects if we are allowing
    // multiple IOVs to run concurrently.
    if (m_proxies.size() != nConcurrentIOVs) {
      assert(m_proxies.size() == 1);
      for (unsigned int i = 1; i < nConcurrentIOVs; ++i) {
        m_proxies.push_back(std::make_shared<cond::persistency::PayloadProxy<DataT>>(
            &currentIov(), &session(), &requests(), m_source.empty() ? (const char*)nullptr : m_source.c_str()));
        m_edmProxies.push_back(std::make_shared<DataProxy>(m_proxies[i]));
        // This does nothing except in the special case of a KeyList PayloadProxy.
        // They all need to have copies of the same IOVProxy object.
        m_proxies[i]->initKeyList(*m_proxies[0]);
      }
      assert(m_proxies.size() == nConcurrentIOVs);
    }
    assert(m_proxies.size() == m_edmProxies.size());
  }

  edm::eventsetup::TypeTag type() const override { return m_type; }
  ProxyP proxy(unsigned int iovIndex) const override { return m_proxies.at(iovIndex); }
  edmProxyP edmProxy(unsigned int iovIndex) const override { return m_edmProxies.at(iovIndex); }

private:
  std::string m_source;
  edm::eventsetup::TypeTag m_type;
  std::vector<std::shared_ptr<cond::persistency::PayloadProxy<DataT>>> m_proxies;
  std::vector<edmProxyP> m_edmProxies;
};

#endif /* CONDCORE_PLUGINSYSTEM_DATAPROXY_H */
