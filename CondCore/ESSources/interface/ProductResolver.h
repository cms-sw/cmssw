#ifndef CondCore_ESSources_ProductResolver_H
#define CondCore_ESSources_ProductResolver_H

#include <cassert>
//#include <iostream>
#include <memory>
#include <string>
#include <mutex>

// user include files
#include "FWCore/Framework/interface/ESSourceProductResolverTemplate.h"
#include "FWCore/Framework/interface/DataKey.h"

#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Types.h"

// expose a cond::PayloadProxy as a eventsetup::ESProductResolver
namespace cond {
  template <typename DataT>
  struct DefaultInitializer {
    void operator()(DataT&) {}
  };

  template <class RecordT, class DataT, typename Initializer = cond::DefaultInitializer<DataT>>
  class ProductResolver : public edm::eventsetup::ESSourceProductResolverTemplate<DataT> {
  public:
    explicit ProductResolver(std::shared_ptr<cond::persistency::PayloadProxy<DataT>> pdata,
                             edm::SerialTaskQueue* iQueue,
                             std::mutex* iMutex)
        : edm::eventsetup::ESSourceProductResolverTemplate<DataT>(iQueue, iMutex), m_data{pdata} {}
    //ProductResolver(); // stop default
    const ProductResolver& operator=(const ProductResolver&) = delete;  // stop default

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
  protected:
    void prefetch(edm::eventsetup::DataKey const& iKey, edm::EventSetupRecordDetails) final {
      m_data->make();
      m_initializer(const_cast<DataT&>((*m_data)()));
    }

    DataT const* fetch() const final { return &(*m_data)(); }

  private:
    void initializeForNewIOV() override { m_data->initializeForNewIOV(); }

    // ---------- member data --------------------------------

    std::shared_ptr<cond::persistency::PayloadProxy<DataT>> m_data;
    Initializer m_initializer;
  };

  /* ABI bridging between the cond world and eventsetup world
   * keep them separated!
   */
  class ProductResolverWrapperBase {
  public:
    typedef std::shared_ptr<cond::persistency::BasePayloadProxy> ProxyP;
    typedef std::shared_ptr<edm::eventsetup::ESProductResolver> esResolverP;

    // limitation of plugin manager...
    typedef std::pair<std::string, std::string> Args;

    virtual edm::eventsetup::TypeTag type() const = 0;
    virtual ProxyP proxy(unsigned int iovIndex) const = 0;
    virtual esResolverP esResolver(unsigned int iovIndex) const = 0;

    ProductResolverWrapperBase();
    // late initialize (to allow to load ALL library first)
    virtual void lateInit(persistency::Session& session,
                          const std::string& tag,
                          const boost::posix_time::ptime& snapshotTime,
                          std::string const& il,
                          std::string const& cs,
                          edm::SerialTaskQueue* queue,
                          std::mutex* mutex) = 0;

    virtual void initConcurrentIOVs(unsigned int nConcurrentIOVs) = 0;

    void addInfo(std::string const& il, std::string const& cs, std::string const& tag);

    virtual ~ProductResolverWrapperBase();

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
    TimeType timeType() const { return m_iovProxy.tagInfo().timeType; }

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
class ProductResolverWrapper : public cond::ProductResolverWrapperBase {
public:
  typedef ::cond::ProductResolver<RecordT, DataT, Initializer> ProductResolver;

  // constructor from plugin...
  explicit ProductResolverWrapper(const char* source = nullptr) : m_source(source ? source : "") {
    //NOTE: We do this so that the type 'DataT' will get registered
    // when the plugin is dynamically loaded
    m_type = edm::eventsetup::DataKey::makeTypeTag<DataT>();
  }

  // late initialize (to allow to load ALL library first)
  void lateInit(cond::persistency::Session& iSession,
                const std::string& tag,
                const boost::posix_time::ptime& snapshotTime,
                std::string const& il,
                std::string const& cs,
                edm::SerialTaskQueue* queue,
                std::mutex* mutex) override {
    setSession(iSession);
    // set the IOVProxy
    loadTag(tag, snapshotTime);
    // Only make the first PayloadProxy object now because we don't know yet
    // how many we will need.
    m_proxies.push_back(std::make_shared<cond::persistency::PayloadProxy<DataT>>(
        &currentIov(), &session(), &requests(), m_source.empty() ? (const char*)nullptr : m_source.c_str()));
    m_esResolvers.push_back(std::make_shared<ProductResolver>(m_proxies[0], queue, mutex));
    addInfo(il, cs, tag);
  }

  void initConcurrentIOVs(unsigned int nConcurrentIOVs) override {
    // Create additional PayloadProxy objects if we are allowing
    // multiple IOVs to run concurrently.
    if (m_proxies.size() != nConcurrentIOVs) {
      assert(m_proxies.size() == 1);
      auto queue = m_esResolvers.front()->queue();
      auto mutex = m_esResolvers.front()->mutex();
      for (unsigned int i = 1; i < nConcurrentIOVs; ++i) {
        m_proxies.push_back(std::make_shared<cond::persistency::PayloadProxy<DataT>>(
            &currentIov(), &session(), &requests(), m_source.empty() ? (const char*)nullptr : m_source.c_str()));
        m_esResolvers.push_back(std::make_shared<ProductResolver>(m_proxies[i], queue, mutex));
        // This does nothing except in the special case of a KeyList PayloadProxy.
        // They all need to have copies of the same IOVProxy object.
        m_proxies[i]->initKeyList(*m_proxies[0]);
      }
      assert(m_proxies.size() == nConcurrentIOVs);
    }
    assert(m_proxies.size() == m_esResolvers.size());
  }

  edm::eventsetup::TypeTag type() const override { return m_type; }
  ProxyP proxy(unsigned int iovIndex) const override { return m_proxies.at(iovIndex); }
  esResolverP esResolver(unsigned int iovIndex) const override { return m_esResolvers.at(iovIndex); }

private:
  std::string m_source;
  edm::eventsetup::TypeTag m_type;
  std::vector<std::shared_ptr<cond::persistency::PayloadProxy<DataT>>> m_proxies;
  std::vector<std::shared_ptr<ProductResolver>> m_esResolvers;
};

#endif /* CondCore_ESSources_ProductResolver_H */
