#ifndef CondCore_CondDB_KeyListProxy_h
#define CondCore_CondDB_KeyListProxy_h

#include "CondCore/CondDB/interface/PayloadProxy.h"
#include "CondCore/CondDB/interface/KeyList.h"
#include <memory>
#include <vector>
#include <string>

namespace cond {

  struct Iov_t;

  namespace persistency {

    class Session;

    template <>
    class PayloadProxy<cond::persistency::KeyList> : public PayloadProxy<std::vector<cond::Time_t>> {
    public:
      typedef std::vector<cond::Time_t> DataT;
      typedef PayloadProxy<DataT> super;

      explicit PayloadProxy(Iov_t const* mostRecentCurrentIov,
                            Session const* mostRecentSession,
                            std::shared_ptr<std::vector<Iov_t>> const* mostRecentRequests,
                            const char* source = nullptr)
          : super(mostRecentCurrentIov, mostRecentSession, mostRecentRequests, source), m_keyList() {
        if (source)
          m_name = source;
      }

      ~PayloadProxy() override {}

      void initKeyList(PayloadProxy const& originalPayloadProxy) { m_keyList.init(originalPayloadProxy.m_keyList); }

      // dereference (does not load)
      const KeyList& operator()() const { return m_keyList; }

      void loadMore(CondGetter const& getter) override { m_keyList.init(getter.get(m_name)); }

    protected:
      void loadPayload() override {
        super::loadPayload();
        m_keyList.setKeys(super::operator()());
      }

    private:
      std::string m_name;
      KeyList m_keyList;
    };
  }  // namespace persistency

}  // namespace cond
#endif
