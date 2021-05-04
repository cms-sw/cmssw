#ifndef CondCore_CondDB_PayloadProxy_h
#define CondCore_CondDB_PayloadProxy_h

#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/Types.h"

#include <memory>
#include <string>
#include <vector>

namespace cond {

  namespace persistency {

    // still needed??
    /* get iov by name (key, tag, ...)
     */
    class CondGetter {
    public:
      virtual ~CondGetter() {}
      virtual IOVProxy get(std::string name) const = 0;
    };

    // implements the not templated part...
    class BasePayloadProxy {
    public:
      //
      BasePayloadProxy(Iov_t const* mostRecentCurrentIov,
                       Session const* mostRecentSession,
                       std::shared_ptr<std::vector<Iov_t>> const* mostRecentRequests);

      virtual ~BasePayloadProxy();

      virtual void make() = 0;

      bool isValid() const;

      virtual void loadMore(CondGetter const&) {}

      void initializeForNewIOV();

    private:
      virtual void loadPayload() = 0;

    protected:
      Iov_t m_iovAtInitialization;
      Session m_session;
      std::shared_ptr<std::vector<Iov_t>> m_requests;

      Iov_t const* m_mostRecentCurrentIov;
      Session const* m_mostRecentSession;
      std::shared_ptr<std::vector<Iov_t>> const* m_mostRecentRequests;
    };

    /* proxy to the payload valid at a given time...

    */
    template <typename DataT>
    class PayloadProxy : public BasePayloadProxy {
    public:
      explicit PayloadProxy(Iov_t const* mostRecentCurrentIov,
                            Session const* mostRecentSession,
                            std::shared_ptr<std::vector<Iov_t>> const* mostRecentRequests,
                            const char* source = nullptr)
          : BasePayloadProxy(mostRecentCurrentIov, mostRecentSession, mostRecentRequests) {}

      ~PayloadProxy() override {}

      void initKeyList(PayloadProxy const&) {}

      // dereference
      const DataT& operator()() const {
        if (!m_data) {
          throwException("The Payload has not been loaded.", "PayloadProxy::operator()");
        }
        return (*m_data);
      }

      void make() override {
        if (isValid()) {
          if (m_iovAtInitialization.payloadId == m_currentPayloadId)
            return;
          m_session.transaction().start(true);
          loadPayload();
          m_session.transaction().commit();
        }
      }

    protected:
      void loadPayload() override {
        if (m_iovAtInitialization.payloadId.empty()) {
          throwException("Can't load payload: no valid IOV found.", "PayloadProxy::loadPayload");
        }
        m_data = m_session.fetchPayload<DataT>(m_iovAtInitialization.payloadId);
        m_currentPayloadId = m_iovAtInitialization.payloadId;
        m_requests->push_back(m_iovAtInitialization);
      }

    private:
      std::unique_ptr<DataT> m_data;
      Hash m_currentPayloadId;
    };

  }  // namespace persistency
}  // namespace cond
#endif  // CondCore_CondDB_PayloadProxy_h
