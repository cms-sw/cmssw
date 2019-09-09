#include "CondCore/CondDB/interface/PayloadProxy.h"

namespace cond {

  namespace persistency {

    BasePayloadProxy::BasePayloadProxy(Iov_t const* mostRecentCurrentIov,
                                       Session const* mostRecentSession,
                                       std::shared_ptr<std::vector<Iov_t>> const* mostRecentRequests)
        : m_mostRecentCurrentIov(mostRecentCurrentIov),
          m_mostRecentSession(mostRecentSession),
          m_mostRecentRequests(mostRecentRequests) {}

    BasePayloadProxy::~BasePayloadProxy() {}

    bool BasePayloadProxy::isValid() const { return m_iovAtInitialization.isValid(); }

    void BasePayloadProxy::initializeForNewIOV() {
      m_iovAtInitialization = *m_mostRecentCurrentIov;
      m_session = *m_mostRecentSession;
      m_requests = *m_mostRecentRequests;
    }

  }  // namespace persistency
}  // namespace cond
