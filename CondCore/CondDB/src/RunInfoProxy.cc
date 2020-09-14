#include "CondCore/CondDB/interface/RunInfoProxy.h"
#include "SessionImpl.h"

namespace cond {

  namespace persistency {

    // implementation details...
    // only hosting data in this case
    class RunInfoProxyData {
    public:
      RunInfoProxyData() : runList() {}

      // the data loaded
      RunInfoProxy::RunInfoData runList;
    };

    RunInfoProxy::Iterator::Iterator() : m_current() {}

    RunInfoProxy::Iterator::Iterator(RunInfoData::const_iterator current) : m_current(current) {}

    RunInfoProxy::Iterator::Iterator(const Iterator& rhs) : m_current(rhs.m_current) {}

    RunInfoProxy::Iterator& RunInfoProxy::Iterator::operator=(const Iterator& rhs) {
      if (this != &rhs) {
        m_current = rhs.m_current;
      }
      return *this;
    }

    cond::RunInfo_t RunInfoProxy::Iterator::operator*() { return cond::RunInfo_t(*m_current); }

    RunInfoProxy::Iterator& RunInfoProxy::Iterator::operator++() {
      m_current++;
      return *this;
    }

    RunInfoProxy::Iterator RunInfoProxy::Iterator::operator++(int) {
      Iterator tmp(*this);
      operator++();
      return tmp;
    }

    bool RunInfoProxy::Iterator::operator==(const Iterator& rhs) const {
      if (m_current != rhs.m_current)
        return false;
      return true;
    }

    bool RunInfoProxy::Iterator::operator!=(const Iterator& rhs) const { return !operator==(rhs); }

    RunInfoProxy::RunInfoProxy() : m_data(), m_session() {}

    RunInfoProxy::RunInfoProxy(const std::shared_ptr<SessionImpl>& session)
        : m_data(new RunInfoProxyData), m_session(session) {}

    RunInfoProxy::RunInfoProxy(const RunInfoProxy& rhs) : m_data(rhs.m_data), m_session(rhs.m_session) {}

    RunInfoProxy& RunInfoProxy::operator=(const RunInfoProxy& rhs) {
      m_data = rhs.m_data;
      m_session = rhs.m_session;
      return *this;
    }

    //
    void RunInfoProxy::load(Time_t low, Time_t up) {
      if (!m_data.get())
        return;

      // clear
      reset();

      checkTransaction("RunInfoProxy::load(Time_t,Time_t)");

      std::string dummy;
      if (!m_session->runInfoSchema().runInfoTable().getInclusiveRunRange(low, up, m_data->runList)) {
        throwException("No runs have been found in the range (" + std::to_string(low) + "," + std::to_string(up) + ")",
                       "RunInfoProxy::load(Time_t,Time_t)");
      }
    }

    void RunInfoProxy::load(const boost::posix_time::ptime& low, const boost::posix_time::ptime& up) {
      if (!m_data.get())
        return;

      // clear
      reset();

      checkTransaction("RunInfoProxy::load(const boost::posix_time::ptime&,const boost::posix_time::ptime&)");

      std::string dummy;
      if (!m_session->runInfoSchema().runInfoTable().getInclusiveTimeRange(low, up, m_data->runList)) {
        throwException("No runs have been found in the interval (" + boost::posix_time::to_simple_string(low) + "," +
                           boost::posix_time::to_simple_string(up) + ")",
                       "RunInfoProxy::load(boost::posix_time::ptime&,const boost::posix_time::ptime&)");
      }
    }

    void RunInfoProxy::reset() {
      if (m_data.get()) {
        m_data->runList.clear();
      }
    }

    void RunInfoProxy::checkTransaction(const std::string& ctx) {
      if (!m_session.get())
        throwException("The session is not active.", ctx);
      if (!m_session->isTransactionActive(false))
        throwException("The transaction is not active.", ctx);
    }

    RunInfoProxy::Iterator RunInfoProxy::begin() const {
      if (m_data.get()) {
        return Iterator(m_data->runList.begin());
      }
      return Iterator();
    }

    RunInfoProxy::Iterator RunInfoProxy::end() const {
      if (m_data.get()) {
        return Iterator(m_data->runList.end());
      }
      return Iterator();
    }

    // comparison functor for iov tuples: Time_t
    struct IOVRunComp {
      bool operator()(const std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime>& x,
                      const cond::Time_t& y) {
        return (y > std::get<0>(x));
      }
    };

    // comparison functor for iov tuples: boost::posix_time::ptime
    struct IOVTimeComp {
      bool operator()(const std::tuple<cond::Time_t, boost::posix_time::ptime, boost::posix_time::ptime>& x,
                      const boost::posix_time::ptime& y) {
        return (y > std::get<2>(x));
      }
    };

    RunInfoProxy::Iterator RunInfoProxy::find(Time_t target) const {
      if (m_data.get()) {
        auto p = std::lower_bound(m_data->runList.begin(), m_data->runList.end(), target, IOVRunComp());
        return Iterator(p);
      }
      return Iterator();
    }

    RunInfoProxy::Iterator RunInfoProxy::find(const boost::posix_time::ptime& target) const {
      if (m_data.get()) {
        auto p = std::lower_bound(m_data->runList.begin(), m_data->runList.end(), target, IOVTimeComp());
        return Iterator(p);
      }
      return Iterator();
    }

    //
    cond::RunInfo_t RunInfoProxy::get(Time_t target) const {
      Iterator it = find(target);
      if (it == Iterator())
        throwException("No data has been found.", "RunInfoProxy::get(Time_t)");
      if (it == end())
        throwException("The target run has not been found in the selected run range.", "RunInfoProxy::get(Time_t)");
      return *it;
    }

    //
    cond::RunInfo_t RunInfoProxy::get(const boost::posix_time::ptime& target) const {
      Iterator it = find(target);
      if (it == Iterator())
        throwException("No data has been found.", "RunInfoProxy::get(const boost::posix_time::ptime&)");
      if (it == end())
        throwException("The target time has not been found in the selected time range.",
                       "RunInfoProxy::get(const boost::posix_time::ptime&)");
      return *it;
    }

    int RunInfoProxy::size() const { return m_data.get() ? m_data->runList.size() : 0; }

  }  // namespace persistency
}  // namespace cond
