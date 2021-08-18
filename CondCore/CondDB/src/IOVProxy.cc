#include <memory>

#include "CondCore/CondDB/interface/IOVProxy.h"
#include "SessionImpl.h"

namespace cond {

  namespace persistency {

    // comparison functor for iov tuples: Time_t only and Time_t,string
    struct IOVComp {
      bool operator()(const cond::Time_t& x, const cond::Time_t& y) { return (x < y); }

      bool operator()(const cond::Time_t& x, const std::tuple<cond::Time_t, cond::Hash>& y) {
        return (x < std::get<0>(y));
      }
    };

    // function to search in the vector the target time
    template <typename T>
    typename std::vector<T>::const_iterator search(const cond::Time_t& val, const std::vector<T>& container) {
      if (container.empty())
        return container.end();
      auto p = std::upper_bound(container.begin(), container.end(), val, IOVComp());
      return (p != container.begin()) ? p - 1 : container.end();
    }

    IOVArray::Iterator::Iterator() : m_current(), m_parent(nullptr) {}

    IOVArray::Iterator::Iterator(IOVContainer::const_iterator current, const IOVArray* parent)
        : m_current(current), m_parent(parent) {}

    IOVArray::Iterator::Iterator(const Iterator& rhs) : m_current(rhs.m_current), m_parent(rhs.m_parent) {}

    IOVArray::Iterator& IOVArray::Iterator::operator=(const Iterator& rhs) {
      if (this != &rhs) {
        m_current = rhs.m_current;
        m_parent = rhs.m_parent;
      }
      return *this;
    }

    cond::Iov_t IOVArray::Iterator::operator*() {
      cond::Iov_t retVal;
      retVal.since = std::get<0>(*m_current);
      auto next = m_current;
      next++;
      if (next == m_parent->m_array->end()) {
        retVal.till = cond::time::MAX_VAL;
      } else {
        retVal.till = cond::time::tillTimeFromNextSince(std::get<0>(*next), m_parent->m_tagInfo.timeType);
      }
      // default is the end of validity when set...
      if (retVal.till > m_parent->m_tagInfo.endOfValidity) {
        retVal.till = m_parent->m_tagInfo.endOfValidity;
      }
      retVal.payloadId = std::get<1>(*m_current);

      return retVal;
    }

    IOVArray::Iterator& IOVArray::Iterator::operator++() {
      m_current++;
      return *this;
    }

    IOVArray::Iterator IOVArray::Iterator::operator++(int) {
      Iterator tmp(*this);
      operator++();
      return tmp;
    }

    bool IOVArray::Iterator::operator==(const Iterator& rhs) const {
      if (m_current != rhs.m_current)
        return false;
      if (m_parent != rhs.m_parent)
        return false;
      return true;
    }

    bool IOVArray::Iterator::operator!=(const Iterator& rhs) const { return !operator==(rhs); }

    IOVArray::IOVArray() : m_array(new IOVContainer) {}

    IOVArray::IOVArray(const IOVArray& rhs) : m_array(), m_tagInfo(rhs.m_tagInfo) {
      m_array = std::make_unique<IOVContainer>(*rhs.m_array);
    }

    IOVArray& IOVArray::operator=(const IOVArray& rhs) {
      m_array = std::make_unique<IOVContainer>(*rhs.m_array);
      m_tagInfo = rhs.m_tagInfo;
      return *this;
    }

    const cond::Tag_t& IOVArray::tagInfo() const { return m_tagInfo; }

    IOVArray::Iterator IOVArray::begin() const { return Iterator(m_array->begin(), this); }

    IOVArray::Iterator IOVArray::end() const { return Iterator(m_array->end(), this); }

    IOVArray::Iterator IOVArray::find(cond::Time_t time) const { return Iterator(search(time, *m_array), this); }

    size_t IOVArray::size() const { return m_array->size(); }

    // returns true if at least one IOV is in the sequence.
    bool IOVArray::isEmpty() const { return m_array->empty(); }

    // implementation details...
    // only hosting data in this case
    class IOVProxyData {
    public:
      IOVProxyData() : iovSequence() {}

      // tag data
      cond::Tag_t tagInfo;
      // iov data
      boost::posix_time::ptime snapshotTime;
      cond::Time_t groupLowerIov = cond::time::MAX_VAL;
      cond::Time_t groupHigherIov = cond::time::MIN_VAL;
      bool cacheInitialized = false;
      std::vector<cond::Time_t> sinceGroups;
      IOVContainer iovSequence;
      // monitoring data
      size_t numberOfQueries = 0;
    };

    IOVProxy::IOVProxy() : m_data(), m_session() {}

    IOVProxy::IOVProxy(const std::shared_ptr<SessionImpl>& session) : m_data(new IOVProxyData), m_session(session) {}

    IOVProxy::IOVProxy(const IOVProxy& rhs) : m_data(rhs.m_data), m_session(rhs.m_session) {}

    IOVProxy& IOVProxy::operator=(const IOVProxy& rhs) {
      m_data = rhs.m_data;
      m_session = rhs.m_session;
      return *this;
    }

    void IOVProxy::load(const std::string& tagName) {
      boost::posix_time::ptime notime;
      load(tagName, notime);
    }

    void IOVProxy::load(const std::string& tagName, const boost::posix_time::ptime& snapshotTime) {
      if (!m_data.get())
        return;

      // clear
      reset();

      checkTransaction("IOVProxyNew::load");

      int dummy;
      if (!m_session->iovSchema().tagTable().select(tagName,
                                                    m_data->tagInfo.timeType,
                                                    m_data->tagInfo.payloadType,
                                                    m_data->tagInfo.synchronizationType,
                                                    m_data->tagInfo.endOfValidity,
                                                    m_data->tagInfo.lastValidatedTime,
                                                    dummy)) {
        throwException("Tag \"" + tagName + "\" has not been found in the database.", "IOVProxy::load");
      }
      m_data->tagInfo.name = tagName;
      m_data->snapshotTime = snapshotTime;
    }

    void IOVProxy::loadGroups() {
      //if( !m_data.get() ) return;

      // clear
      resetIOVCache();

      //checkTransaction( "IOVProxyNew::load" );
      m_session->iovSchema().iovTable().getGroups(m_data->tagInfo.name,
                                                  m_data->snapshotTime,
                                                  cond::time::sinceGroupSize(m_data->tagInfo.timeType),
                                                  m_data->sinceGroups);
      m_data->cacheInitialized = true;
    }

    IOVArray IOVProxy::selectAll() {
      boost::posix_time::ptime no_time;
      return selectAll(no_time);
    }

    IOVArray IOVProxy::selectAll(const boost::posix_time::ptime& snapshottime) {
      if (!m_data.get())
        throwException("No tag has been loaded.", "IOVProxy::selectAll");
      checkTransaction("IOVProxy::selectAll");
      IOVArray ret;
      ret.m_tagInfo = m_data->tagInfo;
      m_session->iovSchema().iovTable().select(
          m_data->tagInfo.name, cond::time::MIN_VAL, cond::time::MAX_VAL, snapshottime, *ret.m_array);
      return ret;
    }

    IOVArray IOVProxy::selectRange(const cond::Time_t& begin, const cond::Time_t& end) {
      boost::posix_time::ptime no_time;
      return selectRange(begin, end, no_time);
    }

    IOVArray IOVProxy::selectRange(const cond::Time_t& begin,
                                   const cond::Time_t& end,
                                   const boost::posix_time::ptime& snapshotTime) {
      if (!m_data.get())
        throwException("No tag has been loaded.", "IOVProxy::selectRange");

      checkTransaction("IOVProxy::selectRange");

      IOVArray ret;
      ret.m_tagInfo = m_data->tagInfo;
      m_session->iovSchema().iovTable().getRange(m_data->tagInfo.name, begin, end, snapshotTime, *ret.m_array);
      return ret;
    }

    bool IOVProxy::selectRange(const cond::Time_t& begin, const cond::Time_t& end, IOVContainer& destination) {
      if (!m_data.get())
        throwException("No tag has been loaded.", "IOVProxy::selectRange");

      checkTransaction("IOVProxy::selectRange");

      boost::posix_time::ptime no_time;
      size_t prevSize = destination.size();
      m_session->iovSchema().iovTable().getRange(m_data->tagInfo.name, begin, end, no_time, destination);
      size_t niov = destination.size() - prevSize;
      return niov > 0;
    }

    //void IOVProxy::reload(){
    //  if(m_data.get() && !m_data->tagInfo.empty()) {
    //	if(m_data->range) loadRange( m_data->tag,  m_data->groupLowerIov, m_data->groupHigherIov, m_data->snapshotTime );
    //	else load( m_data->tag, m_data->snapshotTime, m_data->full );
    //  }
    //}

    void IOVProxy::resetIOVCache() {
      if (m_data.get()) {
        m_data->groupLowerIov = cond::time::MAX_VAL;
        m_data->groupHigherIov = cond::time::MIN_VAL;
        m_data->sinceGroups.clear();
        m_data->iovSequence.clear();
        m_data->numberOfQueries = 0;
      }
    }

    void IOVProxy::reset() {
      if (m_data.get()) {
        m_data->tagInfo.clear();
      }
      resetIOVCache();
    }

    cond::Tag_t IOVProxy::tagInfo() const { return m_data.get() ? m_data->tagInfo : cond::Tag_t(); }

    void setTillToLastIov(cond::Iov_t& target, cond::Time_t endOfValidity) {
      if (endOfValidity < cond::time::MAX_VAL) {
        if (target.since >= endOfValidity) {
          target.clear();
        } else {
          target.till = endOfValidity;
        }
      } else {
        target.till = cond::time::MAX_VAL;
      }
    }

    cond::TagInfo_t IOVProxy::iovSequenceInfo() const {
      if (!m_data.get())
        throwException("No tag has been loaded.", "IOVProxy::iovSequenceInfo");
      checkTransaction("IOVProxy::iovSequenceInfo");
      cond::TagInfo_t ret;
      m_session->iovSchema().iovTable().getSize(m_data->tagInfo.name, m_data->snapshotTime, ret.size);
      cond::Iov_t last;
      bool ok = m_session->iovSchema().iovTable().getLastIov(
          m_data->tagInfo.name, m_data->snapshotTime, ret.lastInterval.since, ret.lastInterval.payloadId);
      if (ok) {
        setTillToLastIov(ret.lastInterval, m_data->tagInfo.endOfValidity);
      }
      return ret;
    }

    std::tuple<std::string, boost::posix_time::ptime, boost::posix_time::ptime> IOVProxy::getMetadata() const {
      if (!m_data.get())
        throwException("No tag has been loaded.", "IOVProxy::getMetadata");
      checkTransaction("IOVProxy::getMetadata");
      std::tuple<std::string, boost::posix_time::ptime, boost::posix_time::ptime> ret;
      if (!m_session->iovSchema().tagTable().getMetadata(
              m_data->tagInfo.name, std::get<0>(ret), std::get<1>(ret), std::get<2>(ret))) {
        throwException("Metadata for tag \"" + m_data->tagInfo.name + "\" have not been found in the database.",
                       "IOVProxy::getMetadata");
      }
      return ret;
    }

    void IOVProxy::checkTransaction(const std::string& ctx) const {
      if (!m_session.get())
        throwException("The session is not active.", ctx);
      if (!m_session->isTransactionActive(false))
        throwException("The transaction is not active.", ctx);
    }

    void IOVProxy::fetchSequence(cond::Time_t lowerGroup, cond::Time_t higherGroup) {
      m_data->iovSequence.clear();
      m_session->iovSchema().iovTable().select(
          m_data->tagInfo.name, lowerGroup, higherGroup, m_data->snapshotTime, m_data->iovSequence);

      if (m_data->iovSequence.empty()) {
        m_data->groupLowerIov = cond::time::MAX_VAL;
        m_data->groupHigherIov = cond::time::MIN_VAL;
      } else {
        if (lowerGroup > cond::time::MIN_VAL) {
          m_data->groupLowerIov = std::get<0>(m_data->iovSequence.front());
        } else {
          m_data->groupLowerIov = cond::time::MIN_VAL;
        }
        m_data->groupHigherIov = std::get<0>(m_data->iovSequence.back());
        if (higherGroup < cond::time::MAX_VAL) {
          m_data->groupHigherIov = cond::time::tillTimeFromNextSince(higherGroup, m_data->tagInfo.timeType);
        } else {
          m_data->groupHigherIov = cond::time::MAX_VAL;
        }
      }

      m_data->numberOfQueries++;
    }

    cond::Iov_t IOVProxy::getInterval(cond::Time_t time) { return getInterval(time, cond::time::MAX_VAL); }

    cond::Iov_t IOVProxy::getInterval(cond::Time_t time, cond::Time_t defaultIovSize) {
      if (!m_data.get())
        throwException("No tag has been loaded.", "IOVProxy::getInterval");
      checkTransaction("IOVProxy::getInterval");
      if (!m_data->cacheInitialized)
        loadGroups();
      cond::Iov_t retVal;
      // organize iovs in pages...
      // first check the available iov cache:
      if (m_data->groupLowerIov == cond::time::MAX_VAL ||  // case 0 : empty cache ( the first request )
          time < m_data->groupLowerIov || time >= m_data->groupHigherIov) {  // case 1 : target outside

        // a new query required!
        // first determine the groups
        auto iGLow = search(time, m_data->sinceGroups);
        if (iGLow == m_data->sinceGroups.end()) {
          // no suitable group=no iov at all! exiting...
          return retVal;
        }
        auto iGHigh = iGLow;
        cond::Time_t lowG = *iGLow;
        iGHigh++;
        cond::Time_t highG = cond::time::MAX_VAL;
        if (iGHigh != m_data->sinceGroups.end())
          highG = *iGHigh;

        // finally, get the iovs for the selected group interval!!
        fetchSequence(lowG, highG);
      }

      // the current iov set is a good one...
      auto iIov = search(time, m_data->iovSequence);
      if (iIov == m_data->iovSequence.end()) {
        return retVal;
      }

      retVal.since = std::get<0>(*iIov);
      auto next = iIov;
      next++;

      // default is the end of validity when set...
      retVal.till = m_data->tagInfo.endOfValidity;
      // for the till, the next element of the sequence has to be looked up
      cond::Time_t tillVal;
      if (next != m_data->iovSequence.end()) {
        tillVal = cond::time::tillTimeFromNextSince(std::get<0>(*next), m_data->tagInfo.timeType);
      } else {
        tillVal = m_data->groupHigherIov;
      }
      if (tillVal < retVal.till)
        retVal.till = tillVal;
      //
      retVal.payloadId = std::get<1>(*iIov);
      if (retVal.till == cond::time::MAX_VAL && defaultIovSize != cond::time::MAX_VAL) {
        if (defaultIovSize == 0) {
          // ???? why?
          retVal.clear();
        } else {
          retVal.since = time;
          retVal.till = retVal.since + defaultIovSize - 1;
          if (time > retVal.till)
            retVal.till = time;
        }
      }
      return retVal;
    }

    cond::Iov_t IOVProxy::getLast() {
      checkTransaction("IOVProxy::getLast");
      cond::Iov_t ret;
      bool ok = m_session->iovSchema().iovTable().getLastIov(
          m_data->tagInfo.name, m_data->snapshotTime, ret.since, ret.payloadId);
      if (ok) {
        setTillToLastIov(ret, m_data->tagInfo.endOfValidity);
      }
      return ret;
    }

    int IOVProxy::loadedSize() const { return m_data.get() ? m_data->iovSequence.size() : 0; }

    int IOVProxy::sequenceSize() const {
      checkTransaction("IOVProxy::sequenceSize");
      size_t ret = 0;
      m_session->iovSchema().iovTable().getSize(m_data->tagInfo.name, m_data->snapshotTime, ret);

      return ret;
    }

    size_t IOVProxy::numberOfQueries() const { return m_data.get() ? m_data->numberOfQueries : 0; }

    std::pair<cond::Time_t, cond::Time_t> IOVProxy::loadedGroup() const {
      return m_data.get() ? std::make_pair(m_data->groupLowerIov, m_data->groupHigherIov)
                          : std::make_pair(cond::time::MAX_VAL, cond::time::MIN_VAL);
    }

    const std::shared_ptr<SessionImpl>& IOVProxy::session() const { return m_session; }

  }  // namespace persistency

}  // namespace cond
