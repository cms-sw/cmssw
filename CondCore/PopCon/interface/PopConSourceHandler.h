#ifndef PopConSourceHandler_H
#define PopConSourceHandler_H

#include "CondCore/CondDB/interface/Session.h"
#include "CondCore/CondDB/interface/Time.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cond {
  class Summary;
}

#include "CondFormats/Common/interface/GenericSummary.h"

namespace popcon {

  /** Online DB source handler, aims at returning the vector of data to be 
   * transferred to the online database
   * Subdetector developers inherit over this class with template parameter of 
   * payload class; 
   * need just to implement the getNewObjects method that loads the calibs,
   * the sourceId methods that return a text identifier of the source,
   * and provide a constructor that accept a ParameterSet
   */
  template <class T>
  class PopConSourceHandler {
  public:
    typedef T value_type;
    typedef PopConSourceHandler<T> self;
    typedef cond::Time_t Time_t;

    typedef std::map<Time_t, std::shared_ptr<T> > Container;
    typedef std::unique_ptr<T> Ref;

    PopConSourceHandler() : m_tagInfo(nullptr), m_logDBEntry(nullptr) {}

    virtual ~PopConSourceHandler() {}

    cond::TagInfo_t const& tagInfo() const { return *m_tagInfo; }

    // return last paylod of the tag
    Ref lastPayload() const { return m_session.fetchPayload<T>(tagInfo().lastInterval.payloadId); }

    // return last successful log entry for the tag in question
    cond::LogDBEntry_t const& logDBEntry() const { return *m_logDBEntry; }

    // FIX ME
    void initialize(const cond::persistency::Session& dbSession,
                    cond::TagInfo_t const& tagInfo,
                    cond::LogDBEntry_t const& logDBEntry) {
      m_session = dbSession;
      m_tagInfo = &tagInfo;
      m_logDBEntry = &logDBEntry;
    }

    // this is the only mandatory interface
    std::pair<Container const*, std::string const> operator()(const cond::persistency::Session& session,
                                                              cond::TagInfo_t const& tagInfo,
                                                              cond::LogDBEntry_t const& logDBEntry) const {
      const_cast<self*>(this)->initialize(session, tagInfo, logDBEntry);
      return std::pair<Container const*, std::string const>(&(const_cast<self*>(this)->returnData()), userTextLog());
    }

    Container const& returnData() {
      getNewObjects();
      for (auto item : m_to_transfer) {
        std::shared_ptr<T> payload(item.first);
        m_iovs.insert(std::make_pair(item.second, payload));
      }
      return m_iovs;
    }

    std::string const& userTextLog() const { return m_userTextLog; }

    //Implement to fill m_to_transfer vector and  m_userTextLog
    //use getOfflineInfo to get the contents of offline DB
    virtual void getNewObjects() = 0;

    // return a string identifing the source
    virtual std::string id() const = 0;

  protected:
    cond::persistency::Session& dbSession() const { return m_session; }

  private:
    mutable cond::persistency::Session m_session;

    cond::TagInfo_t const* m_tagInfo;

    cond::LogDBEntry_t const* m_logDBEntry;

  protected:
    //vector of payload objects and iovinfo to be transferred
    //class looses ownership of payload object
    std::vector<std::pair<T*, Time_t> > m_to_transfer;

    Container m_iovs;

    std::string m_userTextLog;
  };
}  // namespace popcon
#endif
