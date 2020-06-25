#ifndef POPCON_POPCON_H
#define POPCON_POPCON_H
//
// Author: Vincenzo Innocente
// Original Author:  Marcin BOGUSZ
//

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

#include "CondCore/CondDB/interface/Time.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include <iostream>

namespace popcon {

  /* Populator of the Condition DB
   *
   */
  class PopCon {
  public:
    typedef cond::Time_t Time_t;
    //typedef cond::Summary Summary;

    PopCon(const edm::ParameterSet& pset);

    virtual ~PopCon();

    template <typename Source>
    void write(Source const& source);

    template <typename T>
    void writeOne(T* payload, Time_t time);

  private:
    cond::persistency::Session initialize();
    void finalize(Time_t lastTill);

  private:
    edm::Service<cond::service::PoolDBOutputService> m_dbService;

    cond::persistency::Session m_targetSession;

    std::string m_targetConnectionString;

    std::string m_authPath;

    int m_authSys;

    std::string m_record;

    std::string m_payload_name;

    bool m_LoggingOn;

    std::string m_tag;

    cond::TagInfo_t m_tagInfo;

    cond::LogDBEntry_t m_logDBEntry;

    bool m_close;

    Time_t m_lastTill;

    static constexpr const char* const s_version = "5.0";
  };

  template <typename T>
  void PopCon::writeOne(T* payload, Time_t time) {
    m_dbService->writeOne(payload, time, m_record);
  }

  template <typename Container>
  void displayHelper(Container const& payloads) {
    typename Container::const_iterator it;
    for (it = payloads.begin(); it != payloads.end(); it++)
      edm::LogInfo("PopCon") << "Since " << (*it).time << std::endl;
  }

  template <typename Container>
  const std::string displayIovHelper(Container const& payloads) {
    if (payloads.empty())
      return std::string("Nothing to transfer;");
    std::ostringstream s;
    // when only 1 payload is transferred;
    if (payloads.size() == 1)
      s << "Since " << (*payloads.begin()).time << "; ";
    else {
      // when more than one payload are transferred;
      s << "first payload Since " << (*payloads.begin()).time << ", "
        << "last payload Since " << (*payloads.rbegin()).time << "; ";
    }
    return s.str();
  }

  template <typename Source>
  void PopCon::write(Source const& source) {
    typedef typename Source::value_type value_type;
    typedef typename Source::Container Container;

    std::pair<Container const*, std::string const> ret = source(initialize(), m_tagInfo, m_logDBEntry);
    Container const& payloads = *ret.first;

    if (m_LoggingOn) {
      std::ostringstream s;
      s << "PopCon v" << s_version << "; " << displayIovHelper(payloads) << ret.second;
      //s << "PopCon v" << s_version << "; " << cond::userInfo() << displayIovHelper(payloads) << ret.second;
      m_dbService->setLogHeaderForRecord(m_record, source.id(), s.str());
    }
    displayHelper(payloads);

    std::for_each(payloads.begin(),
                  payloads.end(),
                  std::bind(&popcon::PopCon::writeOne<value_type>,
                            this,
                            std::bind(&Container::value_type::payload, std::placeholders::_1),
                            std::bind(&Container::value_type::time, std::placeholders::_1)));

    finalize(payloads.empty() ? Time_t(0) : payloads.back().time);
  }

}  // namespace popcon

#endif  //  POPCON_POPCON_H
