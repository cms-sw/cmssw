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

    PopCon(const edm::ParameterSet& pset);

    virtual ~PopCon();

    template <typename Source>
    void write(Source const& source);

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

  template <typename Source>
  void PopCon::write(Source const& source) {
    typedef typename Source::value_type value_type;
    typedef typename Source::Container Container;

    std::pair<Container const*, std::string const> ret = source(initialize(), m_tagInfo, m_logDBEntry);
    Container const& iovs = *ret.first;

    if (m_LoggingOn) {
      std::string msg("Nothing to transfer;");
      size_t niovs = iovs.size();
      if (niovs) {
        std::ostringstream s;
        if (niovs == 1) {
          s << "Since " << (*iovs.begin()).first << "; ";
        } else {
          s << "first payload Since " << (*iovs.begin()).first << ", "
            << "last payload Since " << (*iovs.rbegin()).first << "; ";
        }
        msg = s.str();
      }
      std::ostringstream s;
      s << "PopCon v" << s_version << "; " << msg << ret.second;
      m_dbService->setLogHeaderForRecord(m_record, source.id(), s.str());
    }
    for (const auto& it : iovs)
      edm::LogInfo("PopCon") << "Since " << it.first << std::endl;

    m_dbService->writeMany(iovs, m_record);

    finalize(iovs.empty() ? Time_t(0) : iovs.rbegin()->first);
  }

}  // namespace popcon

#endif  //  POPCON_POPCON_H
