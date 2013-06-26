#ifndef CondTools_DQM_DQMSummarySourceHandler_h
#define CondTools_DQM_DQMSummarySourceHandler_h

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
//#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/DQMObjects/interface/DQMSummary.h"
#include <string>

namespace popcon {
  class DQMSummarySourceHandler : public popcon::PopConSourceHandler<DQMSummary> {
   public:
    DQMSummarySourceHandler(const edm::ParameterSet & pset);
    ~DQMSummarySourceHandler();
    void getNewObjects();
    std::string id() const;
   private:
    std::string m_name;
    //cond::Time_t m_since;
    unsigned long long m_since;
    // for reading from omds 
    std::string m_connectionString;
    std::string m_user;
    std::string m_pass;
  };
}

#endif
