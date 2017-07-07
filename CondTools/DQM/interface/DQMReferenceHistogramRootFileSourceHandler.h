#ifndef CondTools_DQM_DQMReferenceHistogramRootFileSourceHandler_h
#define CondTools_DQM_DQMReferenceHistogramRootFileSourceHandler_h

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
//#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/FileBlob.h"
#include <string>

namespace popcon {
  class DQMReferenceHistogramRootFileSourceHandler : public popcon::PopConSourceHandler<FileBlob> {
   public:
    DQMReferenceHistogramRootFileSourceHandler(const edm::ParameterSet & pset);
    ~DQMReferenceHistogramRootFileSourceHandler() override;
    void getNewObjects() override;
    std::string id() const override;
   private:
    std::string m_name;
    std::string m_file;
    bool m_zip;
    //cond::Time_t m_since;
    unsigned long long m_since;
    bool m_debugMode;
  };
}

#endif
