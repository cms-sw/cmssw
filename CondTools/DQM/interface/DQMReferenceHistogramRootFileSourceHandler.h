#ifndef CondTools_DQM_DQMReferenceHistogramRootFileSourceHandler_h
#define CondTools_DQM_DQMReferenceHistogramRootFileSourceHandler_h

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/GeometryObjects/interface/GeometryFile.h"
#include <string>

namespace popcon {
  class DQMReferenceHistogramRootFileSourceHandler : public popcon::PopConSourceHandler<GeometryFile> {
   public:
    DQMReferenceHistogramRootFileSourceHandler(const edm::ParameterSet & pset);
    ~DQMReferenceHistogramRootFileSourceHandler();
    void getNewObjects();
    std::string id() const;
   private:
    std::string m_name;
    std::string m_file;
    bool m_zip;
    cond::Time_t m_since;
    bool m_debugMode;
  };
}

#endif
