#ifndef FWCore_Services_JobReportService_h
#define FWCore_Services_JobReportService_h
// -*- C++ -*-
//
// Package:     Services
// Class  :     JobReport
// 
/**\class JobReportService JobReportService.h FWCore/Services/src/JobReportService.h

Description: A service that collections job handling information.

Usage:
The JobReport service collects 'job handling' information (currently
file handling) from several sources, collates the information, and
at appropriate intervales, reports the information to the job report,
through the MessageLogger.

*/

//
// Original Author:  Marc Paterno
//

#include <string>

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  class ConfigurationDescriptions;

  namespace service {
    class JobReportService : public JobReport {
    public:
      JobReportService(ParameterSet const& ps, ActivityRegistry& reg);
      ~JobReportService();

      void postEndJob();

      void frameworkShutdownOnFailure();

      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
    };

    inline
    bool isProcessWideService(JobReportService const*) {
      return true;
    }
  }
}
#endif
