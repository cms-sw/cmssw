
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

#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

namespace edm {
  namespace service {
    class JobReportService : public JobReport {
    public:
      JobReportService(ParameterSet const& ps, ActivityRegistry& reg);
      ~JobReportService();

      void postEndJob();

      void frameworkShutdownOnFailure();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    };

    JobReportService::~JobReportService() {}

    JobReportService::JobReportService(ParameterSet const&, ActivityRegistry& reg) : JobReport() {
      reg.watchPostEndJob(this, &JobReportService::postEndJob);

      // We don't handle PreProcessEvent, because we have to know *which
      // input file* was the event read from. Only the InputSource that
      // did the reading knows this.
    }

    void JobReportService::postEndJob() {
      // This will be called at end-of-job (obviously).
      // Dump information to the MessageLogger's JobSummary.

      // ... not yet implemented ...

      //
      // Any files that are still open should be flushed to the report
      //
      impl()->flushFiles();
    }

    void JobReportService::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.setComment("Enables job reports.");
      descriptions.addDefault(desc);
    }
  }  // namespace service
}  // namespace edm

#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
using edm::service::JobReportService;

typedef edm::serviceregistry::AllArgsMaker<edm::JobReport, JobReportService> JobReportMaker;
DEFINE_FWK_SERVICE_MAKER(JobReportService, JobReportMaker);
