
// -*- C++ -*-
//
// Package:     Services
// Class  :     JobReport
// 
//
// Original Author:  Marc Paterno
// $Id: JobReportService.cc,v 1.2 2010/03/09 16:24:55 wdd Exp $
//


#include "FWCore/Services/src/JobReportService.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {

  namespace service {
    JobReportService::~JobReportService() {}

    JobReportService::JobReportService(ParameterSet const& /*ps*/,
			 ActivityRegistry& reg) :
      JobReport() {

      reg.watchPostEndJob(this, &JobReportService::postEndJob);
      reg.watchJobFailure(this, &JobReportService::frameworkShutdownOnFailure);

      // We don't handle PreProcessEvent, because we have to know *which
      // input file* was the event read from. Only the InputSource that
      // did the reading knows this.
    }

    void 
    JobReportService::postEndJob() {
      // This will be called at end-of-job (obviously).
      // Dump information to the MessageLogger's JobSummary.

      // ... not yet implemented ...      

      // Maybe we should have a member function called from both
      // postEndJob() and frameworkShutdownOnFailure(), so that common
      // elements are reported through common code.

        //
       // Any files that are still open should be flushed to the report
      //
      impl()->flushFiles();

    }

    void
    JobReportService::frameworkShutdownOnFailure() {
      // Dump information to the MessageLogger's JobSummary
      // about the files that aren't already closed,
      // and whatever summary information is wanted.

      // Maybe we should have a member function called from both
      // postEndJob() and frameworkShutdownOnFailure(), so that common
      // elements are reported through common code.
      impl()->flushFiles();
    }

    void
    JobReportService::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
      edm::ParameterSetDescription desc;
      desc.setComment("Enables job reports.");
      descriptions.addDefault(desc);
    }
  } // namespace service
} //namspace edm
