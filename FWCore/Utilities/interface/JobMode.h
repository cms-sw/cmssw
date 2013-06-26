#ifndef FWCore_Utilities_JobMode_h
#define FWCore_Utilities_JobMode_h

// -*- C++ -*-

/*
 An enum indicating the nature of the job, for use (at least initially)
 in deciding what the "hardwired" defaults for MessageLogger configuration
 ought to be.

*/

namespace edm {

  enum JobMode {
         GridJobMode
       , ReleaseValidationJobMode
       , AnalysisJobMode
       , NilJobMode
  };

}
#endif // FWCore_Utilities_JobMode_h
