#ifndef Utilities_JobMode_h
#define Utilities_JobMode_h

// -*- C++ -*-

/*
 An enum indicating the nature of the job, for use (at least initially)
 in deciding what the "hardwired" defaults for MessageLogger configuration
 ought to be.

*/

namespace edm {

  enum JobMode {
         GridJobMode
       , AnalysisJobMode
       , NilJobMode
  };

}
#endif // Utilities_JobMode_h
