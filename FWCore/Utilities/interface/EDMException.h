#ifndef Utilities_EDMException_h
#define Utilities_EDMException_h

/**

 This is the basic exception that is thrown by the framework code.
 It exists primarily to distinguish framework thrown exception types
 from developer thrown exception types. As such there is very little
 interface other than constructors specific to this derived type.

 This is the initial version of the framework/edm error
 and action codes.  Should the error/action lists be completely
 dynamic?  Should they have this fixed part and allow for dynamic
 expansion?  The answer is not clear right now and this will suffice
 for the first version.

 Will ErrorCodes be used as return codes?  Unknown at this time.

**/

#include "FWCore/Utilities/interface/CodedException.h"

#include <string>

namespace edm {
  namespace errors {

    // If you add a new entry to the set of values, make sure to
    // update the translation map in EDMException.cc, and also the
    // actions table in FWCore/Framework/src/Actions.cc

    enum ErrorCodes {
       Unknown=0,
       ProductNotFound,
       InsertFailure,
       Configuration,
       LogicError,
       UnimplementedFeature,
       InvalidReference,
       NullPointerError,
       NoProductSpecified,
       EventTimeout,
       EventCorruption,

       ModuleFailure,
       ScheduleExecutionFailure,
       EventProcessorFailure,

       FileInPathError,
       FatalRootError,

       ProductDoesNotSupportViews,

       NotFound
    };

  }

  typedef edm::CodedException<edm::errors::ErrorCodes> Exception;
}

#endif
