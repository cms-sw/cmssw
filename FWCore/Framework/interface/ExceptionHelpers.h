#ifndef FWCore_Framework_ExceptionHelpers_h
#define FWCore_Framework_ExceptionHelpers_h

// Original Author:  W. David Dagenhart
//         Created:  March 2012

// These methods should only be used by the Framework
// internally!

// These methods are intended to be used at points
// where we catch exceptions earlier than the end
// of the "main" function in cmsRun.cpp. The purpose of
// of catching them early is to print the useful exception
// information before we try cleaning up lumis, runs,
// and files by calling functions like endRun. We have
// experienced problems that seg faults during the
// cleanup caused the primary exception message to be lost.

// The intent is that printing will be disabled in the
// case where there were additional exceptions after
// a primary exception and these additional exceptions
// where thrown while cleaning up runs, lumis, and files.
// The messages from the exceptions after the first tend
// to confuse people more than help.

// At the moment, these functions are called after an
// exception occurs in a module's beginRun, beginLumi,
// event, endLumi, or endRun methods. And also if
// the exception occurs in most of the InputSource virtual
// methods.  I expect that usage might be extended to other
// cases. Note these functions are not needed outside of
// the time where the cleanup of runs, lumis and files might
// occur. If the process is before or after that period, it
// is simpler to let the exceptions should just be allowed
// to unwind up the stack into main.

#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <exception>
#include <functional>
#include <string>

namespace edm {

  void addContextAndPrintException(char const* context,
                                   cms::Exception& ex,
                                   bool disablePrint);

  template <typename TReturn>
  TReturn callWithTryCatchAndPrint(std::function<TReturn (void)> iFunc,
                                   char const* context = 0,
                                   bool disablePrint = false) {

    try {
      try {
        return iFunc();
      }
      catch (cms::Exception& e) { throw; }
      catch (std::bad_alloc& bda) { convertException::badAllocToEDM(); }
      catch (std::exception& e) { convertException::stdToEDM(e); }
      catch (std::string& s) { convertException::stringToEDM(s); }
      catch (char const* c) { convertException::charPtrToEDM(c); }
      catch (...) { convertException::unknownToEDM(); }
    }
    catch(cms::Exception& ex) {
      addContextAndPrintException(context, ex, disablePrint);
      throw;
    }
    return TReturn();
  }
}

#endif
