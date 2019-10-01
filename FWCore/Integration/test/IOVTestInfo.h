#ifndef FWCore_Integration_IOVTestInfo_h
#define FWCore_Integration_IOVTestInfo_h
// -*- C++ -*-
//
// Package:     FWCore/Integration
// Class  :     IOVTestInfo
//
/**\class edmtest::IOVTestInfo

 Description: Class used to test the EventSetup in the integration test

*/
//
// Original Author:  W. David Dagenhart
//         Created:  21 March 2019
//

namespace edmtest {
  struct IOVTestInfo {
    IOVTestInfo() {}
    unsigned int iovStartRun_ = 0;
    unsigned int iovStartLumi_ = 0;
    unsigned int iovEndRun_ = 0;
    unsigned int iovEndLumi_ = 0;
    unsigned int iovIndex_ = 0;
    unsigned long long cacheIdentifier_ = 0;
  };
}  // namespace edmtest
#endif
