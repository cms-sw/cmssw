#ifndef Framework_TestFailuresAnalyzer_h
#define Framework_TestFailuresAnalyzer_h
// -*- C++ -*-
//
// Package:     test
// Class  :     TestFailuresAnalyzer
// 
/**\class TestFailuresAnalyzer TestFailuresAnalyzer.h Framework/test/interface/TestFailuresAnalyzer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep  2 14:17:17 EDT 2005
//

// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
// user include files

// forward declarations
class TestFailuresAnalyzer : public edm::EDAnalyzer {
public:
   explicit TestFailuresAnalyzer(const edm::ParameterSet&);
   ~TestFailuresAnalyzer();
   
   
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   
   virtual void beginJob();
   virtual void endJob();
   
private:
      // ----------member data ---------------------------
      int whichFailure_;
      unsigned long long eventToThrow_;
};


#endif /* test_TestFailuresAnalyzer_h */
