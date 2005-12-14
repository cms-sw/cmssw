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
// $Id: TestFailuresAnalyzer.h,v 1.1 2005/09/02 19:26:08 chrjones Exp $
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
   
   virtual void beginJob(const edm::EventSetup&);
   virtual void endJob();
   
private:
      // ----------member data ---------------------------
      int whichFailure_;
};


#endif /* test_TestFailuresAnalyzer_h */
