#ifndef Framework_TestBeginEndJobAnalyzer_h
#define Framework_TestBeginEndJobAnalyzer_h
// -*- C++ -*-
//
// Package:     test
// Class  :     TestBeginEndJobAnalyzer
// 
/**\class TestBeginEndJobAnalyzer TestBeginEndJobAnalyzer.h Framework/test/interface/TestBeginEndJobAnalyzer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Sep  2 14:17:17 EDT 2005
// $Id$
//

// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
// user include files

// forward declarations
class TestBeginEndJobAnalyzer : public edm::EDAnalyzer {
public:
   explicit TestBeginEndJobAnalyzer(const edm::ParameterSet&);
   ~TestBeginEndJobAnalyzer();
   
   
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   
   virtual void beginJob(const edm::EventSetup&);
   virtual void endJob();
   
   static bool beginJobCalled;
   static bool endJobCalled;
private:
      // ----------member data ---------------------------
};


#endif /* test_TestBeginEndJobAnalyzer_h */
