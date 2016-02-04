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
   
   virtual void beginJob();
   virtual void endJob();
   virtual void beginRun(edm::Run const&, edm::EventSetup const&);
   virtual void endRun(edm::Run const&, edm::EventSetup const&);
   virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
   virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  
   static bool beginJobCalled;
   static bool endJobCalled;
   static bool beginRunCalled;
   static bool endRunCalled;
   static bool beginLumiCalled;
   static bool endLumiCalled;
   static bool destructorCalled;
private:
      // ----------member data ---------------------------
};


#endif /* test_TestBeginEndJobAnalyzer_h */
