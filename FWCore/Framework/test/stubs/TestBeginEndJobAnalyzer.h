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
#include "FWCore/Utilities/interface/GCC11Compatibility.h"
// user include files

// forward declarations
class TestBeginEndJobAnalyzer : public edm::EDAnalyzer {
public:
   explicit TestBeginEndJobAnalyzer(const edm::ParameterSet&);
   ~TestBeginEndJobAnalyzer();
   
   
   virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
   
   virtual void beginJob() override;
   virtual void endJob() override;
   virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
   virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
   virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
   virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

   

   struct Control {
      bool beginJobCalled = false;
      bool endJobCalled = false;
      bool beginRunCalled = false;
      bool endRunCalled = false;
      bool beginLumiCalled = false;
      bool endLumiCalled = false;
      bool destructorCalled = false;
   };

   static Control & control();


private:
      // ----------member data ---------------------------
};


#endif /* test_TestBeginEndJobAnalyzer_h */
