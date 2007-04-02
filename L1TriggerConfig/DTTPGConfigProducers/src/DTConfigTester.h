// -*- C++ -*-
//
// Package:    DTTPGConfigProducer
// Class:      DTConfigTester
// 
/**\class  DTConfigTester DTConfigTester.h L1Trigger/DTConfigProducer/interface/DTConfigTester.h

 Description: tester for DTConfig

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sara Vanini
//         Created:  Sat Mar 17 10:00 CEST 2007
// $Id: 
//
//


// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


//
// class decleration
//

class DTConfigTester : public edm::EDAnalyzer {
   public:
      explicit DTConfigTester(const edm::ParameterSet&);
      ~DTConfigTester();

      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
};

