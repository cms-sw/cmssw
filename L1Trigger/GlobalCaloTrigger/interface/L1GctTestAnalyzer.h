// -*- C++ -*-
//
// Package:    L1GctTestAnalyzer
// Class:      L1GctTestAnalyzer
// 
/**\class L1GctTestAnalyzer L1GctTestAnalyzer.cc L1Trigger/GlobalCaloTrigger/test/L1GctTestAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu May 18 16:45:23 CEST 2006
// $Id$
//
//


// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class L1GctTestAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L1GctTestAnalyzer(const edm::ParameterSet&);
      ~L1GctTestAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
