// -*- C++ -*-
//
// Package:    L1ScalesProducer
// Class:      L1ScalesTester
//
/**\class L1ScalesTester L1ScalesTester.h L1TriggerConfig/L1ScalesProducer/interface/L1ScalesTester.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Tue Oct 3 15:28:00 CEST 2006
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

class L1ScalesTester : public edm::EDAnalyzer {
public:
  explicit L1ScalesTester(const edm::ParameterSet&);
  ~L1ScalesTester() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

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
