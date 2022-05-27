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
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1CaloEtScale.h"
#include "CondFormats/DataRecord/interface/L1EmEtScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1JetEtScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"

//
// class decleration
//

class L1ScalesTester : public edm::one::EDAnalyzer<> {
public:
  explicit L1ScalesTester(const edm::ParameterSet&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  const edm::ESGetToken<L1CaloEtScale, L1EmEtScaleRcd> emScaleToken_;
  const edm::ESGetToken<L1CaloEcalScale, L1CaloEcalScaleRcd> ecalScaleToken_;
  const edm::ESGetToken<L1CaloHcalScale, L1CaloHcalScaleRcd> hcalScaleToken_;
  const edm::ESGetToken<L1CaloEtScale, L1JetEtScaleRcd> jetScaleToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
