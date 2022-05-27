// -*- C++ -*-
//
// Package:    L1CaloInputScaleTester
// Class:      L1CaloInputScaleTester
//
/**\class L1CaloInputScaleTester L1CaloInputScaleTester.cc L1TriggerConfig/L1ScalesProducers/src/L1CaloInputScaleTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/140
//         Created:  Wed Jun 25 16:40:01 CEST 2008
//
//

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondFormats/DataRecord/interface/L1CaloEcalScaleRcd.h"
#include "CondFormats/DataRecord/interface/L1CaloHcalScaleRcd.h"
#include "CondFormats/L1TObjects/interface/L1CaloEcalScale.h"
#include "CondFormats/L1TObjects/interface/L1CaloHcalScale.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"

//
// class declaration
//

class L1CaloInputScaleTester : public edm::one::EDAnalyzer<> {
public:
  explicit L1CaloInputScaleTester(const edm::ParameterSet&);
  ~L1CaloInputScaleTester() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::ESGetToken<L1CaloHcalScale, L1CaloHcalScaleRcd> hcalScaleToken_;
  edm::ESGetToken<L1CaloEcalScale, L1CaloEcalScaleRcd> ecalScaleToken_;
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> transcoderToken_;
  EcalTPGScale::Tokens tokens_;
};
