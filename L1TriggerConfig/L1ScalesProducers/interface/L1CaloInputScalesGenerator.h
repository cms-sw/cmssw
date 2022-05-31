// -*- C++ -*-
//
// Package:    L1ScalesProducers
// Class:      L1CaloInputScalesGenerator
//
/**\class L1CaloInputScalesGenerator L1CaloInputScalesGenerator.cc L1TriggerConfig/L1ScalesProducers/src/L1CaloInputScalesGenerator.cc

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
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGRecord.h"
#include "CalibFormats/CaloTPG/interface/CaloTPGTranscoder.h"

//
// class declaration
//

class L1CaloInputScalesGenerator : public edm::one::EDAnalyzer<> {
public:
  explicit L1CaloInputScalesGenerator(const edm::ParameterSet&);
  ~L1CaloInputScalesGenerator() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::ESGetToken<CaloTPGTranscoder, CaloTPGRecord> transcoderToken_;
  EcalTPGScale::Tokens tokens_;
};
