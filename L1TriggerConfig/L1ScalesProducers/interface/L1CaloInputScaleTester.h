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

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class L1CaloInputScaleTester : public edm::EDAnalyzer {
public:
  explicit L1CaloInputScaleTester(const edm::ParameterSet&);
  ~L1CaloInputScaleTester() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
};
