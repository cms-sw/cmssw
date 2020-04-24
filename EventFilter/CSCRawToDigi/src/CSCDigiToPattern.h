/** \file
 *
 *  \author A. Tumanov - Rice
 */

#include <iostream>
#include <FWCore/Framework/interface/ConsumesCollector.h>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

class CSCDigiToPattern : public edm::EDAnalyzer {
public:
  explicit CSCDigiToPattern(edm::ParameterSet const& conf);
  void analyze(edm::Event const& e, edm::EventSetup const& iSetup) override;

  //virtual void endJob();
private:
  // variables persistent across events should be declared here.
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection> d_token;
  //
};



