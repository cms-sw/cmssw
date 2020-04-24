/** 
 * Demo analyzer for reading digis
 * author A.Tumanov 2/22/06 
 * Updated 10.09.2013 but untested - Tim Cox
 *   
 */

#include <iostream>
#include <FWCore/Framework/interface/ConsumesCollector.h>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDDUStatusDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCRPCDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCDCCFormatStatusDigiCollection.h"

class DigiAnalyzer : public edm::EDAnalyzer {
public:
  explicit DigiAnalyzer(edm::ParameterSet const& conf);
  void analyze(edm::Event const& e, edm::EventSetup const& iSetup) override;

private:

  int eventNumber;

  edm::EDGetTokenT<CSCWireDigiCollection>             wd_token;
  edm::EDGetTokenT<CSCStripDigiCollection>            sd_token;
  edm::EDGetTokenT<CSCComparatorDigiCollection>       cd_token;
  edm::EDGetTokenT<CSCALCTDigiCollection>             al_token;
  edm::EDGetTokenT<CSCCLCTDigiCollection>             cl_token;
  edm::EDGetTokenT<CSCRPCDigiCollection>              rd_token;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection>    co_token;
  edm::EDGetTokenT<CSCDDUStatusDigiCollection>        dd_token;
  edm::EDGetTokenT<CSCDCCFormatStatusDigiCollection>  dc_token;

};



