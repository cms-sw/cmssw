#ifndef EventFilter_CSCDigiToRawModule_h
#define EventFilter_CSCDigiToRawModule_h

/** \class CSCDigiToRawModule
 *
 *  \author A. Tumanov - Rice
 */

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

class CSCDigiToRaw;

class CSCDigiToRawModule : public edm::EDProducer {
 public:
  /// Constructor
  CSCDigiToRawModule(const edm::ParameterSet & pset);

  /// Destructor
  virtual ~CSCDigiToRawModule();

  // Operations
  virtual void produce( edm::Event&, const edm::EventSetup& );

 private:
  CSCDigiToRaw * packer;

  edm::EDGetTokenT<CSCWireDigiCollection>             wd_token;
  edm::EDGetTokenT<CSCStripDigiCollection>            sd_token;
  edm::EDGetTokenT<CSCComparatorDigiCollection>       cd_token;
  edm::EDGetTokenT<CSCALCTDigiCollection>             al_token;
  edm::EDGetTokenT<CSCCLCTDigiCollection>             cl_token;
  edm::EDGetTokenT<CSCCLCTPreTriggerCollection>       pr_token;
  edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection>    co_token;

};
#endif


