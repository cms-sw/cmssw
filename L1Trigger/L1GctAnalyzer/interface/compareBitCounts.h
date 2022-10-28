#ifndef compareBitCounts_h
#define compareBitCounts_h

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"

#include "TH2.h"
#include "TH1.h"

#include "L1Trigger/L1GctAnalyzer/interface/GctErrorAnalyzerDefinitions.h"

class compareBitCounts {
public:
  compareBitCounts(const edm::Handle<L1GctHFBitCountsCollection> &data,
                   const edm::Handle<L1GctHFBitCountsCollection> &emu,
                   const GctErrorAnalyzerMBxInfo &mbxparams);
  ~compareBitCounts();

  bool doCompare(TH1I *errorFlag_hist_);

private:
  edm::Handle<L1GctHFBitCountsCollection> data_, emu_;
  GctErrorAnalyzerMBxInfo mbxparams_;
};

#endif
