#ifndef compareRingSums_h
#define compareRingSums_h

#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "DataFormats/Common/interface/Handle.h"

#include "TH2.h"
#include "TH1.h"

#include "L1Trigger/L1GctAnalyzer/interface/GctErrorAnalyzerDefinitions.h"

class compareRingSums {
public:
  compareRingSums(const edm::Handle<L1GctHFRingEtSumsCollection> &data,
                  const edm::Handle<L1GctHFRingEtSumsCollection> &emu,
                  const GctErrorAnalyzerMBxInfo &mbxparams);
  ~compareRingSums();

  bool doCompare(TH1I *errorFlag_hist_);

private:
  edm::Handle<L1GctHFRingEtSumsCollection> data_, emu_;
  GctErrorAnalyzerMBxInfo mbxparams_;
};

#endif
