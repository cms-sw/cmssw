#include "L1Trigger/L1GctAnalyzer/interface/compareBitCounts.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

compareBitCounts::compareBitCounts(const edm::Handle<L1GctHFBitCountsCollection> &data,
                                   const edm::Handle<L1GctHFBitCountsCollection> &emu,
                                   const GctErrorAnalyzerMBxInfo &mbxparams)
    : data_(data), emu_(emu), mbxparams_(mbxparams) {}

compareBitCounts::~compareBitCounts() {
  //anything need to be destructed?
}

bool compareBitCounts::doCompare(TH1I *errorFlag_hist_) {
  bool errorFlag = false;

  for (unsigned int i = 0; i < data_->size(); i++) {
    //check that we are looking at the triggered Bx in the data
    if (data_->at(i).bx() != mbxparams_.GCTTrigBx)
      continue;

    for (unsigned int j = 0; j < emu_->size(); j++) {
      //now check that we are looking at the corresponding triggered Bx in the emulator
      if (emu_->at(j).bx() != mbxparams_.EmuTrigBx)
        continue;

      for (unsigned int k = 0; k < NUM_GCT_RINGS; k++) {
        //now that we are on the right Bxs for data and emulator, check all the ring bitcounts match
        if (data_->at(i).bitCount(k) == emu_->at(j).bitCount(k)) {
          errorFlag_hist_->Fill(0);  //i.e. the two match
        } else {
          errorFlag_hist_->Fill(1);
          errorFlag = true;
        }
      }
    }
  }

  return errorFlag;
}
