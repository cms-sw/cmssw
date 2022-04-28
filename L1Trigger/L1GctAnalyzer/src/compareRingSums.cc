#include "L1Trigger/L1GctAnalyzer/interface/compareRingSums.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

compareRingSums::compareRingSums(const edm::Handle<L1GctHFRingEtSumsCollection> &data,
                                 const edm::Handle<L1GctHFRingEtSumsCollection> &emu,
                                 const GctErrorAnalyzerMBxInfo &mbxparams)
    : data_(data), emu_(emu), mbxparams_(mbxparams) {}

compareRingSums::~compareRingSums() {}

bool compareRingSums::doCompare(TH1I *errorFlag_hist_) {
  bool errorFlag = false;

  for (unsigned int i = 0; i < data_->size(); i++) {
    //check that the GCT trig bx is being considered
    if (data_->at(i).bx() != mbxparams_.GCTTrigBx)
      continue;

    for (unsigned int j = 0; j < emu_->size(); j++) {
      //now check that the Emu trig bx is being considered
      if (emu_->at(j).bx() != mbxparams_.EmuTrigBx)
        continue;

      //now loop over each ring and make sure the energy sums match
      for (unsigned int k = 0; k < NUM_GCT_RINGS; k++) {
        if (data_->at(i).etSum(k) == emu_->at(j).etSum(k)) {
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
