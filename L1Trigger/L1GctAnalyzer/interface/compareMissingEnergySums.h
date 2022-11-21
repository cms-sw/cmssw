#ifndef compareMissingEnergySums_h
#define compareMissingEnergySums_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/L1GctAnalyzer/interface/GctErrorAnalyzerDefinitions.h"

#include "TH2.h"
#include "TH1.h"

template <class T>
class compareMissingEnergySums {
public:
  compareMissingEnergySums(const T &data, const T &emu, const GctErrorAnalyzerMBxInfo &mbxparams);
  ~compareMissingEnergySums();

  bool doCompare(TH1I *errorFlag_hist_);

private:
  T data_, emu_;
  GctErrorAnalyzerMBxInfo mbxparams_;
};

template <class T>
compareMissingEnergySums<T>::compareMissingEnergySums(const T &data,
                                                      const T &emu,
                                                      const GctErrorAnalyzerMBxInfo &mbxparams)
    : data_(data), emu_(emu), mbxparams_(mbxparams) {}

template <class T>
compareMissingEnergySums<T>::~compareMissingEnergySums() {}

template <class T>
bool compareMissingEnergySums<T>::doCompare(TH1I *errorFlag_hist_) {
  bool errorFlag = false;

  for (unsigned int i = 0; i < data_->size(); i++) {
    if (data_->at(i).bx() != mbxparams_.GCTTrigBx)
      continue;

    for (unsigned int j = 0; j < emu_->size(); j++) {
      if (emu_->at(j).bx() != mbxparams_.EmuTrigBx)
        continue;

      if (data_->at(i).overFlow() && emu_->at(j).overFlow()) {
        //if both overflow bits are set then = match
        errorFlag_hist_->Fill(0);
        return errorFlag;
      }

      //check that we consider non-zero candidates - if we don't, return (don't fill hist)
      if (!data_->at(i).overFlow() && !emu_->at(j).overFlow() && data_->at(i).et() == 0 && emu_->at(j).et() == 0)
        return errorFlag;

      if (!data_->at(i).overFlow() && !emu_->at(j).overFlow() && data_->at(i).et() == emu_->at(j).et() &&
          data_->at(i).phi() == emu_->at(j).phi()) {
        //similarly, if the overflow bits are both off but the mag/phi agree = match
        errorFlag_hist_->Fill(0);
        return errorFlag;
      }

      if (!data_->at(i).overFlow() && !emu_->at(j).overFlow() && data_->at(i).et() == emu_->at(j).et() &&
          data_->at(i).phi() != emu_->at(j).phi()) {
        //if the overflow bits are both off but only the mag agree = mag match
        errorFlag_hist_->Fill(1);
        return errorFlag = true;
      }

      if (!data_->at(i).overFlow() && !emu_->at(j).overFlow() && data_->at(i).et() != emu_->at(j).et() &&
          data_->at(i).phi() == emu_->at(j).phi()) {
        //if the overflow bits are both off but only the phi agree = phi match
        errorFlag_hist_->Fill(2);
        return errorFlag = true;
      }

      //otherwise it's a total unmatch
      errorFlag_hist_->Fill(3);
      errorFlag = true;
      return errorFlag;
    }
  }
  return errorFlag;
}

#endif
