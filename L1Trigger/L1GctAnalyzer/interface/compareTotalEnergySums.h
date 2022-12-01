#ifndef compareTotalEnergySums_h
#define compareTotalEnergySums_h

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
class compareTotalEnergySums {
public:
  compareTotalEnergySums(const T &data, const T &emu, const GctErrorAnalyzerMBxInfo &mbxparams);
  ~compareTotalEnergySums();

  bool doCompare(TH1I *errorFlag_hist_);

private:
  T data_, emu_;
  GctErrorAnalyzerMBxInfo mbxparams_;
};

template <class T>
compareTotalEnergySums<T>::compareTotalEnergySums(const T &data, const T &emu, const GctErrorAnalyzerMBxInfo &mbxparams)
    : data_(data), emu_(emu), mbxparams_(mbxparams) {
  //std::cout << "initialising..." << std::endl;
}

template <class T>
compareTotalEnergySums<T>::~compareTotalEnergySums() {
  //anything need to be destructed?
}

template <class T>
bool compareTotalEnergySums<T>::doCompare(TH1I *errorFlag_hist_) {
  bool errorFlag = false;

  for (unsigned int i = 0; i < data_->size(); i++) {
    //check the GCTTrigBx is the one being considered
    if (data_->at(i).bx() != mbxparams_.GCTTrigBx)
      continue;

    for (unsigned int j = 0; j < emu_->size(); j++) {
      //check the EmuTrigBx is the one being considered
      if (emu_->at(j).bx() != mbxparams_.EmuTrigBx)
        continue;

      //now check if both overflow bits (from the trigbx) are set
      if (data_->at(i).overFlow() && emu_->at(j).overFlow()) {
        //if the overflow bits in data and emulator are set, that's enough to call a match
        errorFlag_hist_->Fill(0);
        return errorFlag;
      }

      //check if both over flow bits are not set and if both values are zero, and if so return the errorFlag
      //without making any modifications (a zero et match doesn't mean so much).
      if (!data_->at(i).overFlow() && !emu_->at(j).overFlow() && data_->at(i).et() == 0 && emu_->at(j).et() == 0)
        return errorFlag;

      //now check if the values correspond, again with both overflow bits not set
      if (!data_->at(i).overFlow() && !emu_->at(j).overFlow() && data_->at(i).et() == emu_->at(j).et()) {
        //if they are both explicitly not set, and the resulting energies are identical, that's a match
        errorFlag_hist_->Fill(0);
        return errorFlag;
      }

      //otherwise, it's a fail
      errorFlag_hist_->Fill(1);
      errorFlag = true;
      return errorFlag;
    }
  }
  return errorFlag;
}

#endif
