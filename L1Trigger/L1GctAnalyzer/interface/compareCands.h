#ifndef compareCands_h
#define compareCands_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "TH2.h"
#include "TH1.h"
#include "L1Trigger/L1GctAnalyzer/interface/GctErrorAnalyzerDefinitions.h"

//first declare stuff about the class template
//notice that we pass our own defined struct of data with necessary mbx information (we can modify this in future without changing much)
//we also avoid messing about with class hierarchy...
template <class T>
class compareCands {
public:
  compareCands(const T &data, const T &emu, const GctErrorAnalyzerMBxInfo &mbxparams);
  ~compareCands();

  bool doCompare(TH1I *errorFlag_hist_,
                 TH1I *mismatchD_Rank,
                 TH2I *mismatchD_EtEtaPhi,
                 TH1I *mismatchE_Rank,
                 TH2I *mismatchE_EtEtaPhi);

private:
  T data_, emu_;
  GctErrorAnalyzerMBxInfo mbxparams_;
};

//now the implementation
template <class T>
compareCands<T>::compareCands(const T &data, const T &emu, const GctErrorAnalyzerMBxInfo &mbxparams)
    : data_(data), emu_(emu), mbxparams_(mbxparams) {
  //std::cout << "initialising..." << std::endl;
}

template <class T>
compareCands<T>::~compareCands() {
  //anything need destructing?
}

template <class T>
bool compareCands<T>::doCompare(TH1I *errorFlag_hist_,
                                TH1I *mismatchD_Rank,
                                TH2I *mismatchD_EtEtaPhi,
                                TH1I *mismatchE_Rank,
                                TH2I *mismatchE_EtEtaPhi) {
  //this code has now been patched to be multiple_bx compliant. However, this still means that only 1 comparison will happen per event, and this has to be
  //matched such that the RCTTrigBx(=0) data is run over the emulator and the EmuTrigBx analysis (Bx=0) corresponds to the GCTTrigBx (Bx=0) analysis
  //These TrigBx parameters are set in the configuration to make things more flexible if things change later

  //define some temporary local variables
  bool errorFlag = false;
  unsigned int i = 0, j = 0;
  std::vector<bool> matched(GCT_OBJECT_QUANTA);
  //this makes a vector of GCT_OBJECT_QUANTA=4 bools, all set to false
  //remember that pushing back will make the vector larger!

  for (i = 0; i < data_->size(); i++) {
    //The first thing to check is that the BX of the data corresponds to the trig Bx (we expect these to be contiguous i.e. data sorted in Bx)
    if (data_->at(i).bx() != mbxparams_.GCTTrigBx)
      continue;

    //If the data candidate has zero rank, move to the next data candidate
    //since all the candidates are ranked in order of rank, this implies all the remaining data candidates also have rank = 0
    if (data_->at(i).rank() == 0)
      continue;

    for (j = 0; j < emu_->size(); j++) {
      //Again, the first thing to check in this loop is that the BX of the emulator data corresponds to the trig Bx
      if (emu_->at(j).bx() != mbxparams_.EmuTrigBx)
        continue;

      if (data_->at(i).rank() == emu_->at(j).rank() &&
          data_->at(i).regionId().ieta() == emu_->at(j).regionId().ieta() &&
          data_->at(i).regionId().iphi() == emu_->at(j).regionId().iphi() && matched.at((j % GCT_OBJECT_QUANTA)) == 0) {
        //this means that the ith data candidate matches the jth emulator candidate
        errorFlag_hist_->Fill(0);                    //fill the errorflag histo in the matched bin
        matched.at((j % GCT_OBJECT_QUANTA)) = true;  //set emulator candidate to matched so it doesn't get re-used
        break;                                       //matched the current data candidate, now move to the next
      }

      if ((j % GCT_OBJECT_QUANTA) + 1 == GCT_OBJECT_QUANTA) {
        errorFlag_hist_->Fill(1);                   //fill the errorflag histo in the unmatched data candidate bin
        mismatchD_Rank->Fill(data_->at(i).rank());  //fill the rank histogram of mismatched data candidates
        mismatchD_EtEtaPhi->Fill(data_->at(i).regionId().ieta(),
                                 data_->at(i).regionId().iphi(),
                                 data_->at(i).rank());  //fill the EtEtaPhi dist for mismatched candidates
        errorFlag = true;                               //set the errorFlag to true
      }
    }
  }

  //loop over the matched boolean vector and see if there are any rank>0 unmatched emu candidates - if there are populate the histogram in the emulator mismatched bin
  for (i = 0; i < matched.size(); i++) {
    //the first thing to check is that the matched flag for object i out of 0,1,2,3 (0 -> GCT_OBJECT_QUANTA-1) is not set - then we can check that the corresponding
    //emulator candidates either have rank = 0 (which is good) or rank > 0 (which is bad)
    if (matched.at(i))
      continue;

    //now loop over the emulator candidates
    for (j = 0; j < emu_->size(); j++) {
      //check that the bx of the emulator candidates is the trigbx
      if (emu_->at(j).bx() != mbxparams_.EmuTrigBx)
        continue;

      //now check that the j%GCT_OBJECT_QUANTA is the same as the index of the false entry in the bool_matched vector so that we are looking at the right candidate
      if ((j % GCT_OBJECT_QUANTA == i) && (emu_->at(j).rank() > 0)) {
        errorFlag_hist_->Fill(2);                  //increment emulator mismatched bin
        mismatchE_Rank->Fill(emu_->at(j).rank());  //fill the rank histogram for unmatched emulator
        mismatchE_EtEtaPhi->Fill(emu_->at(j).regionId().ieta(),
                                 emu_->at(j).regionId().iphi(),
                                 emu_->at(j).rank());  //fill EtEtaPhi for unmatched emu cands
        errorFlag = true;                              //set the errorFlag (if it's not already)
      }
    }
  }

  return errorFlag;
}

#endif
