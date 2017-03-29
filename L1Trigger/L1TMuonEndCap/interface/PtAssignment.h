// calculte pT for a given InternalTrack

#ifndef L1TRIGGER_L1TMUONENDCAP_PTASSIGNMENT_H
#define L1TRIGGER_L1TMUONENDCAP_PTASSIGNMENT_H

#include "FWCore/Framework/interface/EventSetup.h"
#include "L1Trigger/L1TMuonEndCap/interface/MuonInternalTrack.h"
#include "L1Trigger/L1TMuonEndCap/interface/Forest.h"
#include "TGraph.h"


namespace l1t {

  class EmtfPtAssignment {
  public: 
    EmtfPtAssignment(const char * tree_dir="L1Trigger/L1TMuon/data/emtf_luts/v_16_02_21/ModeVariables/trees");    
    
    unsigned long calculateAddress(L1TMuon::InternalTrack track, const edm::EventSetup& es, int mode);
    float calculatePt(unsigned long Address);    


  private:
    // DT Forest:  a proper CondFormat replacement already exists, but is not yet integrated here yet:
    std::vector<int> allowedModes_;
    Forest forest_[16];

  };


}

#endif
