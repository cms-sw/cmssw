#include "DataFormats/METReco/interface/PUSubMETData.h"

namespace reco {

  PUSubMETCandInfo::PUSubMETCandInfo() {
  
    p4_.SetXYZT(0.,0.,0.,0.);
    dZ_ = 0.;
    type_ = PUSubMETCandInfo::kUndefined;
    charge_ = 0;
    isWithinJet_ = false;
    passesLooseJetId_ = 0.;
    offsetEnCorr_ = 0.;
    mva_ = 0.;
    chargedEnFrac_ = 0.; 
  
  }
  
  
  PUSubMETCandInfo::~PUSubMETCandInfo() {

  }


  bool PUSubMETCandInfo::operator<(const reco::PUSubMETCandInfo& jet2) const {
    return p4_.pt() < jet2.p4().pt();
  }


}

