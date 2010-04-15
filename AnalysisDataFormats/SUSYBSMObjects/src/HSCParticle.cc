#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

namespace susybsm {

float HSCParticle::p() const {
  if(      hasMuonRef() && muonRef()->combinedMuon()  .isNonnull()){ return muonRef()->combinedMuon()  ->p();
  }else if(hasMuonRef() && muonRef()->innerTrack()    .isNonnull()){ return muonRef()->innerTrack()    ->p();
  }else if(hasMuonRef() && muonRef()->standAloneMuon().isNonnull()){ return muonRef()->standAloneMuon()->p();
  }else if(hasTrackRef()&& trackRef()                 .isNonnull()){ return trackRef()                 ->p();
  }else return 0.0f;
}

float HSCParticle::pt() const {
  if(      hasMuonRef() && muonRef()->combinedMuon()  .isNonnull()){ return muonRef()->combinedMuon()  ->pt();
  }else if(hasMuonRef() && muonRef()->innerTrack()    .isNonnull()){ return muonRef()->innerTrack()    ->pt();
  }else if(hasMuonRef() && muonRef()->standAloneMuon().isNonnull()){ return muonRef()->standAloneMuon()->pt();
  }else if(hasTrackRef()&& trackRef()                 .isNonnull()){ return trackRef()                 ->pt();
  }else return 0.0f;
}

}
