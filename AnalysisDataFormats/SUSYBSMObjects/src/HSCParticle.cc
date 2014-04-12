#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

namespace susybsm {

int HSCParticle::type() const {
   if      ( hasTrackRef() && !hasMuonRef()){                               return HSCParticleType::innerTrack;
   }else if(!hasTrackRef() &&  hasMuonRef()){                               return HSCParticleType::standAloneMuon;
   }else if( hasTrackRef() &&  hasMuonRef() && muonRef()->isGlobalMuon()){ return HSCParticleType::globalMuon;
   }else if( hasTrackRef() &&  hasMuonRef() && muonRef()->isStandAloneMuon()){ return HSCParticleType::matchedStandAloneMuon;
   }else if( hasTrackRef() &&  hasMuonRef() && muonRef()->isTrackerMuon()){ return HSCParticleType::trackerMuon;
   }else                                                                    return HSCParticleType::unknown;
}

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
