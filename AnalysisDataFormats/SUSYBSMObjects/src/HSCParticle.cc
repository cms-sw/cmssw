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

const reco::DeDxData& HSCParticle::dedx (int i) const {
   switch(i){
      case  0:    return dedxEstim1_;   break;
      case  1:    return dedxEstim2_;   break;
      case  2:    return dedxEstim3_;   break;
      case  3:    return dedxEstim4_;   break;
      case  4:    return dedxEstim5_;   break;
      case  5:    return dedxEstim6_;   break;
      case  6:    return dedxDiscrim1_; break;
      case  7:    return dedxDiscrim2_; break;
      case  8:    return dedxDiscrim3_; break;
      case  9:    return dedxDiscrim4_; break;
      case 10:    return dedxDiscrim5_; break;
      case 11:    return dedxDiscrim6_; break;
      default:    return dedxEstim1_;   break;
   }
}

}
