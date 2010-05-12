#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"

namespace susybsm {

int HSCParticle::type() const {
   if      ( hasTrackRef() && !hasMuonRef()){                               return HSCParticleType::innerTrack;
   }else if(!hasTrackRef() &&  hasMuonRef()){                               return HSCParticleType::standAloneMuon;
   }else if( hasTrackRef() &&  hasMuonRef() && !muonRef()->isGlobalMuon()){ return HSCParticleType::matchedStandAloneMuon;
   }else if( hasTrackRef() &&  hasMuonRef() &&  muonRef()->isGlobalMuon()){ return HSCParticleType::globalMuon;
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
      case 0:    return dedxEstim1_;   break;
      case 1:    return dedxEstim2_;   break;
      case 2:    return dedxEstim3_;   break;
      case 3:    return dedxDiscrim1_; break;
      case 4:    return dedxDiscrim2_; break;
      case 5:    return dedxDiscrim3_; break;
      default:   return dedxEstim1_;   break;
   }
}

}
