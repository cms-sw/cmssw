#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"


#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <limits>

#include "getBestVertex.h"

namespace {
  
template<bool PROMPT>
struct mva {
  mva(const edm::ParameterSet &){}
  float operator()(reco::Track const & trk,
		   reco::BeamSpot const & beamSpot,
		   reco::VertexCollection const & vertices,
		   GBRForest const * forestP) const {

    auto const & forest = *forestP;
    auto tmva_pt_ = trk.pt();
    auto tmva_ndof_ = trk.ndof();
    auto tmva_nlayers_ = trk.hitPattern().trackerLayersWithMeasurement();
    auto tmva_nlayers3D_ = trk.hitPattern().pixelLayersWithMeasurement()
        + trk.hitPattern().numberOfValidStripLayersWithMonoAndStereo();
    auto tmva_nlayerslost_ = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);
    float chi2n =  trk.normalizedChi2();
    float chi2n_no1Dmod = chi2n;
    
    int count1dhits = 0;
    for (auto ith =trk.recHitsBegin(); ith!=trk.recHitsEnd(); ++ith) {
      const auto & hit = *(*ith);
      if (hit.dimension()==1) ++count1dhits;
    }
    
    if (count1dhits > 0) {
      float chi2 = trk.chi2();
      float ndof = trk.ndof();
      chi2n = (chi2+count1dhits)/float(ndof+count1dhits);
    }
    auto tmva_chi2n_ = chi2n;
    auto tmva_chi2n_no1dmod_ = chi2n_no1Dmod;
    auto tmva_eta_ = trk.eta();
    auto tmva_relpterr_ = float(trk.ptError())/std::max(float(trk.pt()),0.000001f);
    auto tmva_nhits_ = trk.numberOfValidHits();
    int lostIn = trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
    int lostOut = trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);
    auto tmva_minlost_ = std::min(lostIn,lostOut);
    auto tmva_lostmidfrac_ = trk.numberOfLostHits() / (trk.numberOfValidHits() + trk.numberOfLostHits());
   
    float gbrVals_[PROMPT ? 16 : 12];
    gbrVals_[0] = tmva_pt_;
    gbrVals_[1] = tmva_lostmidfrac_;
    gbrVals_[2] = tmva_minlost_;
    gbrVals_[3] = tmva_nhits_;
    gbrVals_[4] = tmva_relpterr_;
    gbrVals_[5] = tmva_eta_;
    gbrVals_[6] = tmva_chi2n_no1dmod_;
    gbrVals_[7] = tmva_chi2n_;
    gbrVals_[8] = tmva_nlayerslost_;
    gbrVals_[9] = tmva_nlayers3D_;
    gbrVals_[10] = tmva_nlayers_;
    gbrVals_[11] = tmva_ndof_;

    if (PROMPT) {
      auto tmva_absd0_ = std::abs(trk.dxy(beamSpot.position()));
      auto tmva_absdz_ = std::abs(trk.dz(beamSpot.position()));
      Point bestVertex = getBestVertex(trk,vertices);
      auto tmva_absd0PV_ = std::abs(trk.dxy(bestVertex));
      auto tmva_absdzPV_ = std::abs(trk.dz(bestVertex));
      
      gbrVals_[12] = tmva_absd0PV_;
      gbrVals_[13] = tmva_absdzPV_;
      gbrVals_[14] = tmva_absdz_;
      gbrVals_[15] = tmva_absd0_;
    }

 

    
    return forest.GetClassifier(gbrVals_);
    
  }

  static const char * name();

  static void fillDescriptions(edm::ParameterSetDescription & desc) {
  }
  
  
};

  using TrackMVAClassifierDetached = TrackMVAClassifier<mva<false>>;
  using TrackMVAClassifierPrompt = TrackMVAClassifier<mva<true>>;
  template<>
  const char * mva<false>::name() { return "TrackMVAClassifierDetached";}
  template<>
  const char * mva<true>::name() { return "TrackMVAClassifierPrompt";}
  
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackMVAClassifierDetached);
DEFINE_FWK_MODULE(TrackMVAClassifierPrompt);
