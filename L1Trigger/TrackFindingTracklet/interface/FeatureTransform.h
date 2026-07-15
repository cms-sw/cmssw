#ifndef L1Trigger_TrackFindingTracklet_FeatureTransform_h
#define L1Trigger_TrackFindingTracklet_FeatureTransform_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include <vector>
#include <cmath>
#include <string>

namespace trklet {

  // Controls the conversion between TTTrack features and ML model training features
  // used for Track Quality simulation
  std::vector<float> featureTransform(TTTrack<Ref_Phase2TrackerDigi_>& aTrack,
                                      const std::vector<std::string>& featureNames) {
    // List input features for MVA in proper order below, the current features options are
    // {"phi", "eta", "z0", "bendchi2_bin", "nstub", "nlaymiss_interior", "chi2rphi_bin",
    // "chi2rz_bin"}
    //
    // To use more features, they must be created here and added to feature_map below
    std::vector<float> transformedFeatures;
    // Define feature map, filled as features are generated
    std::map<std::string, float> feature_map;
    // -------- calculate feature variables --------
    // calculate number of missed interior layers from hitpattern
    int tmp_trk_hitpattern = aTrack.hitPattern();
    int nbits = std::floor(std::log2(tmp_trk_hitpattern)) + 1;
    int lay_i = 0;
    int tmp_trk_nlaymiss_interior = 0;
    bool seq = false;
    for (int i = 0; i < nbits; i++) {
      lay_i = ((1 << i) & tmp_trk_hitpattern) >> i;  //0 or 1 in ith bit (right to left)
      if (lay_i && !seq)
        seq = true;  //sequence starts when first 1 found
      if (!lay_i && seq)
        tmp_trk_nlaymiss_interior++;
    }
    // binned chi2 variables
    int tmp_trk_bendchi2_bin = aTrack.getBendChi2Bits();
    int tmp_trk_chi2rphi_bin = aTrack.getChi2RPhiBits();
    int tmp_trk_chi2rz_bin = aTrack.getChi2RZBits();
    // get the nstub
    const std::vector<TTStubRef>& stubRefs = aTrack.getStubRefs();
    int tmp_trk_nstub = stubRefs.size();
    // get other variables directly from TTTrack
    float tmp_trk_z0 = aTrack.z0();
    float tmp_trk_z0_scaled = tmp_trk_z0 / std::abs(aTrack.minZ0);
    float tmp_trk_phi = aTrack.phi();
    float tmp_trk_eta = aTrack.eta();
    float tmp_trk_tanl = aTrack.tanL();
    // -------- fill the feature map ---------
    feature_map["nstub"] = float(tmp_trk_nstub);
    feature_map["z0"] = tmp_trk_z0;
    feature_map["z0_scaled"] = tmp_trk_z0_scaled;
    feature_map["phi"] = tmp_trk_phi;
    feature_map["eta"] = tmp_trk_eta;
    feature_map["nlaymiss_interior"] = float(tmp_trk_nlaymiss_interior);
    feature_map["bendchi2_bin"] = tmp_trk_bendchi2_bin;
    feature_map["chi2rphi_bin"] = tmp_trk_chi2rphi_bin;
    feature_map["chi2rz_bin"] = tmp_trk_chi2rz_bin;
    feature_map["tanl"] = tmp_trk_tanl;
    // fill tensor with track params
    transformedFeatures.reserve(featureNames.size());
    for (const std::string& feature : featureNames)
      transformedFeatures.push_back(feature_map[feature]);
    return transformedFeatures;
  }

}  // namespace trklet

#endif
