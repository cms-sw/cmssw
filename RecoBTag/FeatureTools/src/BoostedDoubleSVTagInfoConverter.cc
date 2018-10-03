#include "RecoBTag/FeatureTools/interface/BoostedDoubleSVTagInfoConverter.h"
#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfoFeatures.h"

#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfo.h"
#include "DataFormats/BTauReco/interface/TaggingVariable.h"

namespace btagbtvdeep {

  void doubleBTagToFeatures(const reco::TaggingVariableList & tag_info_vars,
			    BoostedDoubleSVTagInfoFeatures & tag_info_features) {

    tag_info_features.jetNTracks = tag_info_vars.get(reco::btau::jetNTracks, -999);
    tag_info_features.jetNSecondaryVertices = tag_info_vars.get(reco::btau::jetNSecondaryVertices, -999);
    tag_info_features.trackSip3dSig_0 = tag_info_vars.get(reco::btau::trackSip3dSig_0, -999);
    tag_info_features.trackSip3dSig_1 = tag_info_vars.get(reco::btau::trackSip3dSig_1, -999);
    tag_info_features.trackSip3dSig_2 = tag_info_vars.get(reco::btau::trackSip3dSig_2, -999);
    tag_info_features.trackSip3dSig_3 = tag_info_vars.get(reco::btau::trackSip3dSig_3, -999);
    tag_info_features.tau1_trackSip3dSig_0 = tag_info_vars.get(reco::btau::tau1_trackSip3dSig_0, -999);
    tag_info_features.tau1_trackSip3dSig_1 = tag_info_vars.get(reco::btau::tau1_trackSip3dSig_1, -999);
    tag_info_features.tau2_trackSip3dSig_0 = tag_info_vars.get(reco::btau::tau2_trackSip3dSig_0, -999);
    tag_info_features.tau2_trackSip3dSig_1 = tag_info_vars.get(reco::btau::tau2_trackSip3dSig_1, -999);
    tag_info_features.trackSip2dSigAboveBottom_0 = tag_info_vars.get(reco::btau::trackSip2dSigAboveBottom_0, -999);
    tag_info_features.trackSip2dSigAboveBottom_1 = tag_info_vars.get(reco::btau::trackSip2dSigAboveBottom_1, -999);
    tag_info_features.trackSip2dSigAboveCharm = tag_info_vars.get(reco::btau::trackSip2dSigAboveCharm, -999);
    tag_info_features.tau1_trackEtaRel_0 = tag_info_vars.get(reco::btau::tau1_trackEtaRel_0, -999);
    tag_info_features.tau1_trackEtaRel_1 = tag_info_vars.get(reco::btau::tau1_trackEtaRel_1, -999);
    tag_info_features.tau1_trackEtaRel_2 = tag_info_vars.get(reco::btau::tau1_trackEtaRel_2, -999);
    tag_info_features.tau2_trackEtaRel_0 = tag_info_vars.get(reco::btau::tau2_trackEtaRel_0, -999);
    tag_info_features.tau2_trackEtaRel_1 = tag_info_vars.get(reco::btau::tau2_trackEtaRel_1, -999);
    tag_info_features.tau2_trackEtaRel_2 = tag_info_vars.get(reco::btau::tau2_trackEtaRel_2, -999);
    tag_info_features.tau1_vertexMass = tag_info_vars.get(reco::btau::tau1_vertexMass, -999);
    tag_info_features.tau1_vertexEnergyRatio = tag_info_vars.get(reco::btau::tau1_vertexEnergyRatio, -999);
    tag_info_features.tau1_flightDistance2dSig = tag_info_vars.get(reco::btau::tau1_flightDistance2dSig, -999);
    tag_info_features.tau1_vertexDeltaR = tag_info_vars.get(reco::btau::tau1_vertexDeltaR, -999);
    tag_info_features.tau2_vertexMass = tag_info_vars.get(reco::btau::tau2_vertexMass, -999);
    tag_info_features.tau2_vertexEnergyRatio = tag_info_vars.get(reco::btau::tau2_vertexEnergyRatio, -999);
    tag_info_features.tau2_flightDistance2dSig = tag_info_vars.get(reco::btau::tau2_flightDistance2dSig, -999);
    tag_info_features.tau2_vertexDeltaR = tag_info_vars.get(reco::btau::tau2_vertexDeltaR, -999); // not used
    tag_info_features.z_ratio = tag_info_vars.get(reco::btau::z_ratio, -999);

  }

}



