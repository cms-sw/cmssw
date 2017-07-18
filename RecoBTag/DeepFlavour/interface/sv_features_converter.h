#ifndef RecoSV_DeepFlavour_sv_features_converter_h
#define RecoSV_DeepFlavour_sv_features_converter_h

#include "RecoBTag/DeepFlavour/interface/deep_helpers.h"

namespace deep {

  template <typename SVType,
            typename PVType, typename JetType,
            typename SVFeaturesType>
  void sv_features_converter( const SVType & sv,
                              const PVType & pv, const JetType & jet,
                              SVFeaturesType & sv_features) {

    sv_features.pt = sv.pt();
    sv_features.eta = sv.eta();
    sv_features.phi = sv.phi();
    sv_features.etarel = deep::catch_infs_and_bound(
                           std::fabs(sv.eta()-jet.eta())-0.5,
                           0,-2,0);
    sv_features.phirel = deep::catch_infs_and_bound(
                           std::fabs(reco::deltaPhi(sv.phi(),jet.phi()))-0.5,
                           0,-2,0);
    sv_features.deltaR = deep::catch_infs_and_bound(
                           std::fabs(reco::deltaR(sv,jet))-0.5,
                           0,-2,0);
    sv_features.mass = sv.mass();
    sv_features.ntracks = sv.numberOfDaughters();
    sv_features.chi2 = sv.vertexChi2();
    sv_features.ndf = sv.vertexNdof();
    sv_features.normchi2 = deep::catch_infs_and_bound(
                             sv_features.chi2/sv_features.ndf,
                             1000, -1000, 1000);
    const auto & dxy_meas = deep::vertexDxy(sv,pv);
    sv_features.dxy = dxy_meas.value();
    sv_features.dxyerr = deep::catch_infs_and_bound(
                           dxy_meas.error()-2,
                           0,-2,0);
    sv_features.dxysig = deep::catch_infs_and_bound(
                           sv_features.dxy/dxy_meas.error(),
                           0,-1,800);
    const auto & d3d_meas = deep::vertexD3d(sv,pv);
    sv_features.d3d = d3d_meas.value();
    sv_features.d3derr = deep::catch_infs_and_bound(
                           d3d_meas.error()-2,
                           0,-2,0);
    sv_features.d3dsig = deep::catch_infs_and_bound(
                           d3d_meas.value()/d3d_meas.error(),
                           0,-1,800);
    sv_features.costhetasvpv = deep::vertexDdotP(sv,pv);
    sv_features.enratio = sv.energy()/jet.energy();

  } 


}

#endif //RecoSV_DeepFlavour_sv_features_converter_h
