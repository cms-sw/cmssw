#include "RecoBTag/FeatureTools/interface/deep_helpers.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "RecoBTag/FeatureTools/interface/SecondaryVertexConverter.h"

namespace btagbtvdeep {


  void svToFeatures( const reco::VertexCompositePtrCandidate & sv,
		     const reco::Vertex & pv, const reco::Jet & jet,
		     SecondaryVertexFeatures & sv_features,
		     const bool flip) {

    math::XYZVector jet_dir = jet.momentum().Unit();
    sv_features.pt = sv.pt();
    sv_features.deltaR = catch_infs_and_bound(std::fabs(reco::deltaR(sv, jet_dir))-0.5,
					      0,-2,0);
    sv_features.mass = sv.mass();
    sv_features.ntracks = sv.numberOfDaughters();
    sv_features.chi2 = sv.vertexChi2();
    sv_features.normchi2 = catch_infs_and_bound(sv_features.chi2/sv.vertexNdof(),
						1000, -1000, 1000);
    const auto & dxy_meas = vertexDxy(sv,pv);
    sv_features.dxy = dxy_meas.value();
    sv_features.dxysig = catch_infs_and_bound(dxy_meas.value()/dxy_meas.error(),
					      0,-1,800);
    const auto & d3d_meas = vertexD3d(sv,pv);
    sv_features.d3d = d3d_meas.value();
    sv_features.d3dsig = catch_infs_and_bound(d3d_meas.value()/d3d_meas.error(),
					      0,-1,800);
    sv_features.costhetasvpv = (flip ? -1.f : 1.f)* vertexDdotP(sv,pv);
    sv_features.enratio = sv.energy()/jet.energy();

  }

}

