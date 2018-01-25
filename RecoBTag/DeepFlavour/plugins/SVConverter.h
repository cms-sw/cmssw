#ifndef RecoBTag_DeepFlavour_SVConverter_h
#define RecoBTag_DeepFlavour_SVConverter_h

#include "deep_helpers.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexFeatures.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

namespace btagbtvdeep {

  class SVConverter { 
    public:

      static void SVToFeatures( const reco::VertexCompositePtrCandidate & sv,
                                const reco::Vertex & pv, const reco::Jet & jet,
                                SecondaryVertexFeatures & sv_features) {
    
        sv_features.pt = sv.pt();
        sv_features.deltaR = catch_infs_and_bound(
                               std::fabs(reco::deltaR(sv,jet))-0.5,
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
        sv_features.costhetasvpv = vertexDdotP(sv,pv);
        sv_features.enratio = sv.energy()/jet.energy();
    
      } 
  };

}

#endif //RecoSV_DeepFlavour_SVConverter_h
