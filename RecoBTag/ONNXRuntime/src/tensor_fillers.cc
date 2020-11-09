#include "RecoBTag/ONNXRuntime/interface/tensor_fillers.h"

namespace btagbtvdeep {

    void jet4vec_tensor_filler(float*& ptr, const btagbtvdeep::JetFeatures& jet_features) {

        *ptr = jet_features.pt;
        *(++ptr) = jet_features.eta;
        *(++ptr) = jet_features.phi;
        *(++ptr) = jet_features.mass;

    }


    void seedTrack_tensor_filler(float*& ptr, const btagbtvdeep::SeedingTrackFeatures& seed_features) {

        *ptr = seed_features.pt;
        *(++ptr) = seed_features.eta;
        *(++ptr) = seed_features.phi;
        *(++ptr) = seed_features.mass;
        *(++ptr) = seed_features.dz;
        *(++ptr) = seed_features.dxy;
        *(++ptr) = seed_features.ip3D;
        *(++ptr) = seed_features.sip3D;
        *(++ptr) = seed_features.ip2D;
        *(++ptr) = seed_features.sip2D;
        *(++ptr) = seed_features.signedIp3D;
        *(++ptr) = seed_features.signedSip3D;
        *(++ptr) = seed_features.signedIp2D;
        *(++ptr) = seed_features.signedSip2D;
        *(++ptr) = seed_features.trackProbability3D;
        *(++ptr) = seed_features.trackProbability2D;
        *(++ptr) = seed_features.chi2reduced;
        *(++ptr) = seed_features.nPixelHits;
        *(++ptr) = seed_features.nHits;
        *(++ptr) = seed_features.jetAxisDistance;
        *(++ptr) = seed_features.jetAxisDlength;

    }

    void neighbourTrack_tensor_filler(float*& ptr, const btagbtvdeep::TrackPairFeatures& neighbourTrack_features) {

        *ptr = neighbourTrack_features.pt;
        *(++ptr) = neighbourTrack_features.eta;
        *(++ptr) = neighbourTrack_features.phi;
        *(++ptr) = neighbourTrack_features.dz;
        *(++ptr) = neighbourTrack_features.dxy;
        *(++ptr) = neighbourTrack_features.mass;
        *(++ptr) = neighbourTrack_features.ip3D;
        *(++ptr) = neighbourTrack_features.sip3D;
        *(++ptr) = neighbourTrack_features.ip2D;
        *(++ptr) = neighbourTrack_features.sip2D;
        *(++ptr) = neighbourTrack_features.distPCA;
        *(++ptr) = neighbourTrack_features.dsigPCA;
        *(++ptr) = neighbourTrack_features.x_PCAonSeed;
        *(++ptr) = neighbourTrack_features.y_PCAonSeed;
        *(++ptr) = neighbourTrack_features.z_PCAonSeed;
        *(++ptr) = neighbourTrack_features.xerr_PCAonSeed;
        *(++ptr) = neighbourTrack_features.yerr_PCAonSeed;
        *(++ptr) = neighbourTrack_features.zerr_PCAonSeed;
        *(++ptr) = neighbourTrack_features.x_PCAonTrack;
        *(++ptr) = neighbourTrack_features.y_PCAonTrack;
        *(++ptr) = neighbourTrack_features.z_PCAonTrack;
        *(++ptr) = neighbourTrack_features.xerr_PCAonTrack;
        *(++ptr) = neighbourTrack_features.yerr_PCAonTrack;
        *(++ptr) = neighbourTrack_features.zerr_PCAonTrack;
        *(++ptr) = neighbourTrack_features.dotprodTrack;
        *(++ptr) = neighbourTrack_features.dotprodSeed;
        *(++ptr) = neighbourTrack_features.dotprodTrackSeed2D;
        *(++ptr) = neighbourTrack_features.dotprodTrackSeed3D;
        *(++ptr) = neighbourTrack_features.dotprodTrackSeedVectors2D;
        *(++ptr) = neighbourTrack_features.dotprodTrackSeedVectors3D;
        *(++ptr) = neighbourTrack_features.pvd_PCAonSeed;
        *(++ptr) = neighbourTrack_features.pvd_PCAonTrack;
        *(++ptr) = neighbourTrack_features.dist_PCAjetAxis;
        *(++ptr) = neighbourTrack_features.dotprod_PCAjetMomenta;
        *(++ptr) = neighbourTrack_features.deta_PCAjetDirs;
        *(++ptr) = neighbourTrack_features.dphi_PCAjetDirs;

    }

}  // namespace btagbtvdeep
