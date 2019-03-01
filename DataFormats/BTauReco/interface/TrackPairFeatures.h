#ifndef DataFormats_BTauReco_TrackPairFeatures_h
#define DataFormats_BTauReco_TrackPairFeatures_h

namespace btagbtvdeep {

class TrackPairFeatures {

  public:

    float nearTracks_pt;
    float nearTracks_eta;
    float nearTracks_phi;
    float nearTracks_mass;
    float nearTracks_dz;
    float nearTracks_dxy;
    float nearTracks_3D_ip;
    float nearTracks_3D_sip;
    float nearTracks_2D_ip;
    float nearTracks_2D_sip;
    float nearTracks_PCAdist;
    float nearTracks_PCAdsig;      
    float nearTracks_PCAonSeed_x;
    float nearTracks_PCAonSeed_y;
    float nearTracks_PCAonSeed_z;      
    float nearTracks_PCAonSeed_xerr;
    float nearTracks_PCAonSeed_yerr;
    float nearTracks_PCAonSeed_zerr;      
    float nearTracks_PCAonTrack_x;
    float nearTracks_PCAonTrack_y;
    float nearTracks_PCAonTrack_z;      
    float nearTracks_PCAonTrack_xerr;
    float nearTracks_PCAonTrack_yerr;
    float nearTracks_PCAonTrack_zerr; 
    float nearTracks_dotprodTrack;
    float nearTracks_dotprodSeed;
    float nearTracks_dotprodTrackSeed2D;
    float nearTracks_dotprodTrackSeed3D;
    float nearTracks_dotprodTrackSeedVectors2D;
    float nearTracks_dotprodTrackSeedVectors3D;      
    float nearTracks_PCAonSeed_pvd;
    float nearTracks_PCAonTrack_pvd;
    float nearTracks_PCAjetAxis_dist;
    float nearTracks_PCAjetMomenta_dotprod;
    float nearTracks_PCAjetDirs_DEta;
    float nearTracks_PCAjetDirs_DPhi;

};

}

#endif 
