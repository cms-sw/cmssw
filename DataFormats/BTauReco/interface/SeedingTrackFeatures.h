#ifndef DataFormats_BTauReco_SeedingTrackFeatures_h
#define DataFormats_BTauReco_SeedingTrackFeatures_h

#include <vector>
#include "DataFormats/BTauReco/interface/TrackPairFeatures.h"

namespace btagbtvdeep {

class SeedingTrackFeatures {

  public:

    float seed_pt;
    float seed_eta;
    float seed_phi;
    float seed_mass;    
    float seed_dz;
    float seed_dxy;
    float seed_3D_ip;
    float seed_3D_sip;
    float seed_2D_ip;
    float seed_2D_sip;    
    float seed_3D_signedIp;
    float seed_3D_signedSip;
    float seed_2D_signedIp;
    float seed_2D_signedSip;  
    float seed_3D_TrackProbability;
    float seed_2D_TrackProbability;
    float seed_chi2reduced;
    float seed_nPixelHits;
    float seed_nHits;
    float seed_jetAxisDistance;
    float seed_jetAxisDlength;
    
    std::vector<btagbtvdeep::TrackPairFeatures> seed_nearTracks;
    

};

}

#endif 
