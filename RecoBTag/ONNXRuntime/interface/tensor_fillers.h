#ifndef RecoBTag_ONNXRuntime_tensor_fillers_h
#define RecoBTag_ONNXRuntime_tensor_fillers_h

#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"

namespace btagbtvdeep {
    
    void jet4vec_tensor_filler(float*& ptr, const btagbtvdeep::JetFeatures& jet_features);

    void seedTrack_tensor_filler(float*& ptr, const btagbtvdeep::SeedingTrackFeatures& seed_features);

    void neighbourTrack_tensor_filler(float*& ptr, const btagbtvdeep::TrackPairFeatures& neighbourTrack_features);

}

#endif 
