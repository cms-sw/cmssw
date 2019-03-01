#include "RecoBTag/TensorFlow/interface/tensor_fillers.h"

namespace btagbtvdeep {

  // Note on setting tensor values:
  // Instead of using the more convenient tensor.matrix (etc) methods,
  // we can exploit that in the following methods values are set along
  // the innermost (= last) axis. Those values are stored contiguously in
  // the memory, so it is most performant to get the pointer to the first
  // value and use pointer arithmetic to iterate through the next pointers.

  void jet_tensor_filler(tensorflow::Tensor & tensor,
                         std::size_t jet_n,
                         const btagbtvdeep::DeepFlavourFeatures & features) {

    float* ptr = &tensor.matrix<float>()(jet_n, 0);

    // jet variables
    const auto & jet_features = features.jet_features;
    *ptr     = jet_features.pt;
    *(++ptr) = jet_features.eta;
    // number of elements in different collections
    *(++ptr) = features.c_pf_features.size();
    *(++ptr) = features.n_pf_features.size();
    *(++ptr) = features.sv_features.size();
    *(++ptr) = features.npv;
    // variables from ShallowTagInfo
    const auto & tag_info_features = features.tag_info_features;
    *(++ptr) = tag_info_features.trackSumJetEtRatio;
    *(++ptr) = tag_info_features.trackSumJetDeltaR;
    *(++ptr) = tag_info_features.vertexCategory;
    *(++ptr) = tag_info_features.trackSip2dValAboveCharm;
    *(++ptr) = tag_info_features.trackSip2dSigAboveCharm;
    *(++ptr) = tag_info_features.trackSip3dValAboveCharm;
    *(++ptr) = tag_info_features.trackSip3dSigAboveCharm;
    *(++ptr) = tag_info_features.jetNSelectedTracks;
    *(++ptr) = tag_info_features.jetNTracksEtaRel;

  }

  void db_tensor_filler(tensorflow::Tensor & tensor,
                         std::size_t jet_n,
                         const btagbtvdeep::DeepDoubleXFeatures & features) {

    float* ptr = &tensor.tensor<float, 3>()(jet_n, 0, 0);

    // variables from BoostedDoubleSVTagInfo
    const auto & tag_info_features = features.tag_info_features;
    *ptr = tag_info_features.jetNTracks;
    *(++ptr) = tag_info_features.jetNSecondaryVertices;
    *(++ptr) = tag_info_features.tau1_trackEtaRel_0; 
    *(++ptr) = tag_info_features.tau1_trackEtaRel_1; 
    *(++ptr) = tag_info_features.tau1_trackEtaRel_2; 
    *(++ptr) = tag_info_features.tau2_trackEtaRel_0; 
    *(++ptr) = tag_info_features.tau2_trackEtaRel_1; 
    *(++ptr) = tag_info_features.tau2_trackEtaRel_2; 
    *(++ptr) = tag_info_features.tau1_flightDistance2dSig; 
    *(++ptr) = tag_info_features.tau2_flightDistance2dSig; 
    *(++ptr) = tag_info_features.tau1_vertexDeltaR; 
    // Note: this variable is not used in the 27-input BDT
    //    *(++ptr) = tag_info_features.tau2_vertexDeltaR;
    *(++ptr) = tag_info_features.tau1_vertexEnergyRatio; 
    *(++ptr) = tag_info_features.tau2_vertexEnergyRatio; 
    *(++ptr) = tag_info_features.tau1_vertexMass; 
    *(++ptr) = tag_info_features.tau2_vertexMass; 
    *(++ptr) = tag_info_features.trackSip2dSigAboveBottom_0;
    *(++ptr) = tag_info_features.trackSip2dSigAboveBottom_1;
    *(++ptr) = tag_info_features.trackSip2dSigAboveCharm;
    *(++ptr) = tag_info_features.trackSip3dSig_0; 
    *(++ptr) = tag_info_features.tau1_trackSip3dSig_0; 
    *(++ptr) = tag_info_features.tau1_trackSip3dSig_1; 
    *(++ptr) = tag_info_features.trackSip3dSig_1; 
    *(++ptr) = tag_info_features.tau2_trackSip3dSig_0; 
    *(++ptr) = tag_info_features.tau2_trackSip3dSig_1; 
    *(++ptr) = tag_info_features.trackSip3dSig_2; 
    *(++ptr) = tag_info_features.trackSip3dSig_3; 
    *(++ptr) = tag_info_features.z_ratio;
  }

  void c_pf_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t c_pf_n,
                          const btagbtvdeep::ChargedCandidateFeatures & c_pf_features) {

    float* ptr = &tensor.tensor<float, 3>()(jet_n, c_pf_n, 0);

    *ptr     = c_pf_features.btagPf_trackEtaRel;
    *(++ptr) = c_pf_features.btagPf_trackPtRel;
    *(++ptr) = c_pf_features.btagPf_trackPPar;
    *(++ptr) = c_pf_features.btagPf_trackDeltaR;
    *(++ptr) = c_pf_features.btagPf_trackPParRatio;
    *(++ptr) = c_pf_features.btagPf_trackSip2dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip2dSig;
    *(++ptr) = c_pf_features.btagPf_trackSip3dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip3dSig;
    *(++ptr) = c_pf_features.btagPf_trackJetDistVal;
    *(++ptr) = c_pf_features.ptrel;
    *(++ptr) = c_pf_features.drminsv;
    *(++ptr) = c_pf_features.vtx_ass;
    *(++ptr) = c_pf_features.puppiw;
    *(++ptr) = c_pf_features.chi2;
    *(++ptr) = c_pf_features.quality;

  }
  
  void c_pf_reduced_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t c_pf_n,
                          const btagbtvdeep::ChargedCandidateFeatures & c_pf_features) {

    float* ptr = &tensor.tensor<float, 3>()(jet_n, c_pf_n, 0);

    *ptr     = c_pf_features.btagPf_trackEtaRel;
    *(++ptr) = c_pf_features.btagPf_trackPtRatio;
    *(++ptr) = c_pf_features.btagPf_trackPParRatio;
    *(++ptr) = c_pf_features.btagPf_trackSip2dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip2dSig;
    *(++ptr) = c_pf_features.btagPf_trackSip3dVal;
    *(++ptr) = c_pf_features.btagPf_trackSip3dSig;
    *(++ptr) = c_pf_features.btagPf_trackJetDistVal;

  }


  void n_pf_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t n_pf_n,
                          const btagbtvdeep::NeutralCandidateFeatures & n_pf_features) {

    float* ptr = &tensor.tensor<float, 3>()(jet_n, n_pf_n, 0);

    *ptr     = n_pf_features.ptrel;
    *(++ptr) = n_pf_features.deltaR;
    *(++ptr) = n_pf_features.isGamma;
    *(++ptr) = n_pf_features.hadFrac;
    *(++ptr) = n_pf_features.drminsv;
    *(++ptr) = n_pf_features.puppiw;

  }
  
  
  void sv_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t sv_n,
                          const btagbtvdeep::SecondaryVertexFeatures & sv_features) {

    float* ptr = &tensor.tensor<float, 3>()(jet_n, sv_n, 0);

    *ptr     = sv_features.pt;
    *(++ptr) = sv_features.deltaR;
    *(++ptr) = sv_features.mass;
    *(++ptr) = sv_features.ntracks;
    *(++ptr) = sv_features.chi2;
    *(++ptr) = sv_features.normchi2;
    *(++ptr) = sv_features.dxy;
    *(++ptr) = sv_features.dxysig;
    *(++ptr) = sv_features.d3d;
    *(++ptr) = sv_features.d3dsig;
    *(++ptr) = sv_features.costhetasvpv;
    *(++ptr) = sv_features.enratio;

  }

  
  void sv_reduced_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t sv_n,
                          const btagbtvdeep::SecondaryVertexFeatures & sv_features) {

    float* ptr = &tensor.tensor<float, 3>()(jet_n, sv_n, 0);

    *ptr     = sv_features.d3d;
    *(++ptr) = sv_features.d3dsig;

  }
  
  
  void seed_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t seed_n,
                          const btagbtvdeep::SeedingTrackFeatures & seed_features) {


    float* ptr = &tensor.tensor<float, 3>()(jet_n, seed_n, 0);    
    
     *ptr     = seed_features.seed_pt;
     *(++ptr) = seed_features.seed_eta;
     *(++ptr) = seed_features.seed_phi;
     *(++ptr) = seed_features.seed_mass;    
     *(++ptr) = seed_features.seed_dz;
     *(++ptr) = seed_features.seed_dxy;
     *(++ptr) = seed_features.seed_3D_ip;
     *(++ptr) = seed_features.seed_3D_sip;
     *(++ptr) = seed_features.seed_2D_ip;
     *(++ptr) = seed_features.seed_2D_sip;    
     *(++ptr) = seed_features.seed_3D_signedIp;
     *(++ptr) = seed_features.seed_3D_signedSip;
     *(++ptr) = seed_features.seed_2D_signedIp;
     *(++ptr) = seed_features.seed_2D_signedSip;  
     *(++ptr) = seed_features.seed_3D_TrackProbability;
     *(++ptr) = seed_features.seed_2D_TrackProbability;
     *(++ptr) = seed_features.seed_chi2reduced;
     *(++ptr) = seed_features.seed_nPixelHits;
     *(++ptr) = seed_features.seed_nHits;
     *(++ptr) = seed_features.seed_jetAxisDistance;
     *(++ptr) = seed_features.seed_jetAxisDlength;
     
  }  
  
  void neighbourTracks_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t seed_n,
                          const btagbtvdeep::SeedingTrackFeatures & seed_features) {

    
    std::vector<btagbtvdeep::TrackPairFeatures> neighbourTracks_features = seed_features.seed_nearTracks;      
   
    
    for(unsigned int t_i=0; t_i<neighbourTracks_features.size(); t_i++){  

    float* ptr = &tensor.tensor<float, 3>()(jet_n, t_i,  0);    
        
    *ptr  = neighbourTracks_features[t_i].nearTracks_pt;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_eta;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_phi;
     
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_dz;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_dxy;
     
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_mass;
     
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_3D_ip;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_3D_sip;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_2D_ip;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_2D_sip;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAdist;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAdsig;      
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonSeed_x;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonSeed_y;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonSeed_z;      
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonSeed_xerr;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonSeed_yerr;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonSeed_zerr;      
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonTrack_x;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonTrack_y;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonTrack_z;      
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonTrack_xerr;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonTrack_yerr;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonTrack_zerr; 
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_dotprodTrack;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_dotprodSeed;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_dotprodTrackSeed2D;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_dotprodTrackSeed3D;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_dotprodTrackSeedVectors2D;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_dotprodTrackSeedVectors3D;      
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonSeed_pvd;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAonTrack_pvd;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAjetAxis_dist;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAjetMomenta_dotprod;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAjetDirs_DEta;
     *(++ptr) = neighbourTracks_features[t_i].nearTracks_PCAjetDirs_DPhi;
        
     
    }


  }
  
  

}

