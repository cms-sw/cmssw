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
  
  void jet4vec_tensor_filler(tensorflow::Tensor & tensor,
                         std::size_t jet_n,
                         const btagbtvdeep::DeepFlavourFeatures & features) {

    float* ptr = &tensor.matrix<float>()(jet_n, 0);

    // jet 4 vector variables
    const auto & jet_features = features.jet_features;
    *ptr     = jet_features.pt;
    *(++ptr) = jet_features.eta;
    *(++ptr) = jet_features.phi;
    *(++ptr) = jet_features.mass;
    

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
    
     *ptr     = seed_features.pt;
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
  
  void neighbourTracks_tensor_filler(tensorflow::Tensor & tensor,
                          std::size_t jet_n,
                          std::size_t seed_n,
                          const btagbtvdeep::SeedingTrackFeatures & seed_features) {

    
    std::vector<btagbtvdeep::TrackPairFeatures> neighbourTracks_features = seed_features.nearTracks;      
   
    
    for(unsigned int t_i=0; t_i<neighbourTracks_features.size(); t_i++){  

     float* ptr = &tensor.tensor<float, 3>()(jet_n, t_i,  0);    
        
     *ptr  = neighbourTracks_features[t_i].pt;
     *(++ptr) = neighbourTracks_features[t_i].eta;
     *(++ptr) = neighbourTracks_features[t_i].phi;     
     *(++ptr) = neighbourTracks_features[t_i].dz;
     *(++ptr) = neighbourTracks_features[t_i].dxy;     
     *(++ptr) = neighbourTracks_features[t_i].mass;     
     *(++ptr) = neighbourTracks_features[t_i].ip3D;
     *(++ptr) = neighbourTracks_features[t_i].sip3D;
     *(++ptr) = neighbourTracks_features[t_i].ip2D;
     *(++ptr) = neighbourTracks_features[t_i].sip2D;
     *(++ptr) = neighbourTracks_features[t_i].distPCA;
     *(++ptr) = neighbourTracks_features[t_i].dsigPCA;      
     *(++ptr) = neighbourTracks_features[t_i].x_PCAonSeed;
     *(++ptr) = neighbourTracks_features[t_i].y_PCAonSeed;
     *(++ptr) = neighbourTracks_features[t_i].z_PCAonSeed;      
     *(++ptr) = neighbourTracks_features[t_i].xerr_PCAonSeed;
     *(++ptr) = neighbourTracks_features[t_i].yerr_PCAonSeed;
     *(++ptr) = neighbourTracks_features[t_i].zerr_PCAonSeed;      
     *(++ptr) = neighbourTracks_features[t_i].x_PCAonTrack;
     *(++ptr) = neighbourTracks_features[t_i].y_PCAonTrack;
     *(++ptr) = neighbourTracks_features[t_i].z_PCAonTrack;      
     *(++ptr) = neighbourTracks_features[t_i].xerr_PCAonTrack;
     *(++ptr) = neighbourTracks_features[t_i].yerr_PCAonTrack;
     *(++ptr) = neighbourTracks_features[t_i].zerr_PCAonTrack; 
     *(++ptr) = neighbourTracks_features[t_i].dotprodTrack;
     *(++ptr) = neighbourTracks_features[t_i].dotprodSeed;
     *(++ptr) = neighbourTracks_features[t_i].dotprodTrackSeed2D;
     *(++ptr) = neighbourTracks_features[t_i].dotprodTrackSeed3D;
     *(++ptr) = neighbourTracks_features[t_i].dotprodTrackSeedVectors2D;
     *(++ptr) = neighbourTracks_features[t_i].dotprodTrackSeedVectors3D;      
     *(++ptr) = neighbourTracks_features[t_i].pvd_PCAonSeed;
     *(++ptr) = neighbourTracks_features[t_i].pvd_PCAonTrack;
     *(++ptr) = neighbourTracks_features[t_i].dist_PCAjetAxis;
     *(++ptr) = neighbourTracks_features[t_i].dotprod_PCAjetMomenta;
     *(++ptr) = neighbourTracks_features[t_i].deta_PCAjetDirs;
     *(++ptr) = neighbourTracks_features[t_i].dphi_PCAjetDirs;
        
     
    }


  }
  
  

}

