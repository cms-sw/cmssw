#include <cmath>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFContext.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DataUtils.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/DebugUtils.h"

#include "L1Trigger/L1TMuonEndCapPhase2/interface/Algo/ParameterAssignmentLayer.h"

using namespace emtf::phase2;
using namespace emtf::phase2::algo;

ParameterAssignmentLayer::ParameterAssignmentLayer(const EMTFContext& context) : context_(context) {}

void ParameterAssignmentLayer::apply(const bool& displaced_en, std::vector<track_t>& tracks) const {
  std::vector<int> feature_sites = {0, 1, 2,  3,  4, 5, 6, 7, 8, 9,  10, 11, 0, 1, 2, 3,  4,  5,  6,  7,
                                    8, 9, 10, 11, 0, 1, 2, 3, 4, 11, 0,  1,  2, 3, 4, 11, -1, -1, -1, -1};

  for (auto& track : tracks) {  // Begin loop tracks
    // Init Parameters
    track.pt = 0;
    track.rels = 0;
    track.dxy = 0;
    track.z0 = 0;
    track.beta = 0;

    track.pt_address = 0;
    track.rels_address = 0;
    track.dxy_address = 0;

    // Short-Circuit: Skip invalid tracks
    if (track.valid == 0) {
      continue;
    }

    // Get Features
    const auto& site_mask = track.site_mask;
    const auto& features = track.features;

    // Single batch of NTrackFeatures values
    tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, v3::kNumTrackFeatures});

    if (this->context_.config_.verbosity_ > 1) {
      edm::LogInfo("L1TEMTFpp") << "Parameter Assignment In"
                                << " disp " << displaced_en << " zone " << track.zone << " col " << track.col << " pat "
                                << track.pattern << " qual " << track.quality << " phi " << track.phi << " theta "
                                << track.theta << " features " << std::endl;
    }

    // Prepare input tensor
    float* input_data = input.flat<float>().data();

    for (int i_feature = 0; i_feature < v3::kNumTrackFeatures; ++i_feature) {
      const auto& feature = features[i_feature];
      const auto& feature_site = feature_sites[i_feature];

      bool mask_value = false;

      // Debug Info
      if (this->context_.config_.verbosity_ > 1 && i_feature > 0) {
        edm::LogInfo("L1TEMTFpp") << " ";
      }

      // Mask invalid sites
      if (feature_site > -1) {
        mask_value = (site_mask[feature_site] == 0);
      }

      if (mask_value) {
        (*input_data) = 0.;

        // Debug Info
        if (this->context_.config_.verbosity_ > 1) {
          edm::LogInfo("L1TEMTFpp") << "0";
        }
      } else {
        (*input_data) = feature.to_float();

        // Debug Info
        if (this->context_.config_.verbosity_ > 1) {
          edm::LogInfo("L1TEMTFpp") << feature.to_float();
        }
      }

      input_data++;
    }

    // Debug Info
    if (this->context_.config_.verbosity_ > 1) {
      edm::LogInfo("L1TEMTFpp") << std::endl;
    }

    // Select TF Session
    auto* session_ptr = context_.prompt_session_ptr_;

    if (displaced_en) {
      session_ptr = context_.disp_session_ptr_;
    }

    // Evaluate Prompt
    std::vector<tensorflow::Tensor> outputs;

    tensorflow::run(session_ptr,
                    {{"inputs", input}},  // Input layer name
                    {"Identity"},         // Output layer name
                    &outputs);

    // Assign parameters
    if (displaced_en) {
      // Read displaced pb outputs
      auto pt_address = outputs[0].matrix<float>()(0, 0);
      auto rels_address = outputs[0].matrix<float>()(0, 1);
      auto dxy_address = outputs[0].matrix<float>()(0, 2);

      track.pt_address = std::clamp<float>(pt_address, -512, 511);
      track.rels_address = std::clamp<float>(rels_address, -512, 511);
      track.dxy_address = std::clamp<float>(dxy_address, -512, 511);

      track.q = (track.pt_address < 0);
      track.pt = context_.activation_lut_.lookup_disp_pt(track.pt_address);
      track.rels = context_.activation_lut_.lookup_rels(track.rels_address);
      track.dxy = context_.activation_lut_.lookup_dxy(track.dxy_address);
    } else {
      // Read prompt pb outputs
      auto pt_address = outputs[0].matrix<float>()(0, 0);
      auto rels_address = outputs[0].matrix<float>()(0, 1);

      track.pt_address = std::clamp<float>(pt_address, -512, 511);
      track.rels_address = std::clamp<float>(rels_address, -512, 511);
      track.dxy_address = 0;

      track.q = (track.pt_address < 0);
      track.pt = context_.activation_lut_.lookup_prompt_pt(track.pt_address);
      track.rels = context_.activation_lut_.lookup_rels(track.rels_address);
      track.dxy = 0;
    }

    // DEBUG
    if (this->context_.config_.verbosity_ > 1) {
      edm::LogInfo("L1TEMTFpp") << "Parameter Assignment Out"
                                << " disp " << displaced_en << " zone " << track.zone << " col " << track.col << " pat "
                                << track.pattern << " qual " << track.quality << " q " << track.q << " pt " << track.pt
                                << " rels " << track.rels << " dxy " << track.dxy << " z0 " << track.z0 << " phi "
                                << track.phi << " theta " << track.theta << " beta " << track.beta << " pt_address "
                                << track.pt_address << " rels_address " << track.rels_address << " dxy_address "
                                << track.dxy_address << " valid " << track.valid << std::endl;
    }
  }  // End loop tracks
}
