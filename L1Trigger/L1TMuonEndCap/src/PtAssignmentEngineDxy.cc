#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngineDxy.h"

#include <cassert>
#include <iostream>
#include <sstream>

#include "helper.h"  // assert_no_abort

PtAssignmentEngineDxy::PtAssignmentEngineDxy() : graphDefDxy_(nullptr), sessionDxy_(nullptr) {}

PtAssignmentEngineDxy::~PtAssignmentEngineDxy() {
  if (sessionDxy_ != nullptr) {
    tensorflow::closeSession(sessionDxy_);
  }
  delete graphDefDxy_;
}

void PtAssignmentEngineDxy::configure(int verbose, const std::string pbFileNameDxy) {
  verbose_ = verbose;

  pbFileNameDxy_ = pbFileNameDxy;
  std::string pbFilePathDxy_ = "L1Trigger/L1TMuon/data/emtf_luts/" + pbFileNameDxy_;

  inputNameDxy_ = "input1";
  outputNamesDxy_ = {"Identity"};

  if (graphDefDxy_ == nullptr) {
    graphDefDxy_ = tensorflow::loadGraphDef(edm::FileInPath(pbFilePathDxy_).fullPath());
  }
  emtf_assert(graphDefDxy_ != nullptr);

  if (sessionDxy_ == nullptr) {
    sessionDxy_ = tensorflow::createSession(graphDefDxy_);
  }

  emtf_assert(sessionDxy_ != nullptr);
}

const PtAssignmentEngineAux2017& PtAssignmentEngineDxy::aux() const {
  static const PtAssignmentEngineAux2017 instance;
  return instance;
}

void PtAssignmentEngineDxy::calculate_pt_dxy(const EMTFTrack& track,
                                             emtf::Feature& feature,
                                             emtf::Prediction& prediction) const {
  // This is called for each track instead of for entire track collection as was done in Phase-2 implementation
  preprocessing_dxy(track, feature);
  call_tensorflow_dxy(feature, prediction);
  return;
}

void PtAssignmentEngineDxy::preprocessing_dxy(const EMTFTrack& track, emtf::Feature& feature) const {
  // Mimic Phase-1 EMTF input calculations
  // 6 delta Phis: S1-S2, S1-S3, S1-S4, S2-S3, S2-S4, S3-S4
  // 6 delta Thetas: S1-S2, S1-S3, S1-S4, S2-S3, S2-S4, S3-S4
  // 6 delta Phi signs: S1-S2, S1-S3, S1-S4, S2-S3, S2-S4, S3-S4
  // 6 delta Theta signs: S1-S2, S1-S3, S1-S4, S2-S3, S2-S4, S3-S4
  // 1 track Theta taken from stub coordinate in ME2, ME3, ME4 (in this priority)
  // 4 CSC pattern values (Run 2 convention): S1, S2, S3, S4
  // Total: 29 variables
  std::array<float, 6> x_dphi;
  std::array<float, 6> x_dphi_sign;
  std::array<float, 6> x_dtheta;
  std::array<float, 6> x_dtheta_sign;
  std::array<float, 1> x_trk_theta;
  std::array<float, 4> x_csc_pattern;

  // Initialize x_csc_pattern to zeros
  x_csc_pattern.fill(0);

  EMTFPtLUT data = track.PtLUT();

  const int invalid_dtheta = 127;
  const int invalid_dphi = 8191;

  // // Which stations have hits
  bool st1 = (track.Mode() >= 8);
  bool st2 = ((track.Mode() % 8) >= 4);
  bool st3 = ((track.Mode() % 4) >= 2);
  bool st4 = ((track.Mode() % 2) == 1);

  // Get valid pattern values
  if (st1)
    x_csc_pattern[0] = data.cpattern[0];
  if (st2)
    x_csc_pattern[1] = data.cpattern[1];
  if (st3)
    x_csc_pattern[2] = data.cpattern[2];
  if (st4)
    x_csc_pattern[3] = data.cpattern[3];

  for (int i = 0; i < 6; ++i) {  // There are 6 deltas between 4 stations.
    // Calculate delta phi
    x_dphi[i] = (data.delta_ph[i] != invalid_dphi) ? data.delta_ph[i] : 0;

    // Calculate delta theta
    x_dtheta[i] = (data.delta_th[i] != invalid_dtheta) ? data.delta_th[i] : 0;

    // Get delta phi and theta signs
    x_dphi_sign[i] = data.sign_ph[i];
    x_dtheta_sign[i] = data.sign_th[i];
  }

  // Set dPhi and dTheta values to 0 if there was no hit in the station
  if (!st1) {
    x_dphi[0] = 0;
    x_dphi[1] = 0;
    x_dphi[2] = 0;

    x_dtheta[0] = 0;
    x_dtheta[1] = 0;
    x_dtheta[2] = 0;
  }
  if (!st2) {
    x_dphi[0] = 0;
    x_dphi[3] = 0;
    x_dphi[4] = 0;

    x_dtheta[0] = 0;
    x_dtheta[3] = 0;
    x_dtheta[4] = 0;
  }
  if (!st3) {
    x_dphi[1] = 0;
    x_dphi[3] = 0;
    x_dphi[5] = 0;

    x_dtheta[1] = 0;
    x_dtheta[3] = 0;
    x_dtheta[5] = 0;
  }
  if (!st4) {
    x_dphi[2] = 0;
    x_dphi[4] = 0;
    x_dphi[5] = 0;

    x_dtheta[2] = 0;
    x_dtheta[4] = 0;
    x_dtheta[5] = 0;
  }

  x_trk_theta[0] = track.Theta_fp();

  // Set NN inputs
  feature = {{x_dphi[0],        x_dphi[1],        x_dphi[2],        x_dphi[3],        x_dphi[4],
              x_dphi[5],        x_dphi_sign[0],   x_dphi_sign[1],   x_dphi_sign[2],   x_dphi_sign[3],
              x_dphi_sign[4],   x_dphi_sign[5],   x_dtheta[0],      x_dtheta[1],      x_dtheta[2],
              x_dtheta[3],      x_dtheta[4],      x_dtheta[5],      x_dtheta_sign[0], x_dtheta_sign[1],
              x_dtheta_sign[2], x_dtheta_sign[3], x_dtheta_sign[4], x_dtheta_sign[5], x_csc_pattern[0],
              x_csc_pattern[1], x_csc_pattern[2], x_csc_pattern[3], x_trk_theta[0]}};

  return;
}

void PtAssignmentEngineDxy::call_tensorflow_dxy(const emtf::Feature& feature, emtf::Prediction& prediction) const {
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, emtf::NUM_FEATURES});
  std::vector<tensorflow::Tensor> outputs;
  emtf_assert(feature.size() == emtf::NUM_FEATURES);

  float* d = input.flat<float>().data();
  std::copy(feature.begin(), feature.end(), d);
  tensorflow::run(sessionDxy_, {{inputNameDxy_, input}}, outputNamesDxy_, &outputs);
  emtf_assert(outputs.size() == 1);
  emtf_assert(prediction.size() == emtf::NUM_PREDICTIONS);

  prediction.at(0) = outputs[0].matrix<float>()(0, 0);
  prediction.at(1) = outputs[0].matrix<float>()(0, 1);

  return;
}
