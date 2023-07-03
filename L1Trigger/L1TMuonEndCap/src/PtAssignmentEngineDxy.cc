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
  // 4 bends : set to zero if no CSC hit and thus RPC hit is used
  // 1 FR bit: for ME1 only
  // 1 Ring bit: for ME1 only
  // 1 track Theta taken from stub coordinate in ME2, ME3, ME4 (in this priority)
  // 4 RPC bits indicating if ME or RE hit was used in each station (S1, S2, S3, S4)
  // Total: 23 variables
  std::array<float, 6> x_dphi;
  std::array<float, 6> x_dphi_sign;
  std::array<float, 6> x_dtheta;
  std::array<float, 6> x_dtheta_sign;
  std::array<float, 1> x_trk_theta;
  std::array<float, 4> x_csc_pattern;

  // Initialize to zeros
  x_dphi.fill(0);
  x_dphi_sign.fill(0);
  x_dtheta.fill(0);
  x_dtheta_sign.fill(0);
  //
  x_trk_theta.fill(0);
  x_csc_pattern.fill(0);

  EMTFPtLUT data = track.PtLUT();

  const int invalid_dtheta = 127;
  const int invalid_dphi = 8191;

  // // Which stations have hits
  int st1 = (track.Mode() >= 8);
  int st2 = ((track.Mode() % 8) >= 4);
  int st3 = ((track.Mode() % 4) >= 2);
  int st4 = ((track.Mode() % 2) == 1);

  // Get valid pattern values
  if (st1)
    x_csc_pattern[0] = data.cpattern[0];
  if (st2)
    x_csc_pattern[1] = data.cpattern[1];
  if (st3)
    x_csc_pattern[2] = data.cpattern[2];
  if (st4)
    x_csc_pattern[3] = data.cpattern[3];

  // Calculate delta phi
  x_dphi[0] = (data.delta_ph[0] != invalid_dphi) ? data.delta_ph[0] : 0;
  x_dphi[1] = (data.delta_ph[1] != invalid_dphi) ? data.delta_ph[1] : 0;
  x_dphi[2] = (data.delta_ph[2] != invalid_dphi) ? data.delta_ph[2] : 0;
  x_dphi[3] = (data.delta_ph[3] != invalid_dphi) ? data.delta_ph[3] : 0;
  x_dphi[4] = (data.delta_ph[4] != invalid_dphi) ? data.delta_ph[4] : 0;
  x_dphi[5] = (data.delta_ph[5] != invalid_dphi) ? data.delta_ph[5] : 0;

  // Calculate delta theta
  x_dtheta[0] = (data.delta_th[0] != invalid_dtheta) ? data.delta_th[0] : 0;
  x_dtheta[1] = (data.delta_th[1] != invalid_dtheta) ? data.delta_th[1] : 0;
  x_dtheta[2] = (data.delta_th[2] != invalid_dtheta) ? data.delta_th[2] : 0;
  x_dtheta[3] = (data.delta_th[3] != invalid_dtheta) ? data.delta_th[3] : 0;
  x_dtheta[4] = (data.delta_th[4] != invalid_dtheta) ? data.delta_th[4] : 0;
  x_dtheta[5] = (data.delta_th[5] != invalid_dtheta) ? data.delta_th[5] : 0;

  // Get delta phi and theta signs
  x_dphi_sign[0] = data.sign_ph[0];
  x_dphi_sign[1] = data.sign_ph[1];
  x_dphi_sign[2] = data.sign_ph[2];
  x_dphi_sign[3] = data.sign_ph[3];
  x_dphi_sign[4] = data.sign_ph[4];
  x_dphi_sign[5] = data.sign_ph[5];

  x_dtheta_sign[0] = data.sign_th[0];
  x_dtheta_sign[1] = data.sign_th[1];
  x_dtheta_sign[2] = data.sign_th[2];
  x_dtheta_sign[3] = data.sign_th[3];
  x_dtheta_sign[4] = data.sign_th[4];
  x_dtheta_sign[5] = data.sign_th[5];

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
