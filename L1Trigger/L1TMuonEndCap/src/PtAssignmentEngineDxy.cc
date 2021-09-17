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

  inputNameDxy_ = "batch_normalization_1_input";
  outputNamesDxy_ = {"dense_4/BiasAdd"};

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
  std::array<float, 6> x_dtheta;
  std::array<float, 4> x_bend_emtf;
  std::array<float, 1> x_fr_emtf;
  std::array<float, 1> x_trk_theta;
  std::array<float, 1> x_me11ring;
  std::array<float, 4> x_rpcbit;

  // Initialize to zeros
  x_dphi.fill(0);
  x_dtheta.fill(0);
  //
  x_bend_emtf.fill(0);
  x_fr_emtf.fill(0);
  x_trk_theta.fill(0);
  x_me11ring.fill(0);
  x_rpcbit.fill(0);

  EMTFPtLUT data = track.PtLUT();

  const int invalid_dtheta = 127;
  const int invalid_dphi = 8191;

  // // Variables to extract from the PtLUT
  int dPhi_12, dPhi_13, dPhi_14, dPhi_23, dPhi_24, dPhi_34;
  int dTh_12, dTh_13, dTh_14, dTh_23, dTh_24, dTh_34;
  int fr_1;
  int bend_1, bend_2, bend_3, bend_4;
  int rpc_1, rpc_2, rpc_3, rpc_4;
  int St1_ring2 = data.st1_ring2;

  int pat1 = -99, pat2 = -99, pat3 = -99, pat4 = -99;

  // // Which stations have hits
  int st1 = (track.Mode() >= 8);
  int st2 = ((track.Mode() % 8) >= 4);
  int st3 = ((track.Mode() % 4) >= 2);
  int st4 = ((track.Mode() % 2) == 1);

  // Get valid pattern values
  if (st1)
    pat1 = data.cpattern[0];
  if (st2)
    pat2 = data.cpattern[1];
  if (st3)
    pat3 = data.cpattern[2];
  if (st4)
    pat4 = data.cpattern[3];

  // F/R bit
  fr_1 = data.fr[0];

  // RPC hit in station
  rpc_1 = (st1 ? (pat1 == 0) : 0);
  rpc_2 = (st2 ? (pat2 == 0) : 0);
  rpc_3 = (st3 ? (pat3 == 0) : 0);
  rpc_4 = (st4 ? (pat4 == 0) : 0);

  // Calculate bends from patterns
  bend_1 = aux().calcBendFromPattern(pat1, track.Endcap());
  bend_2 = aux().calcBendFromPattern(pat2, track.Endcap());
  bend_3 = aux().calcBendFromPattern(pat3, track.Endcap());
  bend_4 = aux().calcBendFromPattern(pat4, track.Endcap());

  // Invalid bend value is 0 in the NN
  if (bend_1 == -99)
    bend_1 = 0;
  if (bend_2 == -99)
    bend_2 = 0;
  if (bend_3 == -99)
    bend_3 = 0;
  if (bend_4 == -99)
    bend_4 = 0;

  // In the emulator RPCs get assigned abs(bend) = 5. This needs to be 0 for the NN.
  if (std::abs(bend_1) == 5 && rpc_1 == 1)
    bend_1 = 0;
  if (std::abs(bend_2) == 5 && rpc_2 == 1)
    bend_2 = 0;
  if (std::abs(bend_3) == 5 && rpc_3 == 1)
    bend_3 = 0;
  if (std::abs(bend_4) == 5 && rpc_4 == 1)
    bend_4 = 0;

  // Calculate delta phi
  dPhi_12 = (data.delta_ph[0] != invalid_dphi) ? data.delta_ph[0] * (data.sign_ph[0] ? 1 : -1) : 0;
  dPhi_13 = (data.delta_ph[1] != invalid_dphi) ? data.delta_ph[1] * (data.sign_ph[1] ? 1 : -1) : 0;
  dPhi_14 = (data.delta_ph[2] != invalid_dphi) ? data.delta_ph[2] * (data.sign_ph[2] ? 1 : -1) : 0;
  dPhi_23 = (data.delta_ph[3] != invalid_dphi) ? data.delta_ph[3] * (data.sign_ph[3] ? 1 : -1) : 0;
  dPhi_24 = (data.delta_ph[4] != invalid_dphi) ? data.delta_ph[4] * (data.sign_ph[4] ? 1 : -1) : 0;
  dPhi_34 = (data.delta_ph[5] != invalid_dphi) ? data.delta_ph[5] * (data.sign_ph[5] ? 1 : -1) : 0;

  // Calculate delta theta
  dTh_12 = (data.delta_th[0] != invalid_dtheta) ? data.delta_th[0] * (data.sign_th[0] ? 1 : -1) : 0;
  dTh_13 = (data.delta_th[1] != invalid_dtheta) ? data.delta_th[1] * (data.sign_th[1] ? 1 : -1) : 0;
  dTh_14 = (data.delta_th[2] != invalid_dtheta) ? data.delta_th[2] * (data.sign_th[2] ? 1 : -1) : 0;
  dTh_23 = (data.delta_th[3] != invalid_dtheta) ? data.delta_th[3] * (data.sign_th[3] ? 1 : -1) : 0;
  dTh_24 = (data.delta_th[4] != invalid_dtheta) ? data.delta_th[4] * (data.sign_th[4] ? 1 : -1) : 0;
  dTh_34 = (data.delta_th[5] != invalid_dtheta) ? data.delta_th[5] * (data.sign_th[5] ? 1 : -1) : 0;

  // Set dPhi and dTheta values to 0 if there was no hit in the station
  if (!st1) {
    dPhi_12 = 0;
    dPhi_13 = 0;
    dPhi_14 = 0;

    dTh_12 = 0;
    dTh_13 = 0;
    dTh_14 = 0;
  }
  if (!st2) {
    dPhi_12 = 0;
    dPhi_23 = 0;
    dPhi_24 = 0;

    dTh_12 = 0;
    dTh_23 = 0;
    dTh_24 = 0;
  }
  if (!st3) {
    dPhi_13 = 0;
    dPhi_23 = 0;
    dPhi_34 = 0;

    dTh_13 = 0;
    dTh_23 = 0;
    dTh_34 = 0;
  }
  if (!st4) {
    dPhi_14 = 0;
    dPhi_24 = 0;
    dPhi_34 = 0;

    dTh_14 = 0;
    dTh_24 = 0;
    dTh_34 = 0;
  }

  // Set NN inputs

  // NN was trained with the wrong sign convention. TO BE CHANGED LATER!
  x_dphi[0] = dPhi_12;
  x_dphi[1] = dPhi_13;
  x_dphi[2] = dPhi_14;
  x_dphi[3] = dPhi_23;
  x_dphi[4] = dPhi_24;
  x_dphi[5] = dPhi_34;

  // NN was trained with the wrong sign convention. TO BE CHANGED LATER!
  x_dtheta[0] = dTh_12;
  x_dtheta[1] = dTh_13;
  x_dtheta[2] = dTh_14;
  x_dtheta[3] = dTh_23;
  x_dtheta[4] = dTh_24;
  x_dtheta[5] = dTh_34;

  // NN was trained with the wrong sign convention. TO BE CHANGED LATER!
  x_bend_emtf[0] = bend_1;
  x_bend_emtf[1] = bend_2;
  x_bend_emtf[2] = bend_3;
  x_bend_emtf[3] = bend_4;

  x_fr_emtf[0] = fr_1;
  x_trk_theta[0] = track.Theta_fp();
  x_me11ring[0] = St1_ring2;

  x_rpcbit[0] = rpc_1;
  x_rpcbit[1] = rpc_2;
  x_rpcbit[2] = rpc_3;
  x_rpcbit[3] = rpc_4;

  feature = {{x_dphi[0],      x_dphi[1],      x_dphi[2],      x_dphi[3],      x_dphi[4],    x_dphi[5],
              x_dtheta[0],    x_dtheta[1],    x_dtheta[2],    x_dtheta[3],    x_dtheta[4],  x_dtheta[5],
              x_bend_emtf[0], x_bend_emtf[1], x_bend_emtf[2], x_bend_emtf[3], x_fr_emtf[0], x_trk_theta[0],
              x_me11ring[0],  x_rpcbit[0],    x_rpcbit[1],    x_rpcbit[2],    x_rpcbit[3]}};
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

  const float reg_pt_scale = 100.0;  // a scale factor applied to regression during training
  const float reg_dxy_scale = 1.0;   // a scale factor applied to regression during training

  prediction.at(0) = outputs[0].matrix<float>()(0, 0);
  prediction.at(1) = outputs[0].matrix<float>()(0, 1);

  // Remove scale factor used during training
  prediction.at(0) /= reg_pt_scale;
  prediction.at(1) /= reg_dxy_scale;
  return;
}
