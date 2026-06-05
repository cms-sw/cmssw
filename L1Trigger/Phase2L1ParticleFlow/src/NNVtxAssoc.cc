// Package:      L1Trigger/Phase2L1ParticleFlow
// Class:        NNVtxAssoc
// Description:  Designed to run the track to vertex associations created by the E2E NNVtx.
//               TTTrackNetworkSelector either accepts or rejects that a PF object's (t) track is associated to a vertex (v).
// Authors:      Kai Hong Law and Benjamin Radburn-Smith
// Created:      February 2025

#include "L1Trigger/Phase2L1ParticleFlow/interface/NNVtxAssoc.h"

#ifdef CMSSW_GIT_HASH
#include "DataFormats/L1TParticleFlow/interface/PFTrack.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include <cmath>
#include <iomanip>

NNVtxAssoc::NNVtxAssoc(const std::shared_ptr<hls4mlEmulator::Model> model,
                       const double AssociationThreshold,
                       const std::vector<double>& AssociationNetworkZ0binning,
                       const std::vector<double>& AssociationNetworkEtaBounds,
                       const std::vector<double>& AssociationNetworkZ0ResBins,
                       bool debug)
    : associationThreshold_((l1ct::nn_assoc_t)AssociationThreshold),
      z0_binning_(AssociationNetworkZ0binning),
      eta_bins_(AssociationNetworkEtaBounds),
      res_bins_(AssociationNetworkZ0ResBins),
      modelRef_(model),
      isDebugEnabled_(debug) {
  fPt_ = 0;
  fMVA_ = 0;
  fResBin_ = 0;
  fDz_ = 0;
  log_.setf(std::ios::fixed, std::ios::floatfield);
  log_.precision(3);
}

void NNVtxAssoc::TTTrackNetworkSelector(const l1ct::PFRegionEmu& region,
                                        const l1ct::TkObjEmu& t,
                                        const l1ct::PVObjEmu& v,
                                        l1ct::nn_assoc_t& score) {
  nn_inputtype modelInput[N_NN_ASSOC_FEATURES] = {};  // Do something
  classtype classresult;

  int resbin = 0;
  int Nresbins = std::size(associationNetworkEtaBounds);

  l1ct::eta_t temp_hwEta = t.hwEta;
  if (temp_hwEta < 0)
    temp_hwEta = -temp_hwEta;
  for (int ibin = 0; ibin < Nresbins; ++ibin) {
    if (temp_hwEta > l1ct::Scales::makeEta(associationNetworkEtaBounds[ibin]) &&
        temp_hwEta <= l1ct::Scales::makeEta(associationNetworkEtaBounds[ibin + 1])) {
      resbin = ibin;
      break;
    }
  }

  // The following constants <22, 9> are defined by the quantisation of the Neural Network
  fPt_ = t.hwPt;
  fResBin_ = associationNetworkZ0ResBins[resbin];
  fMVA_ = t.hwQuality;
  fDz_ = t.hwZ0 - v.hwZ0;

  modelInput[0] = fPt_;           // Obj pT
  modelInput[1] = fMVA_;          // Obj track quality
  modelInput[2] = fResBin_ / 16;  // Obj z0 resolution bin (rescaled)
  modelInput[3] = fDz_;           // Obj delta z from the PV

  modelRef_->prepare_input(modelInput);
  modelRef_->predict();
  modelRef_->read_result(&classresult);

  score = (l1ct::nn_assoc_t)classresult;

  if (isDebugEnabled_) {
    LogDebug("NNVtxAssoc") << "\n ===== Vertex Association Output Score =====" << std::endl;
  }
  float NNOutput_exp = 1.0 / (1.0 + exp(-1.0 * (classresult.to_float())));
  if (isDebugEnabled_) {
    LogDebug("NNVtxAssoc") << "Score: " << classresult.to_float() << "exponentiated score " << NNOutput_exp
                           << std::endl;
  }
}

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

edm::ParameterSetDescription NNVtxAssoc::getParameterSetDescription() {
  edm::ParameterSetDescription description;
  description.add<double>("associationThreshold");
  description.add<std::string>("associationNetworkPath");
  description.add<std::vector<double>>("associationNetworkZ0binning");
  description.add<std::vector<double>>("associationNetworkEtaBounds");
  description.add<std::vector<double>>("associationNetworkZ0ResBins");
  return description;
}

void NNVtxAssoc::NNVtxAssocDebug() {
  log_ << "-- NNVtxAssocDebug --\n";
  log_ << "AssociationThreshold: " << this->associationThreshold_ << "\n";
  log_ << "z0_binning: ";
  for (auto i : this->z0_binning_)
    log_ << i << " ";
  log_ << "\n";
  log_ << "eta_bins: ";
  for (auto i : this->eta_bins_)
    log_ << i << " ";
  log_ << "\n";
  log_ << "res_bins: ";
  for (auto i : this->res_bins_)
    log_ << i << " ";
  log_ << "\n";
  edm::LogPrint("NNVtxAssoc") << log_.str();
}

#else

#include "NNVtx/L1TNNVtx_Assoc_Model_v0/NN/L1TNNVtx_Assoc_Model_v0.h"

void EmuNetworkSelector(const l1ct::TkObj& t, const l1ct::PVObjEmu& v, l1ct::nn_assoc_t& output_score) {
  int resbin = 0;
  int Nresbins = std::size(associationNetworkEtaBounds);
  l1ct::eta_t temp_hwEta = t.hwEta;
  if (temp_hwEta < 0)
    temp_hwEta = -temp_hwEta;
  for (int ibin = 0; ibin < Nresbins; ++ibin) {
    if (temp_hwEta > l1ct::Scales::makeEta(associationNetworkEtaBounds[ibin]) &&
        temp_hwEta <= l1ct::Scales::makeEta(associationNetworkEtaBounds[ibin + 1])) {
      resbin = ibin;
      break;
    }
  }
  // The following constants <22, 9> are defined by the quantisation of the Neural Network
  nn_inputtype fPt_ = t.hwPt;
  nn_inputtype fResBin_ = associationNetworkZ0ResBins[resbin];
  nn_inputtype fMVA_ = t.hwQuality;
  nn_inputtype fDz_ = t.hwZ0 - v.hwZ0;

  nn_inputtype association_input[N_NN_ASSOC_FEATURES];
  L1TNNVtx_Assoc_Model_v0::result_t nn_output_score[N_NN_ASSOC_OUTPUTS];

  association_input[0] = fPt_;           // Obj pT
  association_input[1] = fMVA_;          // Obj track quality
  association_input[2] = fResBin_ / 16;  // Obj z0 resolution bin (rescaled)
  association_input[3] = fDz_;           // Obj delta z from the PV

  L1TNNVtx_Assoc_Model_v0::NNvtx_assoc(association_input, nn_output_score);

  output_score = (l1ct::nn_assoc_t)nn_output_score[0];
}
#endif
