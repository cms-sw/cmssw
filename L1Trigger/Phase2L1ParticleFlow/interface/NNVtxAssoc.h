#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_NNVtxAssoc_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_NNVtxAssoc_H

// Package:      L1Trigger/Phase2L1ParticleFlow
// Class:        NNVtxAssoc
// Description:  Designed to run the track to vertex associations created by the E2E NNVtx.
//               TTTrackNetworkSelector either accepts or rejects that a PF object's (t) track is associated to a vertex (v).
// Authors:      Kai Hong Law and Benjamin Radburn-Smith
// Created:      February 2025

#include <string>
#include "DataFormats/L1TParticleFlow/interface/layer1_emulator.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm
namespace tensorflow {
  class Session;
}  // namespace tensorflow
using namespace l1ct;

class NNVtxAssoc {
public:
  NNVtxAssoc(std::string AssociationGraphPath,
             const double AssociationThreshold,
             const std::vector<double>& AssociationNetworkZ0binning,
             const std::vector<double>& AssociationNetworkEtaBounds,
             const std::vector<double>& AssociationNetworkZ0ResBins);

  void NNVtxAssocDebug();
  static edm::ParameterSetDescription getParameterSetDescription();

  template <typename T>
  bool TTTrackNetworkSelector(const PFRegionEmu& region, const T& t, const l1ct::PVObjEmu& v);

private:
  tensorflow::Session* associationSesh_;
  double associationThreshold_;
  std::vector<double> z0_binning_;
  std::vector<double> eta_bins_;
  std::vector<double> res_bins_;
  std::stringstream log_;
};
#endif
