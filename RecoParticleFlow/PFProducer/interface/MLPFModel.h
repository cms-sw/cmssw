#ifndef RecoParticleFlow_PFProducer_interface_MLPFModel
#define RecoParticleFlow_PFProducer_interface_MLPFModel

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

namespace reco::mlpf {
  //The model takes the following number of features for each input PFElement
  static constexpr unsigned int NUM_ELEMENT_FEATURES = 15;

  //these are defined at model creation time and set the random LSH codebook size
  static constexpr int NUM_MAX_ELEMENTS_BATCH = 20000;
  static constexpr int LSH_BIN_SIZE = 100;

  //In CPU mode, we only want to evaluate each event separately
  static constexpr int BATCH_SIZE = 1;

  //The model has 12 outputs for each particle:
  // out[0-7]: particle classification logits
  // out[8]: regressed eta
  // out[9]: regressed phi
  // out[10]: regressed energy
  // out[11]: regressed charge logit
  static constexpr unsigned int NUM_OUTPUTS = 12;
  static constexpr unsigned int NUM_CLASS = 7;
  static constexpr unsigned int IDX_ETA = 8;
  static constexpr unsigned int IDX_PHI = 9;
  static constexpr unsigned int IDX_ENERGY = 10;
  static constexpr unsigned int IDX_CHARGE = 11;

  //index [0, N_pdgids) -> PDGID
  //this maps the absolute values of the predicted PDGIDs to an array of ascending indices
  static const std::vector<int> pdgid_encoding = {0, 1, 2, 11, 13, 22, 130, 211};

  //PFElement::type -> index [0, N_types)
  //this maps the type of the PFElement to an ascending index that is used by the model to distinguish between different elements
  static const std::map<int, int> elem_type_encoding = {
      {0, 0},
      {1, 1},
      {2, 2},
      {3, 3},
      {4, 4},
      {5, 5},
      {6, 6},
      {7, 7},
      {8, 8},
      {9, 9},
      {10, 10},
      {11, 11},
  };

  std::array<float, NUM_ELEMENT_FEATURES> getElementProperties(const reco::PFBlockElement& orig);
  float normalize(float in);

  int argMax(std::vector<float> const& vec);

  reco::PFCandidate makeCandidate(int pred_pid, int pred_charge, float pred_e, float pred_eta, float pred_phi);

  const std::vector<const reco::PFBlockElement*> getPFElements(const reco::PFBlockCollection& blocks);

  void setCandidateRefs(reco::PFCandidate& cand,
                        const std::vector<const reco::PFBlockElement*> elems,
                        size_t ielem_originator);
};  // namespace reco::mlpf

#endif