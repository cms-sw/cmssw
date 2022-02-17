#ifndef RecoParticleFlow_PFProducer_interface_MLPFModel
#define RecoParticleFlow_PFProducer_interface_MLPFModel

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

namespace reco::mlpf {
  //The model takes the following number of features for each input PFElement
  static constexpr unsigned int NUM_ELEMENT_FEATURES = 25;
  static constexpr unsigned int NUM_OUTPUT_FEATURES = 14;

  //these are defined at model creation time and set the random LSH codebook size
  static constexpr int LSH_BIN_SIZE = 64;
  static constexpr int NUM_MAX_ELEMENTS_BATCH = 200 * LSH_BIN_SIZE;

  //In CPU mode, we want to evaluate each event separately
  static constexpr int BATCH_SIZE = 1;

  //The model has 14 outputs for each particle:
  // out[0-7]: particle classification logits for each pdgId
  // out[8]: regressed charge
  // out[9]: regressed pt
  // out[10]: regressed eta
  // out[11]: regressed sin phi
  // out[12]: regressed cos phi
  // out[13]: regressed energy
  static constexpr unsigned int IDX_CLASS = 7;

  static constexpr unsigned int IDX_CHARGE = 8;

  static constexpr unsigned int IDX_PT = 9;
  static constexpr unsigned int IDX_ETA = 10;
  static constexpr unsigned int IDX_SIN_PHI = 11;
  static constexpr unsigned int IDX_COS_PHI = 12;
  static constexpr unsigned int IDX_ENERGY = 13;

  //for consistency with the baseline PFAlgo
  static constexpr float PI_MASS = 0.13957;

  //index [0, N_pdgids) -> PDGID
  //this maps the absolute values of the predicted PDGIDs to an array of ascending indices
  static const std::vector<int> pdgid_encoding = {0, 211, 130, 1, 2, 22, 11, 13};

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

  reco::PFCandidate makeCandidate(int pred_pid,
                                  int pred_charge,
                                  float pred_pt,
                                  float pred_eta,
                                  float pred_sin_phi,
                                  float pred_cos_phi,
                                  float pred_e);

  const std::vector<const reco::PFBlockElement*> getPFElements(const reco::PFBlockCollection& blocks);

  void setCandidateRefs(reco::PFCandidate& cand,
                        const std::vector<const reco::PFBlockElement*> elems,
                        size_t ielem_originator);
};  // namespace reco::mlpf

#endif
