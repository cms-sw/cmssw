
/*

/////////////////////////////////////

// Displaced tau training features with multi-threading : Mykyta Shchedrolosiev , Pritam Palit, created on 01/09/2025  

/////////////////////////////////////

*/

#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <cmath>

#include <boost/math/constants/constants.hpp>

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "PhysicsTools/NanoAOD/interface/DisTauTagScaling.h"

void test_vector(std::vector<float>& values) {
  for (auto& value : values) {
    if (std::isnan(value)) {
      throw std::runtime_error("DisTauTag score output: NaN detected.");
    } else if (std::isinf(value)) {
      throw std::runtime_error("DisTauTag score output: Infinity detected.");
    } else if (!std::isfinite(value)) {
      throw std::runtime_error("DisTauTag score output: Non-standard value detected.");
    }
  }
}

class DisTauTag : public edm::global::EDProducer<> {
public:
  explicit DisTauTag(const edm::ParameterSet&);
  ~DisTauTag() override;

  template <typename Scalar>
  static Scalar getDeltaPhi(Scalar phi1, Scalar phi2);

  static void fill_zero(tensorflow::Tensor&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  template <typename FeatureT>
  float Scale(const Int_t, const Float_t, const bool) const;
  void saveInputs(const tensorflow::Tensor& tensor,
                  const std::string& block_name,
                  const edm::EventID& eventId,
                  size_t jetIndex) const;

  void initializeTensorFlow();

  const edm::FileInPath graphPath_;
  const edm::EDGetTokenT<pat::JetCollection> jets_token;
  const bool save_inputs_;
  const unsigned batch_size_;

  // TensorFlow resources for threading
  tensorflow::GraphDef* graphDef_;
  tensorflow::Session* session_;
};

DisTauTag::~DisTauTag() {
  if (session_) {
    tensorflow::closeSession(session_);
  }
  delete graphDef_;
}

DisTauTag::DisTauTag(const edm::ParameterSet& config)
    : graphPath_(config.getParameter<edm::FileInPath>("graphPath")),
      jets_token(consumes<pat::JetCollection>(config.getParameter<edm::InputTag>("jets"))),
      save_inputs_(config.getParameter<bool>("save_inputs")),
      batch_size_(config.getParameter<unsigned>("batchSize")),
      graphDef_(nullptr),
      session_(nullptr) {
  produces<edm::ValueMap<float>>("score0");
  produces<edm::ValueMap<float>>("score1");
  initializeTensorFlow();
}

void DisTauTag::initializeTensorFlow() {
  if (!session_) {
    const std::string graphFile = graphPath_.fullPath();
    graphDef_ = tensorflow::loadGraphDef(graphFile);
    session_ = tensorflow::createSession(graphDef_);
  }
}

template <typename Scalar>
Scalar DisTauTag::getDeltaPhi(Scalar phi1, Scalar phi2) {
  static constexpr Scalar pi = boost::math::constants::pi<Scalar>();
  Scalar dphi = phi1 - phi2;
  if (dphi > pi)
    dphi -= 2 * pi;
  else if (dphi <= -pi)
    dphi += 2 * pi;
  return dphi;
}

void DisTauTag::fill_zero(tensorflow::Tensor& tensor) {
  size_t size_ = 1;
  int num_dimensions = tensor.shape().dims();
  for (int ii_dim = 0; ii_dim < num_dimensions; ii_dim++)
    size_ = size_ * tensor.shape().dim_size(ii_dim);

  for (size_t ii = 0; ii < size_; ii++)
    tensor.flat<float>()(ii) = 0.0;
}

template <typename FeatureT>
float DisTauTag::Scale(const Int_t idx, const Float_t value, const bool inner) const {
  return std::clamp((value - FeatureT::mean.at(idx).at(inner)) / FeatureT::std.at(idx).at(inner),
                    FeatureT::lim_min.at(idx).at(inner),
                    FeatureT::lim_max.at(idx).at(inner));
}

void DisTauTag::saveInputs(const tensorflow::Tensor& tensor,
                           const std::string& block_name,
                           const edm::EventID& eventId,
                           size_t jetIndex) const {
  const int tau_n = tensor.shape().dim_size(0);
  const int pf_n = tensor.shape().dim_size(1);
  const int ftr_n = tensor.shape().dim_size(2);

  std::string json_file_name = "distag_" + std::to_string(eventId.run()) + "_" +
                               std::to_string(eventId.luminosityBlock()) + "_" + std::to_string(eventId.event()) + "_" +
                               "jet_" + std::to_string(jetIndex) + ".json";

  std::ofstream json_file(json_file_name.data());
  if (!json_file.is_open()) {
    edm::LogError("DisTauTag") << "Failed to open file for saving inputs: " << json_file_name;
    return;
  }

  json_file << "{\"" << block_name << "\":[";
  for (int tau_idx = 0; tau_idx < tau_n; tau_idx++) {
    json_file << "[";
    for (int pf_idx = 0; pf_idx < pf_n; pf_idx++) {
      json_file << "[";
      for (int ftr_idx = 0; ftr_idx < ftr_n; ftr_idx++) {
        json_file << std::setprecision(7) << std::fixed << tensor.tensor<float, 3>()(tau_idx, pf_idx, ftr_idx);
        if (ftr_idx < ftr_n - 1)
          json_file << ", ";
      }
      json_file << "]";
      if (pf_idx < pf_n - 1)
        json_file << ", ";
    }
    json_file << "]";
    if (tau_idx < tau_n - 1)
      json_file << ", ";
  }
  json_file << "]}";
}

void DisTauTag::produce(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  auto jets = event.getHandle(jets_token);

  const size_t jets_size = jets->size();

  std::vector<float> v_score0(jets_size, -9);
  std::vector<float> v_score1(jets_size, -9);

  // Processing jets in batches
  const size_t n_batches = (jets_size + batch_size_ - 1) / batch_size_;

  for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
    const size_t jet_start = batch_idx * batch_size_;
    const size_t jet_end = std::min((batch_idx + 1) * batch_size_, jets_size);
    const size_t current_batch_size = jet_end - jet_start;

    // batched inputs
    tensorflow::Tensor input_1(tensorflow::DT_FLOAT,
                               {static_cast<int>(current_batch_size), setupSizes::nSeq_PfCand, setupSizes::n_PfCand});
    tensorflow::Tensor input_2(
        tensorflow::DT_FLOAT,
        {static_cast<int>(current_batch_size), setupSizes::nSeq_PfCand, setupSizes::n_PfCandCategorical});
    fill_zero(input_1);
    fill_zero(input_2);

    // Filling the batches
    for (size_t batch_jet_idx = 0; batch_jet_idx < current_batch_size; ++batch_jet_idx) {
      const size_t jetIndex = jet_start + batch_jet_idx;
      const auto& jet = jets->at(jetIndex);
      const auto& jet_p4 = jet.polarP4();

      // Get jet daughters and sort by pt
      const size_t nDaughters = jet.numberOfDaughters();
      std::vector<size_t> indices(nDaughters);
      std::iota(indices.begin(), indices.end(), 0);
      std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        const auto& daughter_1 = jet.daughterPtr(a);
        const auto& daughter_2 = jet.daughterPtr(b);
        return daughter_1->polarP4().pt() > daughter_2->polarP4().pt();
      });

      size_t daughter_idx = 0;
      size_t tensor_idx = 0;

      while (tensor_idx < static_cast<size_t>(setupSizes::nSeq_PfCand) && daughter_idx < nDaughters) {
        const auto& jet_daughter = jet.daughterPtr(indices.at(daughter_idx));
        const auto daughter = dynamic_cast<const pat::PackedCandidate*>(jet_daughter.get());
        ++daughter_idx;

        if (!daughter)
          continue;

        auto getVecRef = [&](tensorflow::Tensor& tensor, auto _fe, Float_t value) {
          const int _feature_idx = static_cast<int>(_fe);
          if (_feature_idx < 0)
            return;
          tensor.tensor<float, 3>()(batch_jet_idx, tensor_idx, _feature_idx) =
              Scale<typename FeaturesHelper<decltype(_fe)>::scaler_type>(_feature_idx, value, false);
        };

        {  // General features
          typedef PfCand_Features Br;
          getVecRef(input_1, Br::pfCand_valid, 1.0);
          getVecRef(input_1, Br::pfCand_pt, static_cast<Float_t>(daughter->polarP4().pt()));
          getVecRef(input_1, Br::pfCand_eta, static_cast<Float_t>(daughter->polarP4().eta()));
          getVecRef(input_1, Br::pfCand_phi, static_cast<Float_t>(daughter->polarP4().phi()));
          getVecRef(input_1, Br::pfCand_mass, static_cast<Float_t>(daughter->polarP4().mass()));
          getVecRef(input_1, Br::pfCand_charge, static_cast<Int_t>(daughter->charge()));
          getVecRef(input_1, Br::pfCand_puppiWeight, static_cast<Float_t>(daughter->puppiWeight()));
          getVecRef(input_1, Br::pfCand_puppiWeightNoLep, static_cast<Float_t>(daughter->puppiWeightNoLep()));
          getVecRef(input_1, Br::pfCand_lostInnerHits, static_cast<Int_t>(daughter->lostInnerHits()));
          getVecRef(input_1, Br::pfCand_nPixelHits, static_cast<Int_t>(daughter->numberOfPixelHits()));
          getVecRef(input_1, Br::pfCand_nHits, static_cast<Int_t>(daughter->numberOfHits()));
          getVecRef(input_1, Br::pfCand_caloFraction, static_cast<Float_t>(daughter->caloFraction()));
          getVecRef(input_1, Br::pfCand_hcalFraction, static_cast<Float_t>(daughter->hcalFraction()));
          getVecRef(input_1, Br::pfCand_rawCaloFraction, static_cast<Float_t>(daughter->rawCaloFraction()));
          getVecRef(input_1, Br::pfCand_rawHcalFraction, static_cast<Float_t>(daughter->rawHcalFraction()));

          getVecRef(input_1, Br::pfCand_hasTrackDetails, static_cast<Int_t>(daughter->hasTrackDetails()));

          if (daughter->hasTrackDetails()) {
            if (std::isfinite(daughter->dz()))
              getVecRef(input_1, Br::pfCand_dz, static_cast<Float_t>(daughter->dz()));
            if (std::isfinite(daughter->dzError()))
              getVecRef(input_1, Br::pfCand_dz_error, static_cast<Float_t>(daughter->dzError()));
            if (std::isfinite(daughter->dxyError()))
              getVecRef(input_1, Br::pfCand_dxy_error, static_cast<Float_t>(daughter->dxyError()));

            getVecRef(input_1, Br::pfCand_dxy, static_cast<Float_t>(daughter->dxy()));
            getVecRef(input_1, Br::pfCand_track_chi2, static_cast<Float_t>(daughter->bestTrack()->chi2()));
            getVecRef(input_1, Br::pfCand_track_ndof, static_cast<Float_t>(daughter->bestTrack()->ndof()));
          }

          Float_t jet_eta = jet_p4.eta();
          Float_t jet_phi = jet_p4.phi();
          getVecRef(input_1, PfCand_Features::pfCand_deta, static_cast<Float_t>(daughter->polarP4().eta()) - jet_eta);
          getVecRef(input_1,
                    PfCand_Features::pfCand_dphi,
                    getDeltaPhi<Float_t>(static_cast<Float_t>(daughter->polarP4().phi()), jet_phi));
        }

        {  // Categorical features
          typedef PfCandCategorical_Features Br;
          getVecRef(
              input_2, Br::pfCand_particleType, static_cast<Int_t>(TranslatePdgIdToPFParticleType(daughter->pdgId())));
          getVecRef(input_2, Br::pfCand_pvAssociationQuality, static_cast<Int_t>(daughter->pvAssociationQuality()));
          getVecRef(input_2, Br::pfCand_fromPV, static_cast<Int_t>(daughter->fromPV()));
        }

        ++tensor_idx;
      }
    }

    // Running inference on batch
    std::vector<tensorflow::Tensor> outputs;
    { tensorflow::run(session_, {{"input_1", input_1}, {"input_2", input_2}}, {"final_out"}, &outputs); }

    // Storing results
    for (size_t batch_jet_idx = 0; batch_jet_idx < current_batch_size; ++batch_jet_idx) {
      const size_t jetIndex = jet_start + batch_jet_idx;
      v_score0[jetIndex] = outputs[0].matrix<float>()(batch_jet_idx, 0);
      v_score1[jetIndex] = outputs[0].matrix<float>()(batch_jet_idx, 1);
    }

    // Save inputs if requested (for debugging)
    if (save_inputs_) {
      for (size_t batch_jet_idx = 0; batch_jet_idx < current_batch_size; ++batch_jet_idx) {
        const size_t jetIndex = jet_start + batch_jet_idx;

        // Extract single jet inputs from the batch
        tensorflow::Tensor single_input_1(tensorflow::DT_FLOAT, {1, setupSizes::nSeq_PfCand, setupSizes::n_PfCand});
        tensorflow::Tensor single_input_2(tensorflow::DT_FLOAT,
                                          {1, setupSizes::nSeq_PfCand, setupSizes::n_PfCandCategorical});

        for (size_t i = 0; i < static_cast<size_t>(setupSizes::nSeq_PfCand); ++i) {
          for (size_t j = 0; j < static_cast<size_t>(setupSizes::n_PfCand); ++j) {
            single_input_1.tensor<float, 3>()(0, i, j) = input_1.tensor<float, 3>()(batch_jet_idx, i, j);
          }
          for (size_t j = 0; j < static_cast<size_t>(setupSizes::n_PfCandCategorical); ++j) {
            single_input_2.tensor<float, 3>()(0, i, j) = input_2.tensor<float, 3>()(batch_jet_idx, i, j);
          }
        }

        saveInputs(single_input_1, "PfCand", event.id(), jetIndex);
        saveInputs(single_input_2, "PfCandCategorical", event.id(), jetIndex);
      }
    }
  }

  test_vector(v_score0);
  test_vector(v_score1);

  std::unique_ptr<edm::ValueMap<float>> vm_score0(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_score0(*vm_score0);
  filler_score0.insert(jets, v_score0.begin(), v_score0.end());
  filler_score0.fill();
  event.put(std::move(vm_score0), "score0");

  std::unique_ptr<edm::ValueMap<float>> vm_score1(new edm::ValueMap<float>());
  edm::ValueMap<float>::Filler filler_score1(*vm_score1);
  filler_score1.insert(jets, v_score1.begin(), v_score1.end());
  filler_score1.fill();
  event.put(std::move(vm_score1), "score1");
}

DEFINE_FWK_MODULE(DisTauTag);
