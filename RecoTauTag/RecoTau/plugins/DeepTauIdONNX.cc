/*                                                                                                                                               
 * \class DeepTauIdONNX                                                                                                                          
 *                                                                                                                                               
 * Tau identification using Deep NN (CNN) in ONNX Runtime.                                                                                       
 *                                                                                                                                               
 * \author Pritam Palit, Carnegie Mellon University                                                                                              
 *
 */

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "RecoTauTag/RecoTau/interface/DeepTauIdBaseONNX.h"

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <fstream>

namespace deep_tau {

  class DeepTauCacheONNX {
  public:
    DeepTauCacheONNX(const std::map<std::string, std::string>& graph_names) {
      for (const auto& entry : graph_names)
        sessions_[entry.first] = std::make_unique<cms::Ort::ONNXRuntime>(entry.second);
    }

    const cms::Ort::ONNXRuntime& getSession(const std::string& name = "") const { return *sessions_.at(name); }

  private:
    std::map<std::string, std::unique_ptr<cms::Ort::ONNXRuntime>> sessions_;
  };

}  // namespace deep_tau

class DeepTauIdONNXWrapper : public edm::stream::EDProducer<edm::GlobalCache<deep_tau::DeepTauCacheONNX>> {
public:
  explicit DeepTauIdONNXWrapper(const edm::ParameterSet&) {}
};

class DeepTauIdONNX : public DeepTauIdBaseONNX<DeepTauIdONNXWrapper> {
public:
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    fillDescriptionsHelper(desc);
    desc.add<std::vector<std::string>>("graph_file",
                                       {"core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.onnx"});
    //desc.add<bool>("mem_mapped", false);
    descriptions.add("DeepTauONNX", desc);
  }

  explicit DeepTauIdONNX(const edm::ParameterSet& cfg, const deep_tau::DeepTauCacheONNX* cache)
      : DeepTauIdBaseONNX<DeepTauIdONNXWrapper>(cfg), cache_(cache) {
    if (version_ != 2)
      throw cms::Exception("DeepTauIdONNX") << "version " << version_ << " is not supported.";

    using namespace dnn_inputs_v2;

    if (sub_version_ == 1) {
      tauBlockSize_ = TauBlockInputs::NumberOfInputs;
    } else if ((sub_version_ == 5) || ((sub_version_ == 0) && (year_ == 20161718))) {
      tauBlockSize_ =
          static_cast<int>(TauBlockInputs::NumberOfInputs) - static_cast<int>(TauBlockInputs::varsToDrop.size());
    } else {
      throw cms::Exception("DeepTauIdONNX") << "sub_version " << sub_version_ << " is not supported.";
    }

    tauBlock_.resize(tauBlockSize_, 0.f);

    for (size_t n = 0; n < 2; ++n) {
      const bool is_inner = (n == 0);
      const int n_cells = is_inner ? number_of_inner_cell : number_of_outer_cell;
      const int max_cells = n_cells * n_cells;

      // get zero placeholder from network with 1 zero cell
      eGammaBlock_[is_inner].assign(EgammaBlockInputs::NumberOfInputs, 0.f);
      muonBlock_[is_inner].assign(MuonBlockInputs::NumberOfInputs, 0.f);
      hadronBlock_[is_inner].assign(HadronBlockInputs::NumberOfInputs, 0.f);
      zeroConvFeatures_[is_inner] = getPartialPredictions(is_inner, 1);

      // now assign full size for event processing
      eGammaBlock_[is_inner].assign(max_cells * EgammaBlockInputs::NumberOfInputs, 0.f);
      muonBlock_[is_inner].assign(max_cells * MuonBlockInputs::NumberOfInputs, 0.f);
      hadronBlock_[is_inner].assign(max_cells * HadronBlockInputs::NumberOfInputs, 0.f);
      convFeatures_[is_inner].assign(n_cells * n_cells * number_of_conv_features, 0.f);
    }
  }

  static std::unique_ptr<deep_tau::DeepTauCacheONNX> initializeGlobalCache(const edm::ParameterSet& cfg) {
    const auto graph_name_vector = cfg.getParameter<std::vector<std::string>>("graph_file");
    std::map<std::string, std::string> graph_names;
    for (const auto& entry : graph_name_vector) {
      const size_t sep = entry.find(':');
      std::string entry_name, graph_file;
      if (sep != std::string::npos) {
        entry_name = entry.substr(0, sep);
        graph_file = entry.substr(sep + 1);
      } else {
        entry_name = "";
        graph_file = entry;
      }
      graph_file = edm::FileInPath(graph_file).fullPath();
      if (graph_names.count(entry_name))
        throw cms::Exception("DeepTauCacheONNX") << "Duplicated graph entry name: '" << entry_name << "'";
      graph_names[entry_name] = graph_file;
    }
    return std::make_unique<deep_tau::DeepTauCacheONNX>(graph_names);
  }

  static void globalEndJob(const deep_tau::DeepTauCacheONNX*) {}

  void produce(edm::Event& event, const edm::EventSetup& es) override {
    edm::Handle<TauCollection> taus;
    event.getByToken(tausToken_, taus);

    if (taus->empty()) {
      const std::vector<std::vector<float>> emptyPred;
      createOutputs(event, emptyPred, taus);
      return;
    }

    loadPrediscriminants(event, taus);
    const auto pred = getPredictions(event, taus);
    createOutputs(event, pred, taus);
  }

private:
  void setCellConvFeatures(bool is_inner, int eta_index, int phi_index, const float* features_ptr) {
    using namespace dnn_inputs_v2;
    const int n_cells = is_inner ? number_of_inner_cell : number_of_outer_cell;
    const int base = (eta_index * n_cells + phi_index) * number_of_conv_features;
    for (int f = 0; f < number_of_conv_features; ++f)
      convFeatures_[is_inner][base + f] = features_ptr[f];
  }

  std::vector<float> getPartialPredictions(bool is_inner, size_t n_valid_cells) {
    using namespace dnn_inputs_v2;
    const std::string sess_name = is_inner ? "inner" : "outer";

    std::vector<std::string> input_names;
    std::vector<std::vector<float>> input_values;
    std::vector<std::vector<int64_t>> input_shapes;

    if (is_inner) {
      input_names = {"input_inner_egamma", "input_inner_muon", "input_inner_hadrons"};
    } else {
      input_names = {"input_outer_egamma", "input_outer_muon", "input_outer_hadrons"};
    }

    const int64_t batch = static_cast<int64_t>(n_valid_cells);

    input_values.push_back(
        std::vector<float>(eGammaBlock_[is_inner].begin(),
                           eGammaBlock_[is_inner].begin() + n_valid_cells * EgammaBlockInputs::NumberOfInputs));
    input_shapes.push_back({batch, 1, 1, EgammaBlockInputs::NumberOfInputs});

    input_values.push_back(std::vector<float>(
        muonBlock_[is_inner].begin(), muonBlock_[is_inner].begin() + n_valid_cells * MuonBlockInputs::NumberOfInputs));
    input_shapes.push_back({batch, 1, 1, MuonBlockInputs::NumberOfInputs});

    input_values.push_back(
        std::vector<float>(hadronBlock_[is_inner].begin(),
                           hadronBlock_[is_inner].begin() + n_valid_cells * HadronBlockInputs::NumberOfInputs));
    input_shapes.push_back({batch, 1, 1, HadronBlockInputs::NumberOfInputs});

    const std::vector<std::string> output_names = {is_inner ? "inner_all_dropout_4/Identity"
                                                            : "outer_all_dropout_4/Identity"};

    auto outputs = cache_->getSession(sess_name).run(input_names, input_values, input_shapes, output_names);

    return outputs.at(0);
  }

  template <typename CandidateCastType, typename TauCastType>
  void createConvFeatures(const TauCastType& tau,
                          const size_t tau_index,
                          const edm::RefToBase<reco::BaseTau> tau_ref,
                          const reco::Vertex& pv,
                          double rho,
                          const std::vector<pat::Electron>* electrons,
                          const std::vector<pat::Muon>* muons,
                          const edm::View<reco::Candidate>& pfCands,
                          const CellGrid& grid,
                          TauFunc tau_funcs,
                          bool is_inner) {
    using namespace dnn_inputs_v2;

    const size_t n_valid = grid.num_valid_cells();

    if (n_valid > 0) {
      eGammaBlock_[is_inner].assign(n_valid * EgammaBlockInputs::NumberOfInputs, 0.f);
      muonBlock_[is_inner].assign(n_valid * MuonBlockInputs::NumberOfInputs, 0.f);
      hadronBlock_[is_inner].assign(n_valid * HadronBlockInputs::NumberOfInputs, 0.f);

      unsigned idx = 0;
      for (int eta = -grid.maxEtaIndex(); eta <= grid.maxEtaIndex(); ++eta) {
        for (int phi = -grid.maxPhiIndex(); phi <= grid.maxPhiIndex(); ++phi) {
          const CellIndex cell_index{eta, phi};
          if (grid.find(cell_index) == grid.end())
            continue;

          const Cell& cell = grid.find(cell_index)->second;

          auto eg_it = eGammaBlock_[is_inner].begin() + idx * EgammaBlockInputs::NumberOfInputs;
          auto mu_it = muonBlock_[is_inner].begin() + idx * MuonBlockInputs::NumberOfInputs;
          auto had_it = hadronBlock_[is_inner].begin() + idx * HadronBlockInputs::NumberOfInputs;

          createEgammaBlockInputs<CandidateCastType>(
              idx, tau, tau_index, tau_ref, pv, rho, electrons, pfCands, cell, tau_funcs, is_inner, eg_it);
          createMuonBlockInputs<CandidateCastType>(
              idx, tau, tau_index, tau_ref, pv, rho, muons, pfCands, cell, tau_funcs, is_inner, mu_it);
          createHadronsBlockInputs<CandidateCastType>(
              idx, tau, tau_index, tau_ref, pv, rho, pfCands, cell, tau_funcs, is_inner, had_it);
          ++idx;
        }
      }

      const std::vector<float> partial = getPartialPredictions(is_inner, n_valid);

      unsigned idx2 = 0;
      for (int eta = -grid.maxEtaIndex(); eta <= grid.maxEtaIndex(); ++eta) {
        for (int phi = -grid.maxPhiIndex(); phi <= grid.maxPhiIndex(); ++phi) {
          const CellIndex cell_index{eta, phi};
          const int eta_idx = grid.getEtaTensorIndex(cell_index);
          const int phi_idx = grid.getPhiTensorIndex(cell_index);
          if (grid.find(cell_index) != grid.end()) {
            setCellConvFeatures(is_inner, eta_idx, phi_idx, &partial[idx2 * number_of_conv_features]);
            ++idx2;
          } else {
            setCellConvFeatures(is_inner, eta_idx, phi_idx, zeroConvFeatures_[is_inner].data());
          }
        }
      }

    } else {
      const int n_cells = is_inner ? number_of_inner_cell : number_of_outer_cell;
      convFeatures_[is_inner].assign(n_cells * n_cells * number_of_conv_features, 0.f);
    }
  }

  template <typename CandidateCastType, typename TauCastType>
  std::vector<float> getPredictionsV2(TauCollection::const_reference& tau,
                                      const size_t tau_index,
                                      const edm::RefToBase<reco::BaseTau> tau_ref,
                                      const std::vector<pat::Electron>* electrons,
                                      const std::vector<pat::Muon>* muons,
                                      const edm::View<reco::Candidate>& pfCands,
                                      const reco::Vertex& pv,
                                      double rho,
                                      const edm::EventNumber_t& eventnr,
                                      TauFunc tau_funcs) {
    using namespace dnn_inputs_v2;

    CellGrid inner_grid(number_of_inner_cell, number_of_inner_cell, 0.02, 0.02, disable_CellIndex_workaround_);
    CellGrid outer_grid(number_of_outer_cell, number_of_outer_cell, 0.05, 0.05, disable_CellIndex_workaround_);

    fillGrids(dynamic_cast<const TauCastType&>(tau), *electrons, inner_grid, outer_grid);
    fillGrids(dynamic_cast<const TauCastType&>(tau), *muons, inner_grid, outer_grid);
    fillGrids(dynamic_cast<const TauCastType&>(tau), pfCands, inner_grid, outer_grid);

    std::fill(tauBlock_.begin(), tauBlock_.end(), 0.f);
    auto tau_it = tauBlock_.begin();
    createTauBlockInputs<CandidateCastType>(
        dynamic_cast<const TauCastType&>(tau), tau_index, tau_ref, pv, rho, tau_funcs, tau_it);

    createConvFeatures<CandidateCastType>(dynamic_cast<const TauCastType&>(tau),
                                          tau_index,
                                          tau_ref,
                                          pv,
                                          rho,
                                          electrons,
                                          muons,
                                          pfCands,
                                          inner_grid,
                                          tau_funcs,
                                          true);
    createConvFeatures<CandidateCastType>(dynamic_cast<const TauCastType&>(tau),
                                          tau_index,
                                          tau_ref,
                                          pv,
                                          rho,
                                          electrons,
                                          muons,
                                          pfCands,
                                          outer_grid,
                                          tau_funcs,
                                          false);

    using namespace dnn_inputs_v2;
    const int64_t ni = number_of_inner_cell;
    const int64_t no = number_of_outer_cell;
    const int64_t nf = number_of_conv_features;

    std::vector<std::string> input_names = {"input_tau", "input_inner", "input_outer"};
    std::vector<std::vector<float>> input_values = {tauBlock_, convFeatures_[true], convFeatures_[false]};
    std::vector<std::vector<int64_t>> input_shapes = {{1, tauBlockSize_}, {1, ni, ni, nf}, {1, no, no, nf}};
    const std::vector<std::string> output_names = {"main_output/Softmax"};

    auto outputs = cache_->getSession("core").run(input_names, input_values, input_shapes, output_names);

    std::vector<float> pred = outputs.at(0);

    if (debug_level >= 1) {
      static const std::array<std::string, 4> labels = {"e", "mu", "tau", "jet"};
      std::cout << "output = { ";
      for (int k = 0; k < deep_tau::NumberOfOutputs; ++k) {
        if (k > 0)
          std::cout << ", ";
        std::cout << labels[k] << " = " << pred[k];
      }
      std::cout << " }" << std::endl;
    }

    if (save_inputs_) {
      std::string json_file_name =
          "DeepTauIdONNX_" + std::to_string(eventnr) + "_" + std::to_string(tau_index) + ".json";
      std::ofstream jf(json_file_name);
      jf << "{";

      // tau block
      jf << "\"input_tau\": [";
      for (int i = 0; i < tauBlockSize_; ++i) {
        if (i)
          jf << ", ";
        jf << tauBlock_[i];
      }
      jf << "]";

      // helper lambda to dump a grid block
      auto dumpGrid =
          [&](const std::string& block_name, const std::vector<float>& buf, int n_inputs, const CellGrid& grid) {
            jf << ", \"" << block_name << "\": [";
            const int n_eta = grid.maxEtaIndex();
            const int n_phi = grid.maxPhiIndex();
            int idx = 0;
            for (int eta = -n_eta; eta <= n_eta; ++eta) {
              if (eta != -n_eta)
                jf << ", ";
              jf << "[";
              for (int phi = -n_phi; phi <= n_phi; ++phi) {
                if (phi != -n_phi)
                  jf << ", ";
                jf << "[";
                const CellIndex ci{eta, phi};
                const auto it = grid.find(ci);
                for (int f = 0; f < n_inputs; ++f) {
                  if (f)
                    jf << ", ";
                  float v = 0.f;
                  if (it != grid.end())
                    v = buf[idx * n_inputs + f];
                  jf << v;
                }
                if (it != grid.end())
                  ++idx;
                jf << "]";
              }
              jf << "]";
            }
            jf << "]";
          };

      dumpGrid("input_inner_egamma", eGammaBlock_[true], dnn_inputs_v2::EgammaBlockInputs::NumberOfInputs, inner_grid);
      dumpGrid("input_inner_muon", muonBlock_[true], dnn_inputs_v2::MuonBlockInputs::NumberOfInputs, inner_grid);
      dumpGrid("input_inner_hadrons", hadronBlock_[true], dnn_inputs_v2::HadronBlockInputs::NumberOfInputs, inner_grid);
      dumpGrid("input_outer_egamma", eGammaBlock_[false], dnn_inputs_v2::EgammaBlockInputs::NumberOfInputs, outer_grid);
      dumpGrid("input_outer_muon", muonBlock_[false], dnn_inputs_v2::MuonBlockInputs::NumberOfInputs, outer_grid);
      dumpGrid(
          "input_outer_hadrons", hadronBlock_[false], dnn_inputs_v2::HadronBlockInputs::NumberOfInputs, outer_grid);

      jf << "}";
    }

    return pred;
  }

  std::vector<std::vector<float>> getPredictions(edm::Event& event, edm::Handle<TauCollection> taus) {
    const std::vector<pat::Electron> electron_collection_default;
    const std::vector<pat::Muon> muon_collection_default;
    const reco::TauDiscriminatorContainer basicTauDiscriminators_default;
    const reco::TauDiscriminatorContainer basicTauDiscriminatorsdR03_default;
    const edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>
        pfTauTransverseImpactParameters_default;

    const std::vector<pat::Electron>* electron_collection;
    const std::vector<pat::Muon>* muon_collection;
    const reco::TauDiscriminatorContainer* basicTauDiscriminators;
    const reco::TauDiscriminatorContainer* basicTauDiscriminatorsdR03;
    const edm::AssociationVector<reco::PFTauRefProd, std::vector<reco::PFTauTransverseImpactParameterRef>>*
        pfTauTransverseImpactParameters;

    if (!is_online_) {
      electron_collection = &event.get(electrons_token_);
      muon_collection = &event.get(muons_token_);
      pfTauTransverseImpactParameters = &pfTauTransverseImpactParameters_default;
      basicTauDiscriminators = &basicTauDiscriminators_default;
      basicTauDiscriminatorsdR03 = &basicTauDiscriminatorsdR03_default;
    } else {
      electron_collection = &electron_collection_default;
      muon_collection = &muon_collection_default;
      pfTauTransverseImpactParameters = &event.get(pfTauTransverseImpactParameters_token_);
      basicTauDiscriminators = &event.get(basicTauDiscriminators_inputToken_);
      basicTauDiscriminatorsdR03 = &event.get(basicTauDiscriminatorsdR03_inputToken_);

      if (!discrIndicesMapped_) {
        basicDiscrIndexMap_ =
            matchDiscriminatorIndices(event, basicTauDiscriminators_inputToken_, requiredBasicDiscriminators_);
        basicDiscrdR03IndexMap_ =
            matchDiscriminatorIndices(event, basicTauDiscriminatorsdR03_inputToken_, requiredBasicDiscriminatorsdR03_);
        discrIndicesMapped_ = true;
      }
    }

    TauFunc tauIDs = {basicTauDiscriminators,
                      basicTauDiscriminatorsdR03,
                      pfTauTransverseImpactParameters,
                      basicDiscrIndexMap_,
                      basicDiscrdR03IndexMap_};

    edm::Handle<edm::View<reco::Candidate>> pfCands;
    event.getByToken(pfcandToken_, pfCands);

    edm::Handle<reco::VertexCollection> vertices;
    event.getByToken(vtxToken_, vertices);

    edm::Handle<double> rho;
    event.getByToken(rho_token_, rho);

    const auto& eventnr = event.id().event();

    std::vector<std::vector<float>> predictions(taus->size(), std::vector<float>(deep_tau::NumberOfOutputs, 0.f));

    for (size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
      const edm::RefToBase<reco::BaseTau> tauRef = taus->refAt(tau_index);

      bool passesPrediscriminants;
      if (is_online_) {
        passesPrediscriminants = tauIDs.passPrediscriminants<std::vector<TauDiscInfo<reco::PFTauDiscriminator>>>(
            recoPrediscriminants_, andPrediscriminants_, tauRef);
      } else {
        passesPrediscriminants = tauIDs.passPrediscriminants<std::vector<TauDiscInfo<pat::PATTauDiscriminator>>>(
            patPrediscriminants_, andPrediscriminants_, tauRef);
      }

      if (passesPrediscriminants) {
        std::vector<float> pred;
        if (is_online_) {
          pred = getPredictionsV2<reco::PFCandidate, reco::PFTau>(taus->at(tau_index),
                                                                  tau_index,
                                                                  tauRef,
                                                                  electron_collection,
                                                                  muon_collection,
                                                                  *pfCands,
                                                                  vertices->at(0),
                                                                  *rho,
                                                                  eventnr,
                                                                  tauIDs);
        } else {
          pred = getPredictionsV2<pat::PackedCandidate, pat::Tau>(taus->at(tau_index),
                                                                  tau_index,
                                                                  tauRef,
                                                                  electron_collection,
                                                                  muon_collection,
                                                                  *pfCands,
                                                                  vertices->at(0),
                                                                  *rho,
                                                                  eventnr,
                                                                  tauIDs);
        }

        for (int k = 0; k < deep_tau::NumberOfOutputs; ++k) {
          const float v = pred[k];
          if (!(v >= 0.f && v <= 1.f))
            throw cms::Exception("DeepTauIdONNX")
                << "invalid prediction = " << v << " for tau_index = " << tau_index << ", pred_index = " << k;
          predictions[tau_index][k] = v;
        }
      } else {
        for (int k = 0; k < deep_tau::NumberOfOutputs; ++k)
          predictions[tau_index][k] = (k == 2) ? -1.f : 2.f;
      }
    }
    return predictions;
  }

  const deep_tau::DeepTauCacheONNX* cache_;

  int tauBlockSize_ = 0;
  std::vector<float> tauBlock_;

  std::array<std::vector<float>, 2> eGammaBlock_;
  std::array<std::vector<float>, 2> muonBlock_;
  std::array<std::vector<float>, 2> hadronBlock_;
  std::array<std::vector<float>, 2> convFeatures_;
  std::array<std::vector<float>, 2> zeroConvFeatures_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeepTauIdONNX);
