/*
 * \class DeepTauId
 *
 * Tau identification using Deep NN.
 *
 * \author Konstantin Androsov, INFN Pisa
 *         Christian Veelken, Tallinn
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "RecoTauTag/RecoTau/interface/DeepTauIdBase.h"
#include "FWCore/Utilities/interface/isFinite.h"

namespace deep_tau {
  class DeepTauCache {
  public:
    using GraphPtr = std::shared_ptr<tensorflow::GraphDef>;

    DeepTauCache(const std::map<std::string, std::string>& graph_names, bool mem_mapped) {
      for (const auto& graph_entry : graph_names) {
        // Backend : to be parametrized from the python config
        tensorflow::Options options{tensorflow::Backend::cpu};

        const std::string& entry_name = graph_entry.first;
        const std::string& graph_file = graph_entry.second;
        if (mem_mapped) {
          memmappedEnv_[entry_name] = std::make_unique<tensorflow::MemmappedEnv>(tensorflow::Env::Default());
          const tensorflow::Status mmap_status = memmappedEnv_.at(entry_name)->InitializeFromFile(graph_file);
          if (!mmap_status.ok()) {
            throw cms::Exception("DeepTauCache: unable to initalize memmapped environment for ")
                << graph_file << ". \n"
                << mmap_status.ToString();
          }

          graphs_[entry_name] = std::make_unique<tensorflow::GraphDef>();
          const tensorflow::Status load_graph_status =
              ReadBinaryProto(memmappedEnv_.at(entry_name).get(),
                              tensorflow::MemmappedFileSystem::kMemmappedPackageDefaultGraphDef,
                              graphs_.at(entry_name).get());
          if (!load_graph_status.ok())
            throw cms::Exception("DeepTauCache: unable to load graph from ") << graph_file << ". \n"
                                                                             << load_graph_status.ToString();

          options.getSessionOptions().config.mutable_graph_options()->mutable_optimizer_options()->set_opt_level(
              ::tensorflow::OptimizerOptions::L0);
          options.getSessionOptions().env = memmappedEnv_.at(entry_name).get();

          sessions_[entry_name] = tensorflow::createSession(graphs_.at(entry_name).get(), options);

        } else {
          graphs_[entry_name].reset(tensorflow::loadGraphDef(graph_file));
          sessions_[entry_name] = tensorflow::createSession(graphs_.at(entry_name).get(), options);
        }
      }
    };
    ~DeepTauCache() {
      for (auto& session_entry : sessions_)
        tensorflow::closeSession(session_entry.second);
    }

    // A Session allows concurrent calls to Run(), though a Session must
    // be created / extended by a single thread.
    tensorflow::Session& getSession(const std::string& name = "") const { return *sessions_.at(name); }
    const tensorflow::GraphDef& getGraph(const std::string& name = "") const { return *graphs_.at(name); }

  private:
    std::map<std::string, GraphPtr> graphs_;
    std::map<std::string, tensorflow::Session*> sessions_;
    std::map<std::string, std::unique_ptr<tensorflow::MemmappedEnv>> memmappedEnv_;
  };
}  // namespace deep_tau

class DeepTauIdWrapper : public edm::stream::EDProducer<edm::GlobalCache<deep_tau::DeepTauCache>> {
public:
  DeepTauIdWrapper(const edm::ParameterSet& cfg) {}
};

class DeepTauId : public DeepTauIdBase<DeepTauIdWrapper> {
public:
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    fillDescriptionsHelper(desc);
    desc.add<std::vector<std::string>>("graph_file",
                                       {"RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6.pb"});
    descriptions.add("DeepTau", desc);
  }

public:
  explicit DeepTauId(const edm::ParameterSet& cfg, const deep_tau::DeepTauCache* cache)
      : DeepTauIdBase<DeepTauIdWrapper>(cfg), cache_(cache) {
    if (version_ == 2) {
      using namespace dnn_inputs_v2;
      if (sub_version_ == 1) {
        tauBlockTensor_ = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, TauBlockInputs::NumberOfInputs});
      } else if ((sub_version_ == 5) || ((sub_version_ == 0) && (year_ == 20161718))) {
        tauBlockTensor_ = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT,
            tensorflow::TensorShape{1,
                                    static_cast<int>(TauBlockInputs::NumberOfInputs) -
                                        static_cast<int>(TauBlockInputs::varsToDrop.size())});
      }

      for (size_t n = 0; n < 2; ++n) {
        const bool is_inner = n == 0;
        const auto n_cells = is_inner ? number_of_inner_cell : number_of_outer_cell;
        eGammaTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 1, 1, EgammaBlockInputs::NumberOfInputs});
        muonTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 1, 1, MuonBlockInputs::NumberOfInputs});
        hadronsTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 1, 1, HadronBlockInputs::NumberOfInputs});
        convTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, n_cells, n_cells, number_of_conv_features});
        zeroOutputTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
            tensorflow::DT_FLOAT, tensorflow::TensorShape{1, 1, 1, number_of_conv_features});

        eGammaTensor_[is_inner]->flat<float>().setZero();
        muonTensor_[is_inner]->flat<float>().setZero();
        hadronsTensor_[is_inner]->flat<float>().setZero();

        setCellConvFeatures(*zeroOutputTensor_[is_inner], getPartialPredictions(is_inner), 0, 0, 0);
      }
    }
  }

  void produce(edm::Event& event, const edm::EventSetup& es) override {
    edm::Handle<TauCollection> taus;
    event.getByToken(tausToken_, taus);

    // store empty output collection(s) if tau collection is empty
    if (taus->empty()) {
      const tensorflow::Tensor emptyPrediction(tensorflow::DT_FLOAT, {0, deep_tau::NumberOfOutputs});
      createOutputs(event, emptyPrediction, taus);
      return;
    }

    loadPrediscriminants(event, taus);

    const tensorflow::Tensor& pred = getPredictions(event, taus);
    createOutputs(event, pred, taus);
  }

  static std::unique_ptr<deep_tau::DeepTauCache> initializeGlobalCache(const edm::ParameterSet& cfg) {
    const auto graph_name_vector = cfg.getParameter<std::vector<std::string>>("graph_file");
    std::map<std::string, std::string> graph_names;
    for (const auto& entry : graph_name_vector) {
      const size_t sep_pos = entry.find(':');
      std::string entry_name, graph_file;
      if (sep_pos != std::string::npos) {
        entry_name = entry.substr(0, sep_pos);
        graph_file = entry.substr(sep_pos + 1);
      } else {
        entry_name = "";
        graph_file = entry;
      }
      graph_file = edm::FileInPath(graph_file).fullPath();
      if (graph_names.count(entry_name))
        throw cms::Exception("DeepTauCache") << "Duplicated graph entries";
      graph_names[entry_name] = graph_file;
    }
    bool mem_mapped = cfg.getParameter<bool>("mem_mapped");
    return std::make_unique<deep_tau::DeepTauCache>(graph_names, mem_mapped);
  }

  static void globalEndJob(const deep_tau::DeepTauCache* cache_) {}

private:
  inline void checkInputs(const tensorflow::Tensor& inputs,
                          const std::string& block_name,
                          int n_inputs,
                          const CellGrid* grid = nullptr) const {
    if (debug_level >= 1) {
      std::cout << "<checkInputs>: block_name = " << block_name << std::endl;
      if (block_name == "input_tau") {
        for (int input_index = 0; input_index < n_inputs; ++input_index) {
          float input = inputs.matrix<float>()(0, input_index);
          if (edm::isNotFinite(input)) {
            throw cms::Exception("DeepTauId")
                << "in the " << block_name
                << ", input is not finite, i.e. infinite or NaN, for input_index = " << input_index;
          }
          if (debug_level >= 2) {
            std::cout << block_name << "[var = " << input_index << "] = " << std::setprecision(5) << std::fixed << input
                      << std::endl;
          }
        }
      } else {
        assert(grid);
        int n_eta, n_phi;
        if (block_name.find("input_inner") != std::string::npos) {
          n_eta = 5;
          n_phi = 5;
        } else if (block_name.find("input_outer") != std::string::npos) {
          n_eta = 10;
          n_phi = 10;
        } else
          assert(0);
        int eta_phi_index = 0;
        for (int eta = -n_eta; eta <= n_eta; ++eta) {
          for (int phi = -n_phi; phi <= n_phi; ++phi) {
            const CellIndex cell_index{eta, phi};
            const auto cell_iter = grid->find(cell_index);
            if (cell_iter != grid->end()) {
              for (int input_index = 0; input_index < n_inputs; ++input_index) {
                float input = inputs.tensor<float, 4>()(eta_phi_index, 0, 0, input_index);
                if (edm::isNotFinite(input)) {
                  throw cms::Exception("DeepTauId")
                      << "in the " << block_name << ", input is not finite, i.e. infinite or NaN, for eta = " << eta
                      << ", phi = " << phi << ", input_index = " << input_index;
                }
                if (debug_level >= 2) {
                  std::cout << block_name << "[eta = " << eta << "][phi = " << phi << "][var = " << input_index
                            << "] = " << std::setprecision(5) << std::fixed << input << std::endl;
                }
              }
              eta_phi_index += 1;
            }
          }
        }
      }
    }
  }

  inline void saveInputs(const tensorflow::Tensor& inputs,
                         const std::string& block_name,
                         int n_inputs,
                         const CellGrid* grid = nullptr) {
    if (debug_level >= 1) {
      std::cout << "<saveInputs>: block_name = " << block_name << std::endl;
    }
    if (!is_first_block_)
      (*json_file_) << ", ";
    (*json_file_) << "\"" << block_name << "\": [";
    if (block_name == "input_tau") {
      for (int input_index = 0; input_index < n_inputs; ++input_index) {
        float input = inputs.matrix<float>()(0, input_index);
        if (input_index != 0)
          (*json_file_) << ", ";
        (*json_file_) << input;
      }
    } else {
      assert(grid);
      int n_eta, n_phi;
      if (block_name.find("input_inner") != std::string::npos) {
        n_eta = 5;
        n_phi = 5;
      } else if (block_name.find("input_outer") != std::string::npos) {
        n_eta = 10;
        n_phi = 10;
      } else
        assert(0);
      int eta_phi_index = 0;
      for (int eta = -n_eta; eta <= n_eta; ++eta) {
        if (eta != -n_eta)
          (*json_file_) << ", ";
        (*json_file_) << "[";
        for (int phi = -n_phi; phi <= n_phi; ++phi) {
          if (phi != -n_phi)
            (*json_file_) << ", ";
          (*json_file_) << "[";
          const CellIndex cell_index{eta, phi};
          const auto cell_iter = grid->find(cell_index);
          for (int input_index = 0; input_index < n_inputs; ++input_index) {
            float input = 0.;
            if (cell_iter != grid->end()) {
              input = inputs.tensor<float, 4>()(eta_phi_index, 0, 0, input_index);
            }
            if (input_index != 0)
              (*json_file_) << ", ";
            (*json_file_) << input;
          }
          if (cell_iter != grid->end()) {
            eta_phi_index += 1;
          }
          (*json_file_) << "]";
        }
        (*json_file_) << "]";
      }
    }
    (*json_file_) << "]";
    is_first_block_ = false;
  }

private:
  tensorflow::Tensor getPredictions(edm::Event& event, edm::Handle<TauCollection> taus) {
    // Empty dummy vectors
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

      // Get indices for discriminators
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

    auto const& eventnr = event.id().event();

    tensorflow::Tensor predictions(tensorflow::DT_FLOAT, {static_cast<int>(taus->size()), deep_tau::NumberOfOutputs});

    for (size_t tau_index = 0; tau_index < taus->size(); ++tau_index) {
      const edm::RefToBase<reco::BaseTau> tauRef = taus->refAt(tau_index);

      std::vector<tensorflow::Tensor> pred_vector;

      bool passesPrediscriminants;
      if (is_online_) {
        passesPrediscriminants = tauIDs.passPrediscriminants<std::vector<TauDiscInfo<reco::PFTauDiscriminator>>>(
            recoPrediscriminants_, andPrediscriminants_, tauRef);
      } else {
        passesPrediscriminants = tauIDs.passPrediscriminants<std::vector<TauDiscInfo<pat::PATTauDiscriminator>>>(
            patPrediscriminants_, andPrediscriminants_, tauRef);
      }

      if (passesPrediscriminants) {
        if (version_ == 2) {
          if (is_online_) {
            getPredictionsV2<reco::PFCandidate, reco::PFTau>(taus->at(tau_index),
                                                             tau_index,
                                                             tauRef,
                                                             electron_collection,
                                                             muon_collection,
                                                             *pfCands,
                                                             vertices->at(0),
                                                             *rho,
                                                             eventnr,
                                                             pred_vector,
                                                             tauIDs);
          } else
            getPredictionsV2<pat::PackedCandidate, pat::Tau>(taus->at(tau_index),
                                                             tau_index,
                                                             tauRef,
                                                             electron_collection,
                                                             muon_collection,
                                                             *pfCands,
                                                             vertices->at(0),
                                                             *rho,
                                                             eventnr,
                                                             pred_vector,
                                                             tauIDs);
        } else {
          throw cms::Exception("DeepTauId") << "version " << version_ << " is not supported.";
        }

        for (int k = 0; k < deep_tau::NumberOfOutputs; ++k) {
          const float pred = pred_vector[0].flat<float>()(k);
          if (!(pred >= 0 && pred <= 1))
            throw cms::Exception("DeepTauId")
                << "invalid prediction = " << pred << " for tau_index = " << tau_index << ", pred_index = " << k;
          predictions.matrix<float>()(tau_index, k) = pred;
        }
      } else {
        // This else statement was added as a part of the DeepTau@HLT development. It does not affect the current state
        // of offline DeepTauId code as there the preselection is not used (it was added in the DeepTau@HLT). It returns
        // default values for deepTau score if the preselection failed. Before this statement the values given for this tau
        // were random. k == 2 corresponds to the tau score and all other k values to e, mu and jets. By defining in this way
        // the final score is -1.
        for (int k = 0; k < deep_tau::NumberOfOutputs; ++k) {
          predictions.matrix<float>()(tau_index, k) = (k == 2) ? -1.f : 2.f;
        }
      }
    }
    return predictions;
  }

  template <typename CandidateCastType, typename TauCastType>
  void getPredictionsV2(TauCollection::const_reference& tau,
                        const size_t tau_index,
                        const edm::RefToBase<reco::BaseTau> tau_ref,
                        const std::vector<pat::Electron>* electrons,
                        const std::vector<pat::Muon>* muons,
                        const edm::View<reco::Candidate>& pfCands,
                        const reco::Vertex& pv,
                        double rho,
                        const edm::EventNumber_t& eventnr,
                        std::vector<tensorflow::Tensor>& pred_vector,
                        TauFunc tau_funcs) {
    using namespace dnn_inputs_v2;
    if (debug_level >= 2) {
      std::cout << "<DeepTauId::getPredictionsV2 (moduleLabel = " << moduleDescription().moduleLabel()
                << ")>:" << std::endl;
      std::cout << " tau: pT = " << tau.pt() << ", eta = " << tau.eta() << ", phi = " << tau.phi()
                << ", eventnr = " << eventnr << std::endl;
    }
    CellGrid inner_grid(number_of_inner_cell, number_of_inner_cell, 0.02, 0.02, disable_CellIndex_workaround_);
    CellGrid outer_grid(number_of_outer_cell, number_of_outer_cell, 0.05, 0.05, disable_CellIndex_workaround_);
    fillGrids(dynamic_cast<const TauCastType&>(tau), *electrons, inner_grid, outer_grid);
    fillGrids(dynamic_cast<const TauCastType&>(tau), *muons, inner_grid, outer_grid);
    fillGrids(dynamic_cast<const TauCastType&>(tau), pfCands, inner_grid, outer_grid);

    tauBlockTensor_->flat<float>().setZero();
    createTauBlockInputs<CandidateCastType>(
        dynamic_cast<const TauCastType&>(tau), tau_index, tau_ref, pv, rho, tau_funcs, *tauBlockTensor_);
    checkInputs(*tauBlockTensor_, "input_tau", static_cast<int>(tauBlockTensor_->shape().dim_size(1)));
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
    checkInputs(*eGammaTensor_[true], "input_inner_egamma", EgammaBlockInputs::NumberOfInputs, &inner_grid);
    checkInputs(*muonTensor_[true], "input_inner_muon", MuonBlockInputs::NumberOfInputs, &inner_grid);
    checkInputs(*hadronsTensor_[true], "input_inner_hadrons", HadronBlockInputs::NumberOfInputs, &inner_grid);
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
    checkInputs(*eGammaTensor_[false], "input_outer_egamma", EgammaBlockInputs::NumberOfInputs, &outer_grid);
    checkInputs(*muonTensor_[false], "input_outer_muon", MuonBlockInputs::NumberOfInputs, &outer_grid);
    checkInputs(*hadronsTensor_[false], "input_outer_hadrons", HadronBlockInputs::NumberOfInputs, &outer_grid);

    if (save_inputs_) {
      std::string json_file_name = "DeepTauId_" + std::to_string(eventnr) + "_" + std::to_string(tau_index) + ".json";
      json_file_ = new std::ofstream(json_file_name.data());
      is_first_block_ = true;
      (*json_file_) << "{";
      saveInputs(*tauBlockTensor_, "input_tau", static_cast<int>(tauBlockTensor_->shape().dim_size(1)));
      saveInputs(*eGammaTensor_[true], "input_inner_egamma", EgammaBlockInputs::NumberOfInputs, &inner_grid);
      saveInputs(*muonTensor_[true], "input_inner_muon", MuonBlockInputs::NumberOfInputs, &inner_grid);
      saveInputs(*hadronsTensor_[true], "input_inner_hadrons", HadronBlockInputs::NumberOfInputs, &inner_grid);
      saveInputs(*eGammaTensor_[false], "input_outer_egamma", EgammaBlockInputs::NumberOfInputs, &outer_grid);
      saveInputs(*muonTensor_[false], "input_outer_muon", MuonBlockInputs::NumberOfInputs, &outer_grid);
      saveInputs(*hadronsTensor_[false], "input_outer_hadrons", HadronBlockInputs::NumberOfInputs, &outer_grid);
      (*json_file_) << "}";
      delete json_file_;
      ++file_counter_;
    }

    tensorflow::run(&(cache_->getSession("core")),
                    {{"input_tau", *tauBlockTensor_},
                     {"input_inner", *convTensor_.at(true)},
                     {"input_outer", *convTensor_.at(false)}},
                    {"main_output/Softmax"},
                    &pred_vector);
    if (debug_level >= 1) {
      std::cout << "output = { ";
      for (int idx = 0; idx < deep_tau::NumberOfOutputs; ++idx) {
        if (idx > 0)
          std::cout << ", ";
        std::string label;
        if (idx == 0)
          label = "e";
        else if (idx == 1)
          label = "mu";
        else if (idx == 2)
          label = "tau";
        else if (idx == 3)
          label = "jet";
        else
          assert(0);
        std::cout << label << " = " << pred_vector[0].flat<float>()(idx);
      }
      std::cout << " }" << std::endl;
    }
  }

  tensorflow::Tensor getPartialPredictions(bool is_inner) {
    std::vector<tensorflow::Tensor> pred_vector;
    if (is_inner) {
      tensorflow::run(&(cache_->getSession("inner")),
                      {
                          {"input_inner_egamma", *eGammaTensor_.at(is_inner)},
                          {"input_inner_muon", *muonTensor_.at(is_inner)},
                          {"input_inner_hadrons", *hadronsTensor_.at(is_inner)},
                      },
                      {"inner_all_dropout_4/Identity"},
                      &pred_vector);
    } else {
      tensorflow::run(&(cache_->getSession("outer")),
                      {
                          {"input_outer_egamma", *eGammaTensor_.at(is_inner)},
                          {"input_outer_muon", *muonTensor_.at(is_inner)},
                          {"input_outer_hadrons", *hadronsTensor_.at(is_inner)},
                      },
                      {"outer_all_dropout_4/Identity"},
                      &pred_vector);
    }

    return pred_vector.at(0);
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
    if (debug_level >= 2) {
      std::cout << "<DeepTauId::createConvFeatures (is_inner = " << is_inner << ")>:" << std::endl;
      std::cout << "number of valid cells = " << grid.num_valid_cells() << std::endl;
    }

    const size_t n_valid_cells = grid.num_valid_cells();
    tensorflow::Tensor predTensor;
    tensorflow::Tensor& convTensor = *convTensor_.at(is_inner);

    //check if at least one input is there to
    //avoid calling TF with empty grid #TODO understand why the grid is empty
    if (n_valid_cells > 0) {
      eGammaTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
          tensorflow::DT_FLOAT,
          tensorflow::TensorShape{
              (long long int)grid.num_valid_cells(), 1, 1, dnn_inputs_v2::EgammaBlockInputs::NumberOfInputs});
      muonTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
          tensorflow::DT_FLOAT,
          tensorflow::TensorShape{
              (long long int)grid.num_valid_cells(), 1, 1, dnn_inputs_v2::MuonBlockInputs::NumberOfInputs});
      hadronsTensor_[is_inner] = std::make_unique<tensorflow::Tensor>(
          tensorflow::DT_FLOAT,
          tensorflow::TensorShape{
              (long long int)grid.num_valid_cells(), 1, 1, dnn_inputs_v2::HadronBlockInputs::NumberOfInputs});

      eGammaTensor_[is_inner]->flat<float>().setZero();
      muonTensor_[is_inner]->flat<float>().setZero();
      hadronsTensor_[is_inner]->flat<float>().setZero();

      unsigned idx = 0;
      for (int eta = -grid.maxEtaIndex(); eta <= grid.maxEtaIndex(); ++eta) {
        for (int phi = -grid.maxPhiIndex(); phi <= grid.maxPhiIndex(); ++phi) {
          if (debug_level >= 2) {
            std::cout << "processing ( eta = " << eta << ", phi = " << phi << " )" << std::endl;
          }
          const CellIndex cell_index{eta, phi};
          const auto cell_iter = grid.find(cell_index);
          if (cell_iter != grid.end()) {
            if (debug_level >= 2) {
              std::cout << " creating inputs for ( eta = " << eta << ", phi = " << phi << " ): idx = " << idx
                        << std::endl;
            }
            const Cell& cell = cell_iter->second;
            createEgammaBlockInputs<CandidateCastType>(idx,
                                                       tau,
                                                       tau_index,
                                                       tau_ref,
                                                       pv,
                                                       rho,
                                                       electrons,
                                                       pfCands,
                                                       cell,
                                                       tau_funcs,
                                                       is_inner,
                                                       *eGammaTensor_[is_inner]);
            createMuonBlockInputs<CandidateCastType>(
                idx, tau, tau_index, tau_ref, pv, rho, muons, pfCands, cell, tau_funcs, is_inner, *muonTensor_[is_inner]);
            createHadronsBlockInputs<CandidateCastType>(
                idx, tau, tau_index, tau_ref, pv, rho, pfCands, cell, tau_funcs, is_inner, *hadronsTensor_[is_inner]);
            idx += 1;
          } else {
            if (debug_level >= 2) {
              std::cout << " skipping creation of inputs, because ( eta = " << eta << ", phi = " << phi
                        << " ) is not in the grid !!" << std::endl;
            }
          }
        }
      }
      // Calling TF prediction only if n_valid_cells > 0
      predTensor = getPartialPredictions(is_inner);
    }

    unsigned idx = 0;
    for (int eta = -grid.maxEtaIndex(); eta <= grid.maxEtaIndex(); ++eta) {
      for (int phi = -grid.maxPhiIndex(); phi <= grid.maxPhiIndex(); ++phi) {
        const CellIndex cell_index{eta, phi};
        const int eta_index = grid.getEtaTensorIndex(cell_index);
        const int phi_index = grid.getPhiTensorIndex(cell_index);

        const auto cell_iter = grid.find(cell_index);
        if (cell_iter != grid.end()) {
          setCellConvFeatures(convTensor, predTensor, idx, eta_index, phi_index);
          idx += 1;
        } else {
          setCellConvFeatures(convTensor, *zeroOutputTensor_[is_inner], 0, eta_index, phi_index);
        }
      }
    }
  }

  void setCellConvFeatures(tensorflow::Tensor& convTensor,
                           const tensorflow::Tensor& features,
                           unsigned batch_idx,
                           int eta_index,
                           int phi_index) {
    for (int n = 0; n < dnn_inputs_v2::number_of_conv_features; ++n) {
      convTensor.tensor<float, 4>()(0, eta_index, phi_index, n) = features.tensor<float, 4>()(batch_idx, 0, 0, n);
    }
  }

private:
  const deep_tau::DeepTauCache* cache_;

  std::unique_ptr<tensorflow::Tensor> tauBlockTensor_;
  std::array<std::unique_ptr<tensorflow::Tensor>, 2> eGammaTensor_, muonTensor_, hadronsTensor_, convTensor_,
      zeroOutputTensor_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeepTauId);
