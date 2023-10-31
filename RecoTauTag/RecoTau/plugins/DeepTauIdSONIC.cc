/*
 * \class DeepTauIdSONIC
 *
 * Tau identification using Deep NN with SONIC
 *
 */

#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"
#include "RecoTauTag/RecoTau/interface/DeepTauIdBase.h"

class DeepTauIdSonicProducer : public DeepTauIdBase<TritonEDProducer<>> {
public:
  explicit DeepTauIdSonicProducer(edm::ParameterSet const& cfg) : DeepTauIdBase<TritonEDProducer<>>(cfg){};

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  std::vector<size_t> tau_indices_;

  std::vector<float>* p_tauBlockInputs;
  std::vector<float>* p_egammaInnerBlockInputs;
  std::vector<float>* p_muonInnerBlockInputs;
  std::vector<float>* p_hadronInnerBlockInputs;
  std::vector<float>* p_egammaOuterBlockInputs;
  std::vector<float>* p_muonOuterBlockInputs;
  std::vector<float>* p_hadronOuterBlockInputs;
  std::vector<int64_t>* p_innerGridposInputs;
  std::vector<int64_t>* p_outerGridposInputs;

  template <typename CandidateCastType, typename TauCastType>
  void prepareInputsV2(TauCollection::const_reference& tau,
                       const size_t tau_index,
                       const edm::RefToBase<reco::BaseTau> tau_ref,
                       const std::vector<pat::Electron>* electrons,
                       const std::vector<pat::Muon>* muons,
                       const edm::View<reco::Candidate>& pfCands,
                       const reco::Vertex& pv,
                       double rho,
                       const TauFunc& tau_funcs);

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
                          const TauFunc& tau_funcs,
                          bool is_inner,
                          std::vector<float>* p_egammaBlockInputs,
                          std::vector<float>* p_muonBlockInputs,
                          std::vector<float>* p_hadronBlockInputs,
                          std::vector<int64_t>* p_GridposInputs);
};

void DeepTauIdSonicProducer::acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) {
  edm::Handle<TauCollection> taus;
  iEvent.getByToken(tausToken_, taus);

  loadPrediscriminants(iEvent, taus);

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
    electron_collection = &iEvent.get(electrons_token_);
    muon_collection = &iEvent.get(muons_token_);
    pfTauTransverseImpactParameters = &pfTauTransverseImpactParameters_default;
    basicTauDiscriminators = &basicTauDiscriminators_default;
    basicTauDiscriminatorsdR03 = &basicTauDiscriminatorsdR03_default;
  } else {
    electron_collection = &electron_collection_default;
    muon_collection = &muon_collection_default;
    pfTauTransverseImpactParameters = &iEvent.get(pfTauTransverseImpactParameters_token_);
    basicTauDiscriminators = &iEvent.get(basicTauDiscriminators_inputToken_);
    basicTauDiscriminatorsdR03 = &iEvent.get(basicTauDiscriminatorsdR03_inputToken_);

    // Get indices for discriminators
    if (!discrIndicesMapped_) {
      basicDiscrIndexMap_ =
          matchDiscriminatorIndices(iEvent, basicTauDiscriminators_inputToken_, requiredBasicDiscriminators_);
      basicDiscrdR03IndexMap_ =
          matchDiscriminatorIndices(iEvent, basicTauDiscriminatorsdR03_inputToken_, requiredBasicDiscriminatorsdR03_);
      discrIndicesMapped_ = true;
    }
  }

  TauFunc tauIDs = {basicTauDiscriminators,
                    basicTauDiscriminatorsdR03,
                    pfTauTransverseImpactParameters,
                    basicDiscrIndexMap_,
                    basicDiscrdR03IndexMap_};

  edm::Handle<edm::View<reco::Candidate>> pfCands;
  iEvent.getByToken(pfcandToken_, pfCands);

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vtxToken_, vertices);

  edm::Handle<double> rho;
  iEvent.getByToken(rho_token_, rho);

  // vector to store the indices for the taus passing the selections
  tau_indices_.clear();

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
    if (!passesPrediscriminants)
      continue;

    // tau index that passes the selection
    tau_indices_.push_back(tau_index);
  }

  if (tau_indices_.empty()) {
    // no tau passing the requirement
    // no need to run acquire and inference
    client_->setBatchSize(0);
    return;
  }

  int n_taus = tau_indices_.size();
  // for the regular non-split deep-tau model, set the
  // batch size to the number of taus per event
  client_->setBatchSize(n_taus);

  auto& input_tauBlock = iInput.at("input_tau");
  auto& input_innerEgammaBlock = iInput.at("input_inner_egamma");
  auto& input_outerEgammaBlock = iInput.at("input_outer_egamma");
  auto& input_innerMuonBlock = iInput.at("input_inner_muon");
  auto& input_outerMuonBlock = iInput.at("input_outer_muon");
  auto& input_innerHadronBlock = iInput.at("input_inner_hadrons");
  auto& input_outerHadronBlock = iInput.at("input_outer_hadrons");

  p_innerGridposInputs = nullptr;
  p_outerGridposInputs = nullptr;

  auto data_tauBlock = input_tauBlock.allocate<float>();
  auto data_innerEgammaBlock = input_innerEgammaBlock.allocate<float>();
  auto data_outerEgammaBlock = input_outerEgammaBlock.allocate<float>();
  auto data_innerMuonBlock = input_innerMuonBlock.allocate<float>();
  auto data_outerMuonBlock = input_outerMuonBlock.allocate<float>();
  auto data_innerHadronBlock = input_innerHadronBlock.allocate<float>();
  auto data_outerHadronBlock = input_outerHadronBlock.allocate<float>();

  for (unsigned itau_passed = 0; itau_passed < tau_indices_.size(); ++itau_passed) {
    // batch size is not one, needs to go to corresponding itau
    p_tauBlockInputs = &((*data_tauBlock)[itau_passed]);

    p_egammaInnerBlockInputs = &((*data_innerEgammaBlock)[itau_passed]);
    p_muonInnerBlockInputs = &((*data_innerMuonBlock)[itau_passed]);
    p_hadronInnerBlockInputs = &((*data_innerHadronBlock)[itau_passed]);

    p_egammaOuterBlockInputs = &((*data_outerEgammaBlock)[itau_passed]);
    p_muonOuterBlockInputs = &((*data_outerMuonBlock)[itau_passed]);
    p_hadronOuterBlockInputs = &((*data_outerHadronBlock)[itau_passed]);

    int tau_index = tau_indices_[itau_passed];
    const edm::RefToBase<reco::BaseTau> tauRef = taus->refAt(tau_index);
    prepareInputsV2<pat::PackedCandidate, pat::Tau>(taus->at(tau_index),
                                                    tau_index,
                                                    tauRef,
                                                    electron_collection,
                                                    muon_collection,
                                                    *pfCands,
                                                    vertices->at(0),
                                                    *rho,
                                                    tauIDs);
  }

  // set all input data to the server
  input_tauBlock.toServer(data_tauBlock);

  input_innerEgammaBlock.toServer(data_innerEgammaBlock);
  input_innerMuonBlock.toServer(data_innerMuonBlock);
  input_innerHadronBlock.toServer(data_innerHadronBlock);

  input_outerEgammaBlock.toServer(data_outerEgammaBlock);
  input_outerMuonBlock.toServer(data_outerMuonBlock);
  input_outerHadronBlock.toServer(data_outerHadronBlock);
}

void DeepTauIdSonicProducer::produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) {
  edm::Handle<TauCollection> taus;
  iEvent.getByToken(tausToken_, taus);

  if (taus->empty()) {
    std::vector<std::vector<float>> pred_all(0, std::vector<float>(deep_tau::NumberOfOutputs, 0.));
    createOutputs(iEvent, pred_all, taus);
    return;
  }

  const auto& output_tauval = iOutput.at("main_output/Softmax");
  const auto& outputs_tauval = output_tauval.fromServer<float>();

  // fill the taus passing the selections with the results from produce,
  //  and the taus failing the selections with 2 and -1.0
  std::vector<std::vector<float>> pred_all(taus->size(), std::vector<float>(deep_tau::NumberOfOutputs, 0.));
  for (unsigned itau = 0; itau < taus->size(); ++itau) {
    for (int k = 0; k < deep_tau::NumberOfOutputs; ++k) {
      pred_all[itau][k] = (k == 2) ? -1.f : 2.f;
    }
  }
  for (unsigned itau_passed = 0; itau_passed < tau_indices_.size(); ++itau_passed) {
    int tau_index = tau_indices_[itau_passed];
    std::copy(outputs_tauval[itau_passed].begin(), outputs_tauval[itau_passed].end(), pred_all[tau_index].begin());

    if (debug_level >= 2) {
      for (int i = 0; i < 4; ++i) {
        std::cout << "tau index " << itau_passed << " k " << i << " pred " << pred_all[tau_index][i] << std::endl;
      }
    }
  }
  createOutputs(iEvent, pred_all, taus);
}

template <typename CandidateCastType, typename TauCastType>
void DeepTauIdSonicProducer::prepareInputsV2(TauCollection::const_reference& tau,
                                             const size_t tau_index,
                                             const edm::RefToBase<reco::BaseTau> tau_ref,
                                             const std::vector<pat::Electron>* electrons,
                                             const std::vector<pat::Muon>* muons,
                                             const edm::View<reco::Candidate>& pfCands,
                                             const reco::Vertex& pv,
                                             double rho,
                                             const TauFunc& tau_funcs) {
  using namespace dnn_inputs_v2;
  CellGrid inner_grid(number_of_inner_cell, number_of_inner_cell, 0.02, 0.02, disable_CellIndex_workaround_);
  CellGrid outer_grid(number_of_outer_cell, number_of_outer_cell, 0.05, 0.05, disable_CellIndex_workaround_);
  auto tau_casted = dynamic_cast<const TauCastType&>(tau);
  // fill in the inner and outer grids for electrons, muons, and pfCands
  fillGrids(tau_casted, *electrons, inner_grid, outer_grid);
  fillGrids(tau_casted, *muons, inner_grid, outer_grid);
  fillGrids(tau_casted, pfCands, inner_grid, outer_grid);

  p_tauBlockInputs->insert(p_tauBlockInputs->end(), TauBlockInputs::NumberOfInputs, 0.);
  std::vector<float>::iterator tauIter = p_tauBlockInputs->begin();

  createTauBlockInputs<CandidateCastType>(tau_casted, tau_index, tau_ref, pv, rho, tau_funcs, tauIter);

  // egamma, muon, and hadron inner and outer inputs for the grids
  createConvFeatures<CandidateCastType>(tau_casted,
                                        tau_index,
                                        tau_ref,
                                        pv,
                                        rho,
                                        electrons,
                                        muons,
                                        pfCands,
                                        inner_grid,
                                        tau_funcs,
                                        true,
                                        p_egammaInnerBlockInputs,
                                        p_muonInnerBlockInputs,
                                        p_hadronInnerBlockInputs,
                                        p_innerGridposInputs);
  createConvFeatures<CandidateCastType>(tau_casted,
                                        tau_index,
                                        tau_ref,
                                        pv,
                                        rho,
                                        electrons,
                                        muons,
                                        pfCands,
                                        outer_grid,
                                        tau_funcs,
                                        false,
                                        p_egammaOuterBlockInputs,
                                        p_muonOuterBlockInputs,
                                        p_hadronOuterBlockInputs,
                                        p_outerGridposInputs);
}

template <typename CandidateCastType, typename TauCastType>
void DeepTauIdSonicProducer::createConvFeatures(const TauCastType& tau,
                                                const size_t tau_index,
                                                const edm::RefToBase<reco::BaseTau> tau_ref,
                                                const reco::Vertex& pv,
                                                double rho,
                                                const std::vector<pat::Electron>* electrons,
                                                const std::vector<pat::Muon>* muons,
                                                const edm::View<reco::Candidate>& pfCands,
                                                const CellGrid& grid,
                                                const TauFunc& tau_funcs,
                                                bool is_inner,
                                                std::vector<float>* p_egammaBlockInputs,
                                                std::vector<float>* p_muonBlockInputs,
                                                std::vector<float>* p_hadronBlockInputs,
                                                std::vector<int64_t>* p_GridposInputs) {
  // fill in the block inputs with zeros
  int n_cells = 0;
  int n_cell_oneside = is_inner ? dnn_inputs_v2::number_of_inner_cell : dnn_inputs_v2::number_of_outer_cell;

  n_cells = is_inner ? (dnn_inputs_v2::number_of_inner_cell * dnn_inputs_v2::number_of_inner_cell)
                     : (dnn_inputs_v2::number_of_outer_cell * dnn_inputs_v2::number_of_outer_cell);

  p_egammaBlockInputs->insert(
      p_egammaBlockInputs->end(), n_cells * dnn_inputs_v2::EgammaBlockInputs::NumberOfInputs, 0.);
  std::vector<float>::iterator egammaIter = p_egammaBlockInputs->begin();

  p_muonBlockInputs->insert(p_muonBlockInputs->end(), n_cells * dnn_inputs_v2::MuonBlockInputs::NumberOfInputs, 0.);
  std::vector<float>::iterator muonIter = p_muonBlockInputs->begin();

  p_hadronBlockInputs->insert(
      p_hadronBlockInputs->end(), n_cells * dnn_inputs_v2::HadronBlockInputs::NumberOfInputs, 0.);
  std::vector<float>::iterator hadronIter = p_hadronBlockInputs->begin();

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
          std::cout << " creating inputs for ( eta = " << eta << ", phi = " << phi << " ): idx = " << idx << std::endl;
        }
        const Cell& cell = cell_iter->second;
        const int eta_index = grid.getEtaTensorIndex(cell_index);
        const int phi_index = grid.getPhiTensorIndex(cell_index);
        std::vector<float>::iterator egammaIterCell =
            egammaIter + (eta_index * n_cell_oneside + phi_index) * dnn_inputs_v2::EgammaBlockInputs::NumberOfInputs;
        std::vector<float>::iterator muonIterCell =
            muonIter + (eta_index * n_cell_oneside + phi_index) * dnn_inputs_v2::MuonBlockInputs::NumberOfInputs;
        std::vector<float>::iterator hadronIterCell =
            hadronIter + (eta_index * n_cell_oneside + phi_index) * dnn_inputs_v2::HadronBlockInputs::NumberOfInputs;

        createEgammaBlockInputs<CandidateCastType>(
            idx, tau, tau_index, tau_ref, pv, rho, electrons, pfCands, cell, tau_funcs, is_inner, egammaIterCell);
        createMuonBlockInputs<CandidateCastType>(
            idx, tau, tau_index, tau_ref, pv, rho, muons, pfCands, cell, tau_funcs, is_inner, muonIterCell);
        createHadronsBlockInputs<CandidateCastType>(
            idx, tau, tau_index, tau_ref, pv, rho, pfCands, cell, tau_funcs, is_inner, hadronIterCell);

        idx += 1;
      } else {
        if (debug_level >= 2) {
          std::cout << " skipping creation of inputs, because ( eta = " << eta << ", phi = " << phi
                    << " ) is not in the grid !!" << std::endl;
        }
        // we need to fill in the zeros for the input tensors
        idx += 1;
      }
    }
  }
}

void DeepTauIdSonicProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  TritonClient::fillPSetDescription(desc);
  fillDescriptionsHelper(desc);
  descriptions.add("DeepTauIdSonicProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeepTauIdSonicProducer);
