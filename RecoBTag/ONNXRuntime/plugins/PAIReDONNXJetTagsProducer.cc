#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "RecoBTag/FeatureTools/interface/SecondaryVertexConverter.h"
#include "RecoBTag/FeatureTools/interface/NeutralCandidateConverter.h"
#include "RecoBTag/FeatureTools/interface/ChargedCandidateConverter.h"
#include "RecoBTag/FeatureTools/interface/paired_helper.h"

#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

#include "Rivet/Tools/ParticleIdUtils.hh"

using namespace cms::Ort;
using namespace btagbtvdeep;
using namespace Rivet;

class PAIReDONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit PAIReDONNXJetTagsProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~PAIReDONNXJetTagsProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

private:
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;
  typedef reco::VertexCollection VertexCollection;

  const std::string name_, name_pf_, name_sv_;
  const edm::EDGetTokenT<edm::View<pat::Jet>> jet_token_;
  edm::EDGetTokenT<reco::CandidateView> cand_token_;
  edm::EDGetTokenT<edm::View<reco::GenParticle>> gen_particle_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  const edm::EDGetTokenT<SVCollection> sv_token_;
  // define producer functions
  void beginStream(edm::StreamID) override {}
  void produce(edm::Event&, const edm::EventSetup&) override;
  std::vector<size_t> sort_pf_cands(edm::Event& iEvent,
                                    const edm::EventSetup& iSetup,
                                    edm::Handle<edm::View<reco::Candidate>> cands);
  SVCollection sort_svs(edm::Event& iEvent,
                        const edm::EventSetup& iSetup,
                        edm::Handle<SVCollection> svs,
                        edm::Handle<VertexCollection> vtxs);
  void make_inputs(edm::Event& iEvent,
                   const edm::EventSetup& iSetup,
                   edm::Handle<edm::View<pat::Jet>> jets,
                   unsigned i_jet,
                   unsigned j_jet,
                   SVCollection svs,
                   edm::Handle<edm::View<reco::Candidate>> cands,
                   edm::Handle<VertexCollection> vtxs,
                   std::vector<size_t> pf_sorted_idx);
  int get_n_parton(edm::Event& iEvent,
                   const edm::EventSetup& iSetup,
                   edm::Handle<edm::View<pat::Jet>> jets,
                   unsigned i_jet,
                   unsigned j_jet,
                   edm::Handle<edm::View<reco::GenParticle>> gen_particles,
                   int parton_id);
  void endStream() override {}
  // store hard-coded constants indicating size, structure, and names of input arrays
  enum InputIndexes {
    kPfCandFeatures = 0,
    kPfCandVectors = 1,
    kPfCandMasks = 2,
    kSVFeatures = 3,
    kSVVectors = 4,
    kSVMasks = 5
  };
  constexpr static unsigned n_features_vector_ = 4;
  constexpr static unsigned n_features_pf_ = 15;
  constexpr static unsigned n_features_sv_ = 20;
  constexpr static unsigned n_mask_ = 1;
  unsigned max_pf_cands = 128;
  unsigned max_svs = 10;
  std::vector<unsigned> input_sizes_ = {max_pf_cands * n_features_pf_,
                                        max_pf_cands* n_features_vector_,
                                        max_pf_cands* n_mask_,
                                        max_svs* n_features_sv_,
                                        max_svs* n_features_vector_,
                                        max_svs* n_mask_};
  std::vector<std::vector<int64_t>> input_shapes_ = {{(int64_t)1, (int64_t)n_features_pf_, (int64_t)max_pf_cands},
                                                     {(int64_t)1, (int64_t)n_features_vector_, (int64_t)max_pf_cands},
                                                     {(int64_t)1, (int64_t)n_mask_, (int64_t)max_pf_cands},
                                                     {(int64_t)1, (int64_t)n_features_sv_, (int64_t)max_svs},
                                                     {(int64_t)1, (int64_t)n_features_vector_, (int64_t)max_svs},
                                                     {(int64_t)1, (int64_t)n_mask_, (int64_t)max_svs}};
  std::vector<std::string> input_names = {
      "pf_features", "pf_vectors", "pf_mask", "sv_features", "sv_vectors", "sv_mask"};
  std::vector<std::string> output_names = {"label_bb", "label_cc", "label_ll"};
  std::vector<std::string> pf_cand_feature_names_ = {"pf_cand_log_pt",
                                                     "pf_cand_log_e",
                                                     "pf_cand_log_pt_rel",
                                                     "pf_cand_log_e_rel",
                                                     "pf_cand_delta_R_1",
                                                     "pf_cand_delta_R_2",
                                                     "pf_cand_charge",
                                                     "pf_cand_d0",
                                                     "pf_cand_d0_err",
                                                     "pf_cand_dz",
                                                     "pf_cand_dz_err",
                                                     "pf_cand_eta_rel_1",
                                                     "pf_cand_phi_rel_1",
                                                     "pf_cand_eta_rel_2",
                                                     "pf_cand_phi_rel_2"};
  std::vector<std::string> pf_cand_vector_names_ = {"pf_cand_px", "pf_cand_py", "pf_cand_pz", "pf_cand_e"};
  std::vector<std::string> sv_vector_names_ = {"sv_px", "sv_py", "sv_pz", "sv_e"};
  std::vector<std::string> sv_feature_names_ = {
      "sv_charge", "sv_chi2", "sv_d3",        "sv_d3_sig",    "sv_d0",        "sv_d0_sig",   "sv_eta",
      "sv_mass",   "sv_ndof", "sv_ntracks",   "sv_p_angle",   "sv_phi",       "sv_pt",       "sv_x",
      "sv_y",      "sv_z",    "sv_eta_rel_1", "sv_eta_rel_2", "sv_phi_rel_1", "sv_phi_rel_2"};
  FloatArrays data_;  // initialize actual input array
  // define loose cuts on jets to save space
  float jet_pt_cut = 20;  // note: uses raw pt
  float jet_eta_cut = 2.5;
};

PAIReDONNXJetTagsProducer::PAIReDONNXJetTagsProducer(const edm::ParameterSet& iConfig, const ONNXRuntime* cache)
    : name_(iConfig.getParameter<std::string>("name")),
      name_pf_(iConfig.getParameter<std::string>("name_pf")),
      name_sv_(iConfig.getParameter<std::string>("name_sv")),
      jet_token_(consumes<edm::View<pat::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      cand_token_(consumes<reco::CandidateView>(iConfig.getParameter<edm::InputTag>("candidates"))),
      gen_particle_token_(consumes<edm::View<reco::GenParticle>>(iConfig.getParameter<edm::InputTag>("gen_particles"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))) {
  produces<nanoaod::FlatTable>(name_);
}

void PAIReDONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  //descriptions.addWithDefaultLabel(desc);
  desc.add<std::string>("name", "PAIReDJets");
  desc.add<std::string>("name_pf", "PAIReDPF");
  desc.add<std::string>("name_sv", "PAIReDSV");
  desc.add<edm::InputTag>("jets", edm::InputTag("slimmedJetsPuppi"));
  desc.add<edm::InputTag>("candidates", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("gen_particles", edm::InputTag("prunedGenParticles"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlineSlimmedPrimaryVertices"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("slimmedSecondaryVertices"));
  desc.add<edm::FileInPath>("model_path", edm::FileInPath("RecoBTag/Combined/data/PAIReD/model3.onnx"));
  //descriptions.add("PAIReDJetTable", desc);
}

std::unique_ptr<ONNXRuntime> PAIReDONNXJetTagsProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void PAIReDONNXJetTagsProducer::globalEndJob(const ONNXRuntime* cache) {}

void PAIReDONNXJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // initialize output variables
  std::vector<int> n_bparton, n_cparton;
  std::vector<unsigned> idx_jet1, idx_jet2;
  std::vector<float> bb_score, cc_score, other_score, ll_score;
  std::vector<float> outputs(output_names.size(), -1.0);
  //std::vector<float> reg_pt, reg_mass, reg_phi, reg_eta; // for future when jet mass and kinematics are regressed from the model
  // initialize event components
  edm::Handle<reco::VertexCollection> vtxs;
  iEvent.getByToken(vtx_token_, vtxs);
  edm::Handle<edm::View<pat::Jet>> jets;
  iEvent.getByToken(jet_token_, jets);
  edm::Handle<edm::View<reco::Candidate>> cands;
  iEvent.getByToken(cand_token_, cands);
  edm::Handle<edm::View<reco::GenParticle>> gen_particles;
  iEvent.getByToken(gen_particle_token_, gen_particles);
  edm::Handle<reco::VertexCompositePtrCandidateCollection> svs;
  iEvent.getByToken(sv_token_, svs);
  // sort pf cands and svs
  std::vector<size_t> pf_sorted_idx = sort_pf_cands(iEvent, iSetup, cands);
  reco::VertexCompositePtrCandidateCollection svs_sorted = sort_svs(iEvent, iSetup, svs, vtxs);
  bool isMC = !(iEvent.isRealData());  // store MCvsData boolean for adding truth information
  // loop over paired jets and institute pt/eta cuts
  for (unsigned i_jet = 0; i_jet < jets->size() - 1 && jets->size() > 0; ++i_jet) {
    const auto& jet1 = jets->at(i_jet);
    if (jet1.pt() * jet1.jecFactor("Uncorrected") < jet_pt_cut || abs(jet1.eta()) > jet_eta_cut)
      continue;
    for (unsigned j_jet = i_jet + 1; j_jet < jets->size(); ++j_jet) {
      const auto& jet2 = jets->at(j_jet);
      if (jet2.pt() * jet2.jecFactor("Uncorrected") < jet_pt_cut || abs(jet2.eta()) > jet_eta_cut)
        continue;
      // init input data storage and fill it
      data_.clear();
      for (const auto& len : input_sizes_) {
        data_.emplace_back(1 * len, 0);
      }
      make_inputs(iEvent, iSetup, jets, i_jet, j_jet, svs_sorted, cands, vtxs, pf_sorted_idx);
      // run prediction
      outputs = globalCache()->run(input_names, data_, input_shapes_)[0];
      assert(outputs.size() == output_names.size());
      // store output data
      idx_jet1.emplace_back(i_jet);
      idx_jet2.emplace_back(j_jet);
      bb_score.emplace_back(outputs[0]);
      cc_score.emplace_back(outputs[1]);
      ll_score.emplace_back(outputs[2]);
      // reg_pt.emplace_back(outputs[3]);
      // reg_mass.emplace_back(outputs[4]);
      // reg_eta.emplace_back(outputs[5]);
      // reg_phi.emplace_back(outputs[6]);
      if (isMC) {
        n_bparton.emplace_back(get_n_parton(iEvent, iSetup, jets, i_jet, j_jet, gen_particles, 5));
        n_cparton.emplace_back(get_n_parton(iEvent, iSetup, jets, i_jet, j_jet, gen_particles, 4));
      }
    }
  }
  // save paired jet flat table
  auto pjTable = std::make_unique<nanoaod::FlatTable>(idx_jet1.size(), name_, false);
  pjTable->addColumn<unsigned>("idx_jet1", idx_jet1, "Index of constituent jet 1");
  pjTable->addColumn<unsigned>("idx_jet2", idx_jet2, "Index of constituent jet 2");
  pjTable->addColumn<float>("ll_score", ll_score, "Model score for ll jet", 10);
  pjTable->addColumn<float>("cc_score", cc_score, "Model score for cc jet", 10);
  pjTable->addColumn<float>("bb_score", bb_score, "Model score for bb jet", 10);
  // pjTable->addColumn<float>("mass", reg_mass, "Regressed PAIReD jet mass", 10);
  // pjTable->addColumn<float>("pt", reg_pt, "Regressed PAIReD jet pt", 10);
  // pjTable->addColumn<float>("eta", reg_eta, "Regressed PAIReD jet eta", 10);
  // pjTable->addColumn<float>("phi", reg_phi, "Regressed PAIReD jet phi", 10);
  if (isMC) {
    pjTable->addColumn<int>("n_bparton", n_bparton, "Number of b partons", 10);
    pjTable->addColumn<int>("n_cparton", n_cparton, "Number of c partons", 10);
  }
  iEvent.put(std::move(pjTable), name_);
}

std::vector<size_t> PAIReDONNXJetTagsProducer::sort_pf_cands(edm::Event& iEvent,
                                                             const edm::EventSetup& iSetup,
                                                             edm::Handle<edm::View<reco::Candidate>> cands) {
  // sort pf candidates with positive PUPPI weight by pt
  std::vector<float> pf_pts;
  std::vector<size_t> pf_unsorted_idx;
  // generate list of indices and pts with positive puppi weight
  for (unsigned entry = 0; entry < cands->size(); ++entry) {
    const reco::Candidate* cand = &(cands->at(entry));
    auto packed_cand = dynamic_cast<const pat::PackedCandidate*>(cand);
    if (packed_cand->puppiWeight() > 0) {
      pf_pts.emplace_back(cand->pt());
      pf_unsorted_idx.emplace_back(entry);
    }
  }
  // sort the list of pts
  if (pf_unsorted_idx.size() < 2)
    return pf_unsorted_idx;
  std::vector<size_t> pf_sorted_idx(pf_unsorted_idx.size());
  std::iota(pf_sorted_idx.begin(), pf_sorted_idx.end(), 0);
  std::sort(
      pf_sorted_idx.begin(), pf_sorted_idx.end(), [&pf_pts](size_t i, size_t j) { return pf_pts[i] > pf_pts[j]; });
  // put back original indices into sorted list
  for (unsigned i = 0; i < pf_unsorted_idx.size(); ++i) {
    pf_sorted_idx[i] = pf_unsorted_idx[pf_sorted_idx[i]];
  }
  return pf_sorted_idx;
}

reco::VertexCompositePtrCandidateCollection PAIReDONNXJetTagsProducer::sort_svs(edm::Event& iEvent,
                                                                                const edm::EventSetup& iSetup,
                                                                                edm::Handle<SVCollection> svs,
                                                                                edm::Handle<VertexCollection> vtxs) {
  // sort secondary vertices by dxy
  const auto& pv = vtxs->at(0);
  auto svs_sorted = *svs;
  std::sort(svs_sorted.begin(), svs_sorted.end(), [&pv](const auto& sva, const auto& svb) {
    return vertexD3d(sva, pv).value() / vertexD3d(sva, pv).error() >
           vertexD3d(svb, pv).value() / vertexD3d(svb, pv).error();
  });
  return svs_sorted;
}

void PAIReDONNXJetTagsProducer::make_inputs(edm::Event& iEvent,
                                            const edm::EventSetup& iSetup,
                                            edm::Handle<edm::View<pat::Jet>> jets,
                                            unsigned i_jet,
                                            unsigned j_jet,
                                            SVCollection svs,
                                            edm::Handle<edm::View<reco::Candidate>> cands,
                                            edm::Handle<VertexCollection> vtxs,
                                            std::vector<size_t> pf_sorted_idx) {
  // get primary vertex and jet data
  const auto& pv = vtxs->at(0);
  const auto& jet1 = jets->at(i_jet);
  edm::RefToBase<pat::Jet> jet_ref1(jets, i_jet);
  const auto& jet2 = jets->at(j_jet);
  edm::RefToBase<pat::Jet> jet_ref2(jets, j_jet);
  float jet1_pt = jet1.pt() * jet1.jecFactor("Uncorrected");
  float jet2_pt = jet2.pt() * jet2.jecFactor("Uncorrected");
  float jet1_energy = jet1.energy() * jet1.jecFactor("Uncorrected");
  float jet2_energy = jet2.energy() * jet2.jecFactor("Uncorrected");
  // store only indices of the first 128 candidates within the PAIReD ellipse that have a positive PUPPI weight
  std::vector<size_t> pf_idx;
  for (unsigned entry_i = 0; entry_i < pf_sorted_idx.size(); ++entry_i) {
    unsigned entry = pf_sorted_idx[entry_i];
    const reco::Candidate* cand = &(cands->at(entry));
    if (!inEllipse(jet1.eta(), jet1.phi(), jet2.eta(), jet2.phi(), (*cand).eta(), (*cand).phi()))
      continue;
    pf_idx.emplace_back(entry);
    if (pf_idx.size() == max_pf_cands)
      break;
  }
  // loop through stored candidates and save all the input information
  float* ptr = nullptr;  // initialize some pointers used to fill data
  float* start = nullptr;
  for (size_t i_cand = 0; i_cand < max_pf_cands && i_cand < pf_idx.size(); ++i_cand) {
    // get pf cand
    auto entry = pf_idx[i_cand];
    const reco::Candidate* cand = &(cands->at(entry));
    auto packed_cand = dynamic_cast<const pat::PackedCandidate*>(cand);
    // store vector
    std::map<std::string, float> pf_cand_info;
    pf_cand_info["pf_cand_px"] = catch_infs_and_bound(packed_cand->px(), 0, -1e32, 1e32);
    pf_cand_info["pf_cand_py"] = catch_infs_and_bound(packed_cand->py(), 0, -1e32, 1e32);
    pf_cand_info["pf_cand_pz"] = catch_infs_and_bound(packed_cand->pz(), 0, -1e32, 1e32);
    pf_cand_info["pf_cand_e"] = catch_infs_and_bound(packed_cand->energy(), 0, -1e32, 1e32);
    // store features
    pf_cand_info["pf_cand_log_pt"] = catch_infs_and_bound((log(packed_cand->pt()) - 1.7) / 0.7, 0, -5, 5);
    pf_cand_info["pf_cand_log_e"] = catch_infs_and_bound((log(packed_cand->energy()) - 2.0) / 0.7, 0, -5, 5);
    pf_cand_info["pf_cand_log_pt_rel"] =
        catch_infs_and_bound((log(packed_cand->pt() / (jet1_pt + jet2_pt)) + 4.7) / 0.7, 0, -5, 5);
    pf_cand_info["pf_cand_log_e_rel"] =
        catch_infs_and_bound((log(packed_cand->energy() / (jet1_energy + jet2_energy)) + 4.7) / 0.7, 0, -5, 5);
    pf_cand_info["pf_cand_delta_R_1"] = catch_infs_and_bound((reco::deltaR(*packed_cand, jet1) - 0.2) / 4.0, 0, -5, 5);
    pf_cand_info["pf_cand_delta_R_2"] = catch_infs_and_bound((reco::deltaR(*packed_cand, jet2) - 0.2) / 4.0, 0, -5, 5);
    // impact parameter features dependent on charge -> store default values for neutral candidates
    pf_cand_info["pf_cand_charge"] = catch_infs_and_bound(packed_cand->charge(), 0, -1e32, 1e32);
    pf_cand_info["pf_cand_d0"] = std::tanh(-1);
    pf_cand_info["pf_cand_d0_err"] = 0;
    pf_cand_info["pf_cand_dz"] = std::tanh(-1);
    pf_cand_info["pf_cand_dz_err"] = 0;
    if (pf_cand_info["pf_cand_charge"] != 0 && packed_cand->hasTrackDetails()) {
      pf_cand_info["pf_cand_d0"] = catch_infs_and_bound(std::tanh(packed_cand->dxy()), 0, -1e32, 1e32);
      pf_cand_info["pf_cand_d0_err"] = catch_infs_and_bound(packed_cand->dxyError(), 0, 0, 1);
      pf_cand_info["pf_cand_dz"] = catch_infs_and_bound(std::tanh(packed_cand->dz()), 0, -1e32, 1e32);
      pf_cand_info["pf_cand_dz_err"] = catch_infs_and_bound(packed_cand->dzError(), 0, 0, 1);
    }
    // flip relative eta if jet eta is negative
    pf_cand_info["pf_cand_eta_rel_1"] = catch_infs_and_bound((packed_cand->eta() - jet1.eta()), 0, -1e32, 1e32);
    if (jet1.eta() < 0)
      pf_cand_info["pf_cand_eta_rel_1"] *= -1;
    pf_cand_info["pf_cand_phi_rel_1"] =
        catch_infs_and_bound((reco::deltaPhi(packed_cand->phi(), jet1.phi())), 0, -1e32, 1e32);
    pf_cand_info["pf_cand_eta_rel_2"] = catch_infs_and_bound((packed_cand->eta() - jet2.eta()), 0, -1e32, 1e32);
    if (jet2.eta() < 0)
      pf_cand_info["pf_cand_eta_rel_2"] *= -1;
    pf_cand_info["pf_cand_phi_rel_2"] =
        catch_infs_and_bound((reco::deltaPhi(packed_cand->phi(), jet2.phi())), 0, -1e32, 1e32);
    // enter features information into input data
    ptr = &data_[kPfCandFeatures][i_cand];
    start = ptr;
    for (size_t feature_idx = 0; feature_idx < pf_cand_feature_names_.size(); ++feature_idx) {
      *ptr = pf_cand_info[pf_cand_feature_names_[feature_idx]];
      if (feature_idx != pf_cand_feature_names_.size() - 1)
        ptr += max_pf_cands;
    }
    assert(start + (n_features_pf_ - 1) * max_pf_cands == ptr);
    // enter vector information into input data
    ptr = &data_[kPfCandVectors][i_cand];
    start = ptr;
    for (size_t feature_idx = 0; feature_idx < pf_cand_vector_names_.size(); ++feature_idx) {
      *ptr = pf_cand_info[pf_cand_vector_names_[feature_idx]];
      if (feature_idx != pf_cand_vector_names_.size() - 1)
        ptr += max_pf_cands;
    }
    assert(start + (n_features_vector_ - 1) * max_pf_cands == ptr);
    // update mask to be one
    ptr = &data_[kPfCandMasks][i_cand];
    *ptr = 1;
  }
  // loop through secondary vertices
  size_t sv_n = 0;  // index inside paired jet
  for (const auto& sv : svs) {
    // check to make sure sv is in ellipse and passes cut on dl significance (same cut made for SVs in NanoAOD)
    VertexDistance3D vdist;
    Measurement1D dl_cut = vdist.distance(
        vtxs->front(), VertexState(RecoVertex::convertPos(sv.position()), RecoVertex::convertError(sv.error())));
    if (dl_cut.significance() <= 3.0)
      continue;
    if (!inEllipse(jet1.eta(), jet1.phi(), jet2.eta(), jet2.phi(), sv.eta(), sv.phi()))
      continue;
    // save sv info to map starting with vector information
    std::map<std::string, float> sv_info;
    sv_info["sv_px"] = catch_infs_and_bound(sv.px(), 0, -1e32, 1e32);
    sv_info["sv_py"] = catch_infs_and_bound(sv.py(), 0, -1e32, 1e32);
    sv_info["sv_pz"] = catch_infs_and_bound(sv.pz(), 0, -1e32, 1e32);
    sv_info["sv_e"] = catch_infs_and_bound(sv.energy(), 0, -1e32, 1e32);
    // store feature information
    float sv_charge = 0;  // sum over charges of SV daughters
    for (size_t i_sv_daughter = 0; i_sv_daughter < sv.numberOfDaughters(); ++i_sv_daughter) {
      sv_charge += sv.daughter(i_sv_daughter)->charge();
    }
    sv_info["sv_charge"] = catch_infs_and_bound(sv_charge, 0, -1e32, 1e32);
    sv_info["sv_chi2"] = catch_infs_and_bound(sv.vertexNormalizedChi2(), 0, -1e32, 1e32);
    const auto& d3d_meas = vertexD3d(sv, pv);
    sv_info["sv_d3"] = catch_infs_and_bound(d3d_meas.value(), 0, -1e32, 1e32);
    sv_info["sv_d3_sig"] = catch_infs_and_bound(d3d_meas.value() / d3d_meas.error(), 0, -1e32, 1e32);
    const auto& dxy_meas = vertexDxy(sv, pv);
    sv_info["sv_d0"] = catch_infs_and_bound(dxy_meas.value(), 0, -1e32, 1e32);
    sv_info["sv_d0_sig"] = catch_infs_and_bound(dxy_meas.value() / dxy_meas.error(), 0, -1e32, 1e32);
    sv_info["sv_eta"] = catch_infs_and_bound(sv.eta(), 0, -1e32, 1e32);
    sv_info["sv_mass"] = catch_infs_and_bound(sv.mass(), 0, -1e32, 1e32);
    sv_info["sv_ndof"] = catch_infs_and_bound(sv.vertexNdof(), 0, -1e32, 1e32);
    sv_info["sv_ntracks"] = catch_infs_and_bound(sv.numberOfDaughters(), 0, -1e32, 1e32);
    sv_info["sv_p_angle"] = catch_infs_and_bound(std::acos(-1 * vertexDdotP(sv, pv)), 0, -1e32, 1e32);
    sv_info["sv_phi"] = catch_infs_and_bound(sv.phi(), 0, -1e32, 1e32);
    sv_info["sv_pt"] = catch_infs_and_bound(sv.pt(), 0, -1e32, 1e32);
    sv_info["sv_x"] = catch_infs_and_bound(sv.position().x(), 0, -1e32, 1e32);
    sv_info["sv_y"] = catch_infs_and_bound(sv.position().y(), 0, -1e32, 1e32);
    sv_info["sv_z"] = catch_infs_and_bound(sv.position().z(), 0, -1e32, 1e32);
    // flip relative eta if jet eta is negtive
    sv_info["sv_eta_rel_1"] = catch_infs_and_bound((sv.eta() - jet1.eta()), 0, -1e32, 1e32);
    sv_info["sv_eta_rel_2"] = catch_infs_and_bound((sv.eta() - jet2.eta()), 0, -1e32, 1e32);
    if (jet1.eta() < 0)
      sv_info["sv_eta_rel_1"] *= -1;
    if (jet2.eta() < 0)
      sv_info["sv_eta_rel_2"] *= -1;
    sv_info["sv_phi_rel_1"] = catch_infs_and_bound((reco::deltaPhi(sv.phi(), jet1.phi())), 0, -1e32, 1e32);
    sv_info["sv_phi_rel_2"] = catch_infs_and_bound((reco::deltaPhi(sv.phi(), jet2.phi())), 0, -1e32, 1e32);
    // save secondary vertex features to input data array
    ptr = &data_[kSVFeatures][sv_n];
    start = ptr;
    for (size_t feature_idx = 0; feature_idx < sv_feature_names_.size(); ++feature_idx) {
      *ptr = sv_info[sv_feature_names_[feature_idx]];
      if (feature_idx != sv_feature_names_.size() - 1)
        ptr += max_svs;
    }
    assert(start + (n_features_sv_ - 1) * max_svs == ptr);
    // save secondary vertex vectors to input data array
    ptr = &data_[kSVVectors][sv_n];
    start = ptr;
    for (size_t feature_idx = 0; feature_idx < sv_vector_names_.size(); ++feature_idx) {
      *ptr = sv_info[sv_vector_names_[feature_idx]];
      if (feature_idx != sv_vector_names_.size() - 1)
        ptr += max_svs;
    }
    assert(start + (n_features_vector_ - 1) * max_svs == ptr);
    // update mask and sv index
    ptr = &data_[kSVMasks][sv_n];
    *ptr = 1;
    ++sv_n;
  }
}

int PAIReDONNXJetTagsProducer::get_n_parton(edm::Event& iEvent,
                                            const edm::EventSetup& iSetup,
                                            edm::Handle<edm::View<pat::Jet>> jets,
                                            unsigned i_jet,
                                            unsigned j_jet,
                                            edm::Handle<edm::View<reco::GenParticle>> gen_particles,
                                            int parton_id) {
  const auto& jet1 = jets->at(i_jet);
  const auto& jet2 = jets->at(j_jet);
  // using hadron ghost tagging as the truth information
  int n_parton = 0;
  for (unsigned i = 0; i < gen_particles->size(); ++i) {
    const auto* genp = &(gen_particles->at(i));
    if (genp->isLastCopy() && inEllipse(jet1.eta(), jet1.phi(), jet2.eta(), jet2.phi(), (*genp).eta(), (*genp).phi())) {
      if (parton_id == 4)
        n_parton += Rivet::PID::hasCharm(genp->pdgId());
      else if (parton_id == 5)
        n_parton += Rivet::PID::hasBottom(genp->pdgId());
    }
  }
  return n_parton;
}

DEFINE_FWK_MODULE(PAIReDONNXJetTagsProducer);
