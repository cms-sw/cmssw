#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

struct DeepMETCache {
  std::atomic<tensorflow::GraphDef*> graph_def;
};

class DeepMETProducer : public edm::stream::EDProducer<edm::GlobalCache<DeepMETCache> > {
public:
  explicit DeepMETProducer(const edm::ParameterSet&, const DeepMETCache*);
  void produce(edm::Event& event, const edm::EventSetup& setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // static methods for handling the global cache
  static std::unique_ptr<DeepMETCache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(DeepMETCache*);

private:
  const edm::EDGetTokenT<std::vector<pat::PackedCandidate> > pf_token_;
  float norm_;
  bool ignore_leptons_;
  unsigned int max_n_pf_;

  tensorflow::Session* session_;

  std::unordered_map<int, int32_t> charge_embedding_;
  std::unordered_map<int, int32_t> pdg_id_embedding_;
};

namespace {
  float divide_and_rm_outlier(float val, float norm) {
    float ret_val = val / norm;
    if (ret_val > 1e6 || ret_val < -1e6)
      return 0.;
    return ret_val;
  }
}  // namespace

DeepMETProducer::DeepMETProducer(const edm::ParameterSet& cfg, const DeepMETCache* cache)
    : pf_token_(consumes<std::vector<pat::PackedCandidate> >(cfg.getParameter<edm::InputTag>("pf_src"))),
      norm_(cfg.getParameter<double>("norm_factor")),
      ignore_leptons_(cfg.getParameter<bool>("ignore_leptons")),
      max_n_pf_(cfg.getParameter<unsigned int>("max_n_pf")) {
  session_ = tensorflow::createSession(cache->graph_def);
  produces<pat::METCollection>();
  charge_embedding_ = {{-1, 0}, {0, 1}, {1, 2}};
  pdg_id_embedding_ = {
      {-211, 0}, {-13, 1}, {-11, 2}, {0, 3}, {1, 4}, {2, 5}, {11, 6}, {13, 7}, {22, 8}, {130, 9}, {211, 10}};
  ;
}

void DeepMETProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  edm::Handle<std::vector<pat::PackedCandidate> > pf_h;
  event.getByToken(pf_token_, pf_h);

  // PF keys [b'PF_dxy', b'PF_dz', b'PF_eta', b'PF_mass', b'PF_pt', b'PF_puppiWeight', b'PF_px', b'PF_py']
  tensorflow::TensorShape shape({1, max_n_pf_, 8});
  tensorflow::TensorShape cat_shape({1, max_n_pf_, 1});
  tensorflow::Tensor input(tensorflow::DT_FLOAT, shape);
  tensorflow::Tensor input_cat0(tensorflow::DT_FLOAT, cat_shape);
  tensorflow::Tensor input_cat1(tensorflow::DT_FLOAT, cat_shape);
  tensorflow::Tensor input_cat2(tensorflow::DT_FLOAT, cat_shape);

  tensorflow::TensorShape out_shape({1, 2});
  tensorflow::Tensor output(tensorflow::DT_FLOAT, out_shape);

  tensorflow::NamedTensorList input_list = {
      {"input", input}, {"input_cat0", input_cat0}, {"input_cat1", input_cat1}, {"input_cat2", input_cat2}};

  // Set all inputs to zero
  input.flat<float>().setZero();
  input_cat0.flat<float>().setZero();
  input_cat1.flat<float>().setZero();
  input_cat2.flat<float>().setZero();

  size_t i_pf = 0;
  float px_leptons = 0.;
  float py_leptons = 0.;
  for (const auto& pf : *pf_h) {
    if (ignore_leptons_) {
      int pdg_id = std::abs(pf.pdgId());
      if (pdg_id == 11 || pdg_id == 13) {
        px_leptons += pf.px();
        py_leptons += pf.py();
        continue;
      }
    }

    // fill the tensor
    float* ptr = &input.tensor<float, 3>()(0, i_pf, 0);
    *ptr = pf.dxy();
    *(++ptr) = pf.dz();
    *(++ptr) = pf.eta();
    *(++ptr) = pf.mass();
    *(++ptr) = divide_and_rm_outlier(pf.pt(), norm_);
    *(++ptr) = pf.puppiWeight();
    *(++ptr) = divide_and_rm_outlier(pf.px(), norm_);
    *(++ptr) = divide_and_rm_outlier(pf.py(), norm_);
    input_cat0.tensor<float, 3>()(0, i_pf, 0) = charge_embedding_[pf.charge()];
    input_cat1.tensor<float, 3>()(0, i_pf, 0) = pdg_id_embedding_[pf.pdgId()];
    input_cat2.tensor<float, 3>()(0, i_pf, 0) = pf.fromPV();

    ++i_pf;
    if (i_pf > max_n_pf_) {
      break;  // output a warning?
    }
  }

  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> output_names = {"output/BiasAdd"};

  // run the inference and return met
  tensorflow::run(session_, input_list, output_names, &outputs);

  // The DNN directly estimates the missing px and py, not the recoil
  float px = outputs[0].tensor<float, 2>()(0, 0) * norm_;
  float py = outputs[0].tensor<float, 2>()(0, 1) * norm_;

  px -= px_leptons;
  py -= py_leptons;

  auto pf_mets = std::make_unique<pat::METCollection>();
  reco::LeafCandidate::LorentzVector p4(px, py, 0., std::sqrt(px * px + py * py));
  const reco::Candidate::Point vtx(0.0, 0.0, 0.0);
  pf_mets->emplace_back(reco::MET(p4, vtx));
  event.put(std::move(pf_mets));
}

std::unique_ptr<DeepMETCache> DeepMETProducer::initializeGlobalCache(const edm::ParameterSet& params) {
  // this method is supposed to create, initialize and return a DeepMETCache instance
  std::unique_ptr<DeepMETCache> cache = std::make_unique<DeepMETCache>();

  // load the graph def and save it
  std::string graphPath = params.getParameter<std::string>("graph_path");
  if (!graphPath.empty()) {
    graphPath = edm::FileInPath(graphPath).fullPath();
    cache->graph_def = tensorflow::loadGraphDef(graphPath);
  }

  return cache;
}

void DeepMETProducer::globalEndJob(DeepMETCache* cache) { delete cache->graph_def; }

void DeepMETProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pf_src", edm::InputTag("packedPFCandidates"));
  desc.add<bool>("ignore_leptons", false);
  desc.add<double>("norm_factor", 50.);
  desc.add<unsigned int>("max_n_pf", 4500);
  desc.add<std::string>("graph_path", "RecoMET/METPUSubtraction/data/deepmet/deepmet_v1_2018.pb");
  descriptions.add("deepMETProducer", desc);
}

DEFINE_FWK_MODULE(DeepMETProducer);
