#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "RecoMET/METPUSubtraction/interface/DeepMETHelp.h"

using namespace deepmet_helper;

class DeepMETProducer : public edm::stream::EDProducer<edm::GlobalCache<tensorflow::SessionCache> > {
public:
  explicit DeepMETProducer(const edm::ParameterSet&, const tensorflow::SessionCache*);
  void produce(edm::Event& event, const edm::EventSetup& setup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  // static methods for handling the global cache
  static std::unique_ptr<tensorflow::SessionCache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(tensorflow::SessionCache*){};

private:
  const edm::EDGetTokenT<std::vector<pat::PackedCandidate> > pf_token_;
  const float norm_;
  const bool ignore_leptons_;
  const unsigned int max_n_pf_;

  const tensorflow::Session* session_;

  tensorflow::Tensor input_;
  tensorflow::Tensor input_cat0_;
  tensorflow::Tensor input_cat1_;
  tensorflow::Tensor input_cat2_;
};

DeepMETProducer::DeepMETProducer(const edm::ParameterSet& cfg, const tensorflow::SessionCache* cache)
    : pf_token_(consumes<std::vector<pat::PackedCandidate> >(cfg.getParameter<edm::InputTag>("pf_src"))),
      norm_(cfg.getParameter<double>("norm_factor")),
      ignore_leptons_(cfg.getParameter<bool>("ignore_leptons")),
      max_n_pf_(cfg.getParameter<unsigned int>("max_n_pf")),
      session_(cache->getSession()) {
  produces<pat::METCollection>();

  const tensorflow::TensorShape shape({1, max_n_pf_, 8});
  const tensorflow::TensorShape cat_shape({1, max_n_pf_, 1});

  input_ = tensorflow::Tensor(tensorflow::DT_FLOAT, shape);
  input_cat0_ = tensorflow::Tensor(tensorflow::DT_FLOAT, cat_shape);
  input_cat1_ = tensorflow::Tensor(tensorflow::DT_FLOAT, cat_shape);
  input_cat2_ = tensorflow::Tensor(tensorflow::DT_FLOAT, cat_shape);
}

void DeepMETProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  auto const& pfs = event.get(pf_token_);

  const tensorflow::NamedTensorList input_list = {
      {"input", input_}, {"input_cat0", input_cat0_}, {"input_cat1", input_cat1_}, {"input_cat2", input_cat2_}};

  // Set all inputs to zero
  input_.flat<float>().setZero();
  input_cat0_.flat<float>().setZero();
  input_cat1_.flat<float>().setZero();
  input_cat2_.flat<float>().setZero();

  size_t i_pf = 0;
  float px_leptons = 0.;
  float py_leptons = 0.;
  const float scale = 1. / norm_;
  for (const auto& pf : pfs) {
    if (ignore_leptons_) {
      int pdg_id = std::abs(pf.pdgId());
      if (pdg_id == 11 || pdg_id == 13) {
        px_leptons += pf.px();
        py_leptons += pf.py();
        continue;
      }
    }

    // fill the tensor
    // PF keys [b'PF_dxy', b'PF_dz', b'PF_eta', b'PF_mass', b'PF_pt', b'PF_puppiWeight', b'PF_px', b'PF_py']
    float* ptr = &input_.tensor<float, 3>()(0, i_pf, 0);
    *ptr = rm_outlier(pf.dxy());
    *(++ptr) = rm_outlier(pf.dz());
    *(++ptr) = rm_outlier(pf.eta());
    *(++ptr) = rm_outlier(pf.mass());
    *(++ptr) = scale_and_rm_outlier(pf.pt(), scale);
    *(++ptr) = rm_outlier(pf.puppiWeight());
    *(++ptr) = scale_and_rm_outlier(pf.px(), scale);
    *(++ptr) = scale_and_rm_outlier(pf.py(), scale);
    input_cat0_.tensor<float, 3>()(0, i_pf, 0) = charge_embedding.at(pf.charge());
    input_cat1_.tensor<float, 3>()(0, i_pf, 0) = pdg_id_embedding.at(pf.pdgId());
    input_cat2_.tensor<float, 3>()(0, i_pf, 0) = pf.fromPV();

    ++i_pf;
    if (i_pf == max_n_pf_) {
      break;  // output a warning?
    }
  }

  std::vector<tensorflow::Tensor> outputs;
  const std::vector<std::string> output_names = {"output/BiasAdd"};

  // run the inference and return met
  tensorflow::run(session_, input_list, output_names, &outputs);

  // The DNN directly estimates the missing px and py, not the recoil
  float px = outputs[0].tensor<float, 2>()(0, 0) * norm_;
  float py = outputs[0].tensor<float, 2>()(0, 1) * norm_;

  px -= px_leptons;
  py -= py_leptons;

  LogDebug("produce") << "<DeepMETProducer::produce>:" << std::endl
                      << " MET from DeepMET Producer is MET_x " << px << " and MET_y " << py << std::endl;

  auto pf_mets = std::make_unique<pat::METCollection>();
  const reco::Candidate::LorentzVector p4(px, py, 0., std::hypot(px, py));
  pf_mets->emplace_back(reco::MET(p4, {}));
  event.put(std::move(pf_mets));
}

std::unique_ptr<tensorflow::SessionCache> DeepMETProducer::initializeGlobalCache(const edm::ParameterSet& params) {
  // this method is supposed to create, initialize and return a SessionCache instance
  std::string graphPath = edm::FileInPath(params.getParameter<std::string>("graph_path")).fullPath();
  return std::make_unique<tensorflow::SessionCache>(graphPath);
}

void DeepMETProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pf_src", edm::InputTag("packedPFCandidates"));
  desc.add<bool>("ignore_leptons", false);
  desc.add<double>("norm_factor", 50.);
  desc.add<unsigned int>("max_n_pf", 4500);
  desc.add<std::string>("graph_path", "RecoMET/METPUSubtraction/data/models/deepmet/deepmet_v1_2018/model.graphdef");
  descriptions.add("deepMETProducer", desc);
}

DEFINE_FWK_MODULE(DeepMETProducer);
