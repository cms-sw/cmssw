#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

#include "RecoBTag/ONNXRuntime/interface/tensor_fillers.h"
#include "RecoBTag/ONNXRuntime/interface/tensor_configs.h"

using namespace cms::Ort;

class DeepVertexONNXJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit DeepVertexONNXJetTagsProducer(const edm::ParameterSet&, const ONNXRuntime*);
  ~DeepVertexONNXJetTagsProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const ONNXRuntime*);

private:
  typedef std::vector<reco::DeepFlavourTagInfo> TagInfoCollection;
  typedef reco::JetTagCollection JetTagCollection;

  void produce(edm::Event&, const edm::EventSetup&) override;

  void make_inputs(unsigned i_jet, const reco::DeepFlavourTagInfo& taginfo);

  const edm::EDGetTokenT<TagInfoCollection> src_;
  std::vector<std::string> flav_names_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  const double min_jet_pt_;
  const double max_jet_eta_;

  enum InputIndexes { kGlobal = 0, kSeedingTracks = 1, kNeighbourTracks = 2 };
  const static unsigned n_features_global_ = deepvertex::n_features_global;
  const static unsigned n_seed_ = deepvertex::n_seed;
  const static unsigned n_features_seed_ = deepvertex::n_features_seed;
  const static unsigned n_neighbor_ = deepvertex::n_neighbor;
  const static unsigned n_features_neighbor_ = deepvertex::n_features_neighbor;

  const static std::vector<unsigned> input_sizes_;

  // hold the input data
  FloatArrays data_;
};

const std::vector<unsigned> DeepVertexONNXJetTagsProducer::input_sizes_{n_features_global_,
                                                                        n_seed_* n_features_seed_,
                                                                        n_neighbor_* n_features_neighbor_,
                                                                        n_neighbor_* n_features_neighbor_,
                                                                        n_neighbor_* n_features_neighbor_,
                                                                        n_neighbor_* n_features_neighbor_,
                                                                        n_neighbor_* n_features_neighbor_,
                                                                        n_neighbor_* n_features_neighbor_,
                                                                        n_neighbor_* n_features_neighbor_,
                                                                        n_neighbor_* n_features_neighbor_,
                                                                        n_neighbor_* n_features_neighbor_,
                                                                        n_neighbor_* n_features_neighbor_};

DeepVertexONNXJetTagsProducer::DeepVertexONNXJetTagsProducer(const edm::ParameterSet& iConfig, const ONNXRuntime* cache)
    : src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      flav_names_(iConfig.getParameter<std::vector<std::string>>("flav_names")),
      input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")),
      min_jet_pt_(iConfig.getParameter<double>("min_jet_pt")),
      max_jet_eta_(iConfig.getParameter<double>("max_jet_eta")) {
  // get output names from flav_names
  for (const auto& flav_name : flav_names_) {
    produces<JetTagCollection>(flav_name);
  }

  assert(input_names_.size() == input_sizes_.size());
}

DeepVertexONNXJetTagsProducer::~DeepVertexONNXJetTagsProducer() {}

void DeepVertexONNXJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfDeepFlavourJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepFlavourTagInfos"));
  desc.add<std::vector<std::string>>("input_names",
                                     {"input_1",
                                      "input_2",
                                      "input_3",
                                      "input_4",
                                      "input_5",
                                      "input_6",
                                      "input_7",
                                      "input_8",
                                      "input_9",
                                      "input_10",
                                      "input_11",
                                      "input_12"});
  desc.add<edm::FileInPath>("model_path", edm::FileInPath("RecoBTag/Combined/data/DeepVertex/phase1_deepvertex.onnx"));
  desc.add<std::vector<std::string>>("output_names", {"dense_6"});
  desc.add<std::vector<std::string>>("flav_names", std::vector<std::string>{"probb", "probc", "probuds", "probg"});
  desc.add<double>("min_jet_pt", 15.0);
  desc.add<double>("max_jet_eta", 2.5);

  descriptions.add("pfDeepVertexJetTags", desc);
}

std::unique_ptr<ONNXRuntime> DeepVertexONNXJetTagsProducer::initializeGlobalCache(const edm::ParameterSet& iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void DeepVertexONNXJetTagsProducer::globalEndJob(const ONNXRuntime* cache) {}

void DeepVertexONNXJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);

  data_.clear();

  std::vector<std::unique_ptr<JetTagCollection>> output_tags;
  if (!tag_infos->empty()) {
    unsigned good_taginfo_count = 0;
    std::vector<bool> good_taginfo_jets(tag_infos->size(), false);
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto& jet_ref = (*tag_infos)[jet_n].jet();
      if (jet_ref->pt() > min_jet_pt_ && std::fabs(jet_ref->eta()) < max_jet_eta_) {
        good_taginfo_count++;
        good_taginfo_jets[jet_n] = true;
      }
    }

    // init data storage w correct size
    for (const auto& len : input_sizes_) {
      data_.emplace_back(good_taginfo_count * len, 0);
    }

    // initialize output collection
    auto jet_ref = tag_infos->begin()->jet();
    auto ref2prod = edm::makeRefToBaseProdFrom(jet_ref, iEvent);
    for (std::size_t i = 0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>(ref2prod));
    }

    // convert inputs
    unsigned inputs_done_count = 0;
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      if (good_taginfo_jets[jet_n]) {
        const auto& taginfo = (*tag_infos)[jet_n];
        make_inputs(inputs_done_count, taginfo);
        inputs_done_count++;
      }
    }

    // run prediction
    assert(inputs_done_count == good_taginfo_count);
    const auto outputs = globalCache()->run(input_names_, data_, {}, output_names_, good_taginfo_count)[0];
    assert(outputs.size() == flav_names_.size() * good_taginfo_count);

    // get the outputs
    unsigned i_output = 0;
    for (unsigned jet_n = 0; jet_n < tag_infos->size(); ++jet_n) {
      const auto& jet_ref = (*tag_infos)[jet_n].jet();
      for (std::size_t flav_n = 0; flav_n < flav_names_.size(); flav_n++) {
        if (good_taginfo_jets[jet_n]) {
          (*(output_tags[flav_n]))[jet_ref] = outputs[i_output];
          ++i_output;
        } else {
          (*(output_tags[flav_n]))[jet_ref] = -2;
        }
      }
    }
  } else {
    // create empty output collection
    for (std::size_t i = 0; i < flav_names_.size(); i++) {
      output_tags.emplace_back(std::make_unique<JetTagCollection>());
    }
  }

  // put into the event
  for (std::size_t flav_n = 0; flav_n < flav_names_.size(); ++flav_n) {
    iEvent.put(std::move(output_tags[flav_n]), flav_names_[flav_n]);
  }
}

void DeepVertexONNXJetTagsProducer::make_inputs(unsigned i_jet, const reco::DeepFlavourTagInfo& taginfo) {
  const auto& features = taginfo.features();
  float* ptr = nullptr;
  const float* start = nullptr;
  unsigned offset = 0;

  // jet variables
  offset = i_jet * input_sizes_[kGlobal];
  const auto& jet_features = features.jet_features;
  ptr = &data_[kGlobal][offset];
  start = ptr;
  jet4vec_tensor_filler(ptr, jet_features);
  assert(start + n_features_global_ - 1 == ptr);

  // seeds
  auto max_seed_n = std::min(features.seed_features.size(), (std::size_t)n_seed_);
  offset = i_jet * input_sizes_[kSeedingTracks];
  for (std::size_t seed_n = 0; seed_n < max_seed_n; seed_n++) {
    const auto& seed_features = features.seed_features[seed_n];
    ptr = &data_[kSeedingTracks][offset + seed_n * n_features_seed_];
    start = ptr;
    seedTrack_tensor_filler(ptr, seed_features);
    assert(start + n_features_seed_ - 1 == ptr);
  }

  // neighbours
  offset = i_jet * input_sizes_[kNeighbourTracks];
  for (std::size_t seed_n = 0; seed_n < max_seed_n; seed_n++) {
    const auto& neighbourTracks_features = features.seed_features[seed_n].nearTracks;
    auto max_neighbour_n = std::min(neighbourTracks_features.size(), (std::size_t)n_neighbor_);
    for (std::size_t neighbour_n = 0; neighbour_n < max_neighbour_n; neighbour_n++) {
      ptr = &data_[kNeighbourTracks + seed_n][offset + neighbour_n * n_features_neighbor_];
      start = ptr;
      neighbourTrack_tensor_filler(ptr, neighbourTracks_features[neighbour_n]);
      assert(start + n_features_neighbor_ - 1 == ptr);
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepVertexONNXJetTagsProducer);
