
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/DeepFormats/interface/DeepFlavourTagInfo.h"

#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "RecoBTag/DeepFlavour/interface/tensor_fillers.h"

// Declaration of the data structure that is hold by the edm::GlobalCache.
// In TensorFlow, the computational graph is stored in a stateless meta graph object which can be
// shared by multiple session instances which handle the initialization of variables related to the
// meta graph. Following this approach in CMSSW, a meta graph should be stored in a GlobalCache
// which can be accesses by sessions owned by multiple stream module copies. Instead of using only
// the plain meta graph, we make use of a Cache struct that can be extended in the future if nedded.
// In addition, the meta graph is protected via std::atomic, which should not affect the performance
// as it is only accessed in the module constructor and not in the actual produce loop.
struct Cache {
  Cache() : metaGraph(nullptr)
  {
  }

  std::atomic<tf::MetaGraphDef*> metaGraph;
};

class DeepFlavourTFJetTagsProducer : public edm::stream::EDProducer<edm::GlobalCache<Cache>> {

  public:
    explicit DeepFlavourTFJetTagsProducer(const edm::ParameterSet&, const Cache*);
    ~DeepFlavourTFJetTagsProducer() override;

    static void fillDescriptions(edm::ConfigurationDescriptions&);

    static std::unique_ptr<Cache> initializeGlobalCache(const edm::ParameterSet&);
    static void globalEndJob(const Cache*);

  private:
    typedef std::vector<reco::DeepFlavourTagInfo> TagInfoCollection;
    typedef reco::JetTagCollection JetTagCollection;

    void beginStream(edm::StreamID) override {}
    void produce(edm::Event&, const edm::EventSetup&) override;
    void endStream() override {}

    const edm::EDGetTokenT< TagInfoCollection > src_;
    std::vector<std::pair<std::string,std::vector<unsigned int>>> flav_pairs_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::string> lp_names_;

    // session for TF evaluation
    tf::Session* session_;

    // combined names of all input tensors for faster evaluation
    std::vector<std::string> all_input_names_;

    // vector of learning phase tensors, i.e., boolean scalar tensors pointing to false
    std::vector<tf::Tensor> lp_tensors_;
};

DeepFlavourTFJetTagsProducer::DeepFlavourTFJetTagsProducer(const edm::ParameterSet& iConfig, const Cache* cache) :
  src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
  input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
  output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")),
  lp_names_(iConfig.getParameter<std::vector<std::string>>("lp_names")),
  session_(nullptr)
{
  // create the session using the meta graph from the cache
  edm::FileInPath graphPath(iConfig.getParameter<edm::FileInPath>("graph_path"));
  std::string exportDir = graphPath.fullPath().substr(0, graphPath.fullPath().find_last_of("/"));
  session_ = tf::createSession(cache->metaGraph, exportDir);

  // get output names from flav_table
  const auto & flav_pset = iConfig.getParameter<edm::ParameterSet>("flav_table");
  for (const auto flav_pair : flav_pset.tbl()) {
    const auto & flav_name = flav_pair.first;
    flav_pairs_.emplace_back(flav_name,
                             flav_pset.getParameter<std::vector<unsigned int>>(flav_name));
  }

  for (const auto & flav_pair : flav_pairs_) {
    produces<JetTagCollection>(flav_pair.first);
  }

  // flag inputs (required because of batch norm)
  // names for the learing phase placeholders (to init and set as false)
  for (size_t i = 0; i < lp_names_.size(); i++) {
    // create a bool tensor, set its value to false and store it
    tf::Tensor t(tf::DT_BOOL, {});
    t.scalar<bool>()() = false;
    lp_tensors_.push_back(t);
  }

  // store combined input tensor names
  all_input_names_ = lp_names_;
  all_input_names_.insert(all_input_names_.end(), input_names_.begin(), input_names_.end());
}

DeepFlavourTFJetTagsProducer::~DeepFlavourTFJetTagsProducer()
{
  // close and delete the session
  if (session_ != nullptr) {
    tf::closeSession(session_);
  }
}

void DeepFlavourTFJetTagsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{

  // pfDeepFlavourJetTags
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("pfDeepFlavourTagInfos"));
  desc.add<std::vector<std::string>>("input_names", 
    { "input_1", "input_2", "input_3", "input_4", "input_5" });
  desc.add<edm::FileInPath>("graph_path",
    edm::FileInPath("RecoBTag/Combined/data/DeepFlavourV01_C/saved_model.pb"));
  desc.add<std::vector<std::string>>("lp_names",
    {"cpf_input_batchnorm/keras_learning_phase"});
  desc.add<std::vector<std::string>>("output_names",
    { "ID_pred/Softmax", "regression_pred/BiasAdd", });
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::vector<unsigned int>>("probb", {0});
    psd0.add<std::vector<unsigned int>>("probbb", {1});
    psd0.add<std::vector<unsigned int>>("problepb", {2});
    psd0.add<std::vector<unsigned int>>("probc", {3});
    psd0.add<std::vector<unsigned int>>("probuds", {4});
    psd0.add<std::vector<unsigned int>>("probg", {5});
    desc.add<edm::ParameterSetDescription>("flav_table", psd0);
  }
  descriptions.add("pfDeepFlavourJetTags", desc);
}

std::unique_ptr<Cache> DeepFlavourTFJetTagsProducer::initializeGlobalCache(const edm::ParameterSet& iConfig)
{
  // set the tensorflow log level to error
  tf::setLogging("3");

  // build the exportDir from graph_path
  edm::FileInPath graphPath(iConfig.getParameter<edm::FileInPath>("graph_path"));
  std::string exportDir = graphPath.fullPath().substr(0, graphPath.fullPath().find_last_of("/"));

  // create the cache instance and attach the meta graph to it
  Cache* cache = new Cache();
  cache->metaGraph = tf::loadMetaGraph(exportDir);

  return std::unique_ptr<Cache>(cache);
}

void DeepFlavourTFJetTagsProducer::globalEndJob(const Cache* cache)
{
  if (cache->metaGraph != nullptr) {
    delete cache->metaGraph;
  }
}

void DeepFlavourTFJetTagsProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);

  std::vector<std::unique_ptr<JetTagCollection>> output_tags;

  int64_t n_jets = tag_infos->size();
  std::vector<tf::TensorShape> input_sizes {
    {n_jets, 15},         // input_1 - global jet features
    {n_jets, 25, 16},     // input_2 - charged pf
    {n_jets, 25, 6},      // input_3 - neutral pf
    {n_jets, 4, 12},      // input_4 - vertices 
    {n_jets, 1}           // input_5 - jet pt for reg 
  };

  // create input tensors based on the input_sizes in this event
  std::vector<tf::Tensor> input_tensors;
  for (std::size_t i=0; i < input_sizes.size(); i++) {
    tf::Tensor t(tf::DT_FLOAT, input_sizes.at(i));
    t.flat<float>().setZero();
    input_tensors.push_back(t);
  }

  // fill values
  for (std::size_t jet_n=0; jet_n < tag_infos->size(); jet_n++) {

    // jet and other global features
    const auto & features = tag_infos->at(jet_n).features();
    jet_tensor_filler(input_tensors.at(0), jet_n, features);

    // c_pf candidates
    auto max_c_pf_n = std::min(features.c_pf_features.size(), (std::size_t) input_sizes.at(1).dim_size(1));
    for (std::size_t c_pf_n=0; c_pf_n < max_c_pf_n; c_pf_n++) {
      const auto & c_pf_features = features.c_pf_features.at(c_pf_n);
      c_pf_tensor_filler(input_tensors.at(1), jet_n, c_pf_n, c_pf_features);
    }

    // n_pf candidates
    auto max_n_pf_n = std::min(features.n_pf_features.size(), (std::size_t)input_sizes.at(2).dim_size(1));
    for (std::size_t n_pf_n=0; n_pf_n < max_n_pf_n; n_pf_n++) {
      const auto & n_pf_features = features.n_pf_features.at(n_pf_n);
      n_pf_tensor_filler(input_tensors.at(2), jet_n, n_pf_n, n_pf_features);
    }

    // sv candidates
    auto max_sv_n = std::min(features.sv_features.size(), (std::size_t)input_sizes.at(3).dim_size(1));
    for (std::size_t sv_n=0; sv_n < max_sv_n; sv_n++) {
      const auto & sv_features = features.sv_features.at(sv_n);
      sv_tensor_filler(input_tensors.at(3), jet_n, sv_n, sv_features);
    }

    // last input: jet pt
    input_tensors.at(4).matrix<float>()(jet_n, 0) = features.jet_features.pt;
  }

  // run the session
  input_tensors.insert(input_tensors.begin(), lp_tensors_.begin(), lp_tensors_.end());
  std::vector<tf::Tensor> outputs;
  tf::run(session_, all_input_names_, input_tensors, output_names_, &outputs);

  // create output collection
  for (std::size_t i=0; i < flav_pairs_.size(); i++) {
    if (!tag_infos->empty()) {
      auto jet_ref = tag_infos->begin()->jet();
      output_tags.emplace_back(std::make_unique<JetTagCollection>(
            edm::makeRefToBaseProdFrom(jet_ref, iEvent)));
    } else {
      output_tags.emplace_back(std::make_unique<JetTagCollection>());
    }
  }

  // set output values for flavour probs
  for (std::size_t jet_n=0; jet_n < tag_infos->size(); jet_n++) {
    const auto & jet_ref = tag_infos->at(jet_n).jet();
    for (std::size_t flav_n=0; flav_n < flav_pairs_.size(); flav_n++) {
      const auto & flav_pair = flav_pairs_.at(flav_n);
      float o_sum = 0.;
      for (const unsigned int & ind : flav_pair.second) {
        o_sum += outputs.at(0).matrix<float>()(jet_n, ind);
      }
      (*(output_tags.at(flav_n)))[jet_ref] = o_sum;
    }
  }

  for (std::size_t i=0; i < flav_pairs_.size(); i++) {
    iEvent.put(std::move(output_tags[i]), flav_pairs_.at(i).first);
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourTFJetTagsProducer);
