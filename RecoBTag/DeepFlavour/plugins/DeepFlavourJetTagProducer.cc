
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/DeepFormats/interface/DeepFlavourTagInfo.h"

#include "DNN/Tensorflow/interface/Graph.h"
#include "DNN/Tensorflow/interface/Tensor.h"

#include "RecoBTag/DeepFlavour/interface/tensor_fillers.h"



class DeepFlavourJetTagProducer : public edm::stream::EDProducer<> {

  public:
	  explicit DeepFlavourJetTagProducer(const edm::ParameterSet&);
	  ~DeepFlavourJetTagProducer();

	  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:

    typedef std::vector<reco::DeepFlavourTagInfo> TagInfoCollection;
    typedef reco::JetTagCollection JetTagCollection;

	  virtual void beginStream(edm::StreamID) override {}
	  virtual void produce(edm::Event&, const edm::EventSetup&) override;
	  virtual void endStream() override {}

	  const edm::EDGetTokenT< TagInfoCollection > src_;
    std::string graph_path_;
    std::vector<std::string> outputs_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    // graph for TF evaluation
    dnn::tf::Graph graph_;
    // not owing vector of pointers for inputs and outputs
    std::vector<dnn::tf::Tensor *> dnn_inputs_;
    std::vector<dnn::tf::Tensor *> dnn_outputs_;


};

DeepFlavourJetTagProducer::DeepFlavourJetTagProducer(const edm::ParameterSet& iConfig) :
  src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
  graph_path_(iConfig.getParameter<std::string>("graph_path")),
  outputs_(iConfig.getParameter<std::vector<std::string>>("outputs")),
  input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
  output_names_(iConfig.getParameter<std::vector<std::string>>("output_names")),
  graph_(graph_path_)
{

  for (const auto & output : outputs_) {
    produces<JetTagCollection>(output);
  }

  for (const auto & input_name : input_names_) {
    dnn_inputs_.emplace_back(graph_.defineInput(new dnn::tf::Tensor(input_name)));
  }

  // required because of batch norm
  auto learning_phase_name = "cpf_batchnorm0/keras_learning_phase:0";
  auto learning_phase = graph_.defineInput(new dnn::tf::Tensor(learning_phase_name));
  learning_phase->setArray(0, nullptr);
  learning_phase->setValue<bool>(false);

  for (const auto & output_name : output_names_) {
    dnn_outputs_.emplace_back(graph_.defineOutput(new dnn::tf::Tensor(output_name)));
  }


  

}


DeepFlavourJetTagProducer::~DeepFlavourJetTagProducer()
{
}

void DeepFlavourJetTagProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
}

void DeepFlavourJetTagProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<TagInfoCollection> tag_infos;
  iEvent.getByToken(src_, tag_infos);

  std::vector<std::unique_ptr<JetTagCollection>> output_tags;

  auto n_jets = dnn::tf::Shape(tag_infos->size());
  std::vector<std::vector<dnn::tf::Shape>> input_sizes {
    {n_jets, 15},         // input_1 - global jet features
    {n_jets, 25, 17},     // input_2 - charged pf
    {n_jets, 25, 6},      // input_3 - neutral pf
    {n_jets, 4, 12},      // input_4 - vertices 
    {n_jets, 1}           // input_5 - jet pt for reg 
  };
  
  // initalize inputs
  // CMMSW-DNN sets zeros by default
  for (std::size_t i=0; i < input_sizes.size(); i++) {
    auto & input_shape = input_sizes.at(i);
    dnn_inputs_.at(i)->setArray(input_shape.size(),
                                &input_shape[0]);
  }

  // fill values
  for (std::size_t jet_n=0; jet_n < tag_infos->size(); jet_n++) {
    // jet and oother global features
    const auto & features = tag_infos->at(jet_n).features();
    jet_tensor_filler(dnn_inputs_.at(0), jet_n, features);
    
  }
  

  // compute graph
  graph_.eval();

  // create output collection
  for (std::size_t i=0; i < outputs_.size(); i++) {
    if (tag_infos->size() > 0) {
      auto jet_ref = tag_infos->begin()->jet();
      output_tags.emplace_back(std::make_unique<JetTagCollection>(
            edm::makeRefToBaseProdFrom(jet_ref, iEvent)));
    } else {
      output_tags.emplace_back(std::make_unique<JetTagCollection>());
    }
  }

  

  for (std::size_t i=0; i < outputs_.size(); i++) {
    iEvent.put(std::move(output_tags.at(i)), outputs_.at(i));
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourJetTagProducer);
