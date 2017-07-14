
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/DeepFormats/interface/DeepFlavourTagInfo.h"


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
    std::vector<std::string> outputs_;

};

DeepFlavourJetTagProducer::DeepFlavourJetTagProducer(const edm::ParameterSet& iConfig) :
  src_(consumes<TagInfoCollection>(iConfig.getParameter<edm::InputTag>("src"))),
  outputs_(iConfig.getParameter<std::vector<std::string>>("outputs"))
{

  edm::LogInfo("DeepFlavourJetTags") << "outputs size: " << outputs_.size() << " " <<  outputs_.at(0);

  for (const auto & output : outputs_) {
    produces<JetTagCollection>(output);
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
