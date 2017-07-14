
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Jet.h"

#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"

#include "DataFormats/DeepFormats/interface/DeepFlavourTagInfo.h"
#include "DataFormats/DeepFormats/interface/DeepFlavourFeatures.h"

#include "RecoBTag/DeepFlavour/interface/jet_features_converter.h"
#include "RecoBTag/DeepFlavour/interface/btag_features_converter.h"

class DeepFlavourTagInfoProducer : public edm::stream::EDProducer<> {

  public:
	  explicit DeepFlavourTagInfoProducer(const edm::ParameterSet&);
	  ~DeepFlavourTagInfoProducer();

	  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    typedef std::vector<reco::DeepFlavourTagInfo> DeepFlavourTagInfoCollection;
    typedef reco::VertexCompositePtrCandidateCollection SVCollection;
    typedef std::vector<reco::ShallowTagInfo> ShallowTagInfoCollection;

	  virtual void beginStream(edm::StreamID) override {}
	  virtual void produce(edm::Event&, const edm::EventSetup&) override;
	  virtual void endStream() override {}

    edm::EDGetTokenT<edm::View<pat::Jet>>  jet_token_;
    edm::EDGetTokenT<SVCollection> sv_token_;
    edm::EDGetTokenT<ShallowTagInfoCollection> shallow_tag_info_token_;

};

DeepFlavourTagInfoProducer::DeepFlavourTagInfoProducer(const edm::ParameterSet& iConfig) :
  jet_token_(consumes<edm::View<pat::Jet> >(iConfig.getParameter<edm::InputTag>("jets"))),
  sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
  shallow_tag_info_token_(consumes<ShallowTagInfoCollection>(iConfig.getParameter<edm::InputTag>("shallow_tag_infos")))
{
  produces<DeepFlavourTagInfoCollection>();
}


DeepFlavourTagInfoProducer::~DeepFlavourTagInfoProducer()
{
}

void DeepFlavourTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
}

void DeepFlavourTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  auto output_tag_infos = std::make_unique<DeepFlavourTagInfoCollection>();

  edm::Handle<edm::View<pat::Jet>> jets;
  iEvent.getByToken(jet_token_, jets);

  edm::Handle<SVCollection> svs;
  iEvent.getByToken(sv_token_, svs);

  edm::Handle<ShallowTagInfoCollection> shallow_tag_infos;
  iEvent.getByToken(shallow_tag_info_token_, shallow_tag_infos);


  for (const auto & tag_info : *shallow_tag_infos) {

    // create data containing structure
    deep::DeepFlavourFeatures features;

    auto jet_ref = tag_info.jet();
    // TODO: add an isAvailable check
    auto jet = dynamic_cast<const pat::Jet &>(jet_ref.operator*());
    // fill basic jet features
    deep::jet_features_converter(jet, features.jet_features);

    // fill features from ShallowTagInfo
    
    
    output_tag_infos->emplace_back(features, jet_ref);
  }

  


  iEvent.put(std::move(output_tag_infos));

}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourTagInfoProducer);
