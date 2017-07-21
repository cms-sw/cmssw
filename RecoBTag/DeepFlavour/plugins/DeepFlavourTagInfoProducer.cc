
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/DeepFormats/interface/DeepFlavourTagInfo.h"
#include "DataFormats/DeepFormats/interface/DeepFlavourFeatures.h"

#include "RecoBTag/DeepFlavour/interface/jet_features_converter.h"
#include "RecoBTag/DeepFlavour/interface/btag_features_converter.h"
#include "RecoBTag/DeepFlavour/interface/sv_features_converter.h"
#include "RecoBTag/DeepFlavour/interface/n_pf_features_converter.h"
#include "RecoBTag/DeepFlavour/interface/c_pf_features_converter.h"

#include "RecoBTag/DeepFlavour/interface/TrackInfoBuilder.h"
#include "RecoBTag/DeepFlavour/interface/sorting_modules.h"

class DeepFlavourTagInfoProducer : public edm::stream::EDProducer<> {

  public:
	  explicit DeepFlavourTagInfoProducer(const edm::ParameterSet&);
	  ~DeepFlavourTagInfoProducer();

	  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    typedef std::vector<reco::DeepFlavourTagInfo> DeepFlavourTagInfoCollection;
    typedef reco::VertexCompositePtrCandidateCollection SVCollection;
    typedef reco::VertexCollection VertexCollection;
    typedef std::vector<reco::ShallowTagInfo> ShallowTagInfoCollection;

	  virtual void beginStream(edm::StreamID) override {}
	  virtual void produce(edm::Event&, const edm::EventSetup&) override;
	  virtual void endStream() override {}

    
    const double jet_radius_;
    edm::EDGetTokenT<edm::View<pat::Jet>>  jet_token_;
    edm::EDGetTokenT<VertexCollection> vtx_token_;
    edm::EDGetTokenT<SVCollection> sv_token_;
    edm::EDGetTokenT<ShallowTagInfoCollection> shallow_tag_info_token_;
    


};

DeepFlavourTagInfoProducer::DeepFlavourTagInfoProducer(const edm::ParameterSet& iConfig) :
  jet_radius_(iConfig.getParameter<double>("jet_radius")),
  jet_token_(consumes<edm::View<pat::Jet> >(iConfig.getParameter<edm::InputTag>("jets"))),
  vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
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

  edm::Handle<VertexCollection> vtxs;
  iEvent.getByToken(vtx_token_, vtxs);
  // reference to primary vertex
  const auto & pv = vtxs->at(0);

  edm::Handle<SVCollection> svs;
  iEvent.getByToken(sv_token_, svs);

  edm::Handle<ShallowTagInfoCollection> shallow_tag_infos;
  iEvent.getByToken(shallow_tag_info_token_, shallow_tag_infos);

  edm::ESHandle<TransientTrackBuilder> track_builder; 
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", track_builder);

  for (const auto & tag_info : *shallow_tag_infos) {

    // create data containing structure
    deep::DeepFlavourFeatures features;

    auto jet_ref = tag_info.jet();
    // TODO: add an isAvailable check
    auto jet = dynamic_cast<const pat::Jet &>(jet_ref.operator*());
    // fill basic jet features
    deep::jet_features_converter(jet, features.jet_features);
    // fill number of pv
    features.npv = vtxs->size();

    // fill features from ShallowTagInfo
    const auto & tag_info_vars = tag_info.taggingVariables();
    btag_features_converter(tag_info_vars, features.tag_info_features);

 
    // copy which will be sorted
    auto svs_sorted = *svs;     
    // sort by dxy
    std::sort(svs_sorted.begin(), svs_sorted.end(),
              [&pv](const auto & sva, const auto &svb)
              { return deep::sv_vertex_comparator(sva, svb, pv); });
    // fill features from secondary vertices
    for (const auto & sv : svs_sorted) {
      if (reco::deltaR(sv, jet) > jet_radius_) continue;
      else {
        features.sv_features.emplace_back();
        // in C++17 could just get from emplace_back output
        auto & sv_features = features.sv_features.back();
        deep::sv_features_converter(sv, pv, jet, sv_features);
      }
    }

    // stuff required for dealing with pf candidates 
    math::XYZVector jet_dir = jet.momentum().Unit();
    GlobalVector jet_ref_track_dir(jet.px(),jet.py(),jet.pz());

    std::vector<sorting::sortingClass<pat::PackedCandidate>> sortedall;

    deep::TrackInfoBuilder trackinfo(track_builder);

    // unsorted reference to sv
    const auto & svs_unsorted = *svs;     
    // fill collection, from DeepTNtuples plus some styling
    for (unsigned int i = 0; i <  jet.numberOfDaughters(); i++){
        auto packed_cand = dynamic_cast<const pat::PackedCandidate*>(jet.daughter(i));
        if(packed_cand){

            trackinfo.buildTrackInfo(packed_cand,jet_dir,jet_ref_track_dir,pv);

            sortedall.emplace_back(packed_cand, trackinfo.getTrackSip2dSig(),
                    -deep::mindrsvpfcand(svs_unsorted,packed_cand), packed_cand->pt()/jet.pt());

        }
    }

    // sort collection (open the black-box if you please) 
    std::sort(sortedall.begin(),sortedall.end(),
              sorting::sortingClass<pat::PackedCandidate>::compareByABCInv);

  for (const auto& s : sortedall) {

    // get ref and check that is correct
    const auto& packed_cand =s.get();
    if(!packed_cand) continue;

    float drminpfcandsv = deep::mindrsvpfcand(svs_unsorted, packed_cand);
    
    if (packed_cand->charge() != 0) {
      // is charged candidate, add with default values
      features.c_pf_features.emplace_back();
      // build track info
      trackinfo.buildTrackInfo(packed_cand,jet_dir,jet_ref_track_dir,pv);
      // in C++17 could just get from emplace_back output
      auto & c_pf_features = features.c_pf_features.back();
      // fill feature structure 
      deep::c_pf_features_converter(packed_cand, jet, trackinfo, 
                                    drminpfcandsv, c_pf_features);
    } else {
      // is neutral candidate, add with default values
      features.n_pf_features.emplace_back();
      // in C++17 could just get from emplace_back output
      auto & n_pf_features = features.n_pf_features.back();
      // fill feature structure 
      deep::n_pf_features_converter(packed_cand, jet, drminpfcandsv, 
                                   n_pf_features);
    }

    
  }

    

    output_tag_infos->emplace_back(features, jet_ref);
  }

  iEvent.put(std::move(output_tag_infos));

}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourTagInfoProducer);
