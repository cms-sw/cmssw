
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/BTauReco/interface/BoostedDoubleSVTagInfo.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "DataFormats/BTauReco/interface/DeepDoubleXFeatures.h"
#include "DataFormats/BTauReco/interface/DeepDoubleXTagInfo.h"

#include "RecoBTag/FeatureTools/interface/BoostedDoubleSVTagInfoConverter.h"
#include "RecoBTag/FeatureTools/interface/NeutralCandidateConverter.h"
#include "RecoBTag/FeatureTools/interface/ChargedCandidateConverter.h"
#include "RecoBTag/FeatureTools/interface/JetConverter.h"
#include "RecoBTag/FeatureTools/interface/SecondaryVertexConverter.h"

#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"
#include "RecoBTag/FeatureTools/interface/sorting_modules.h"

#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"

class DeepDoubleXTagInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit DeepDoubleXTagInfoProducer(const edm::ParameterSet&);
  ~DeepDoubleXTagInfoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef std::vector<reco::DeepDoubleXTagInfo> DeepDoubleXTagInfoCollection;
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;
  typedef reco::VertexCollection VertexCollection;
  typedef edm::View<reco::BoostedDoubleSVTagInfo> BoostedDoubleSVTagInfoCollection;

  void beginStream(edm::StreamID) override {}
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override {}

  const double jet_radius_;
  const double min_jet_pt_;
  const double min_candidate_pt_;

  edm::EDGetTokenT<edm::View<reco::Jet>> jet_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  edm::EDGetTokenT<SVCollection> sv_token_;
  edm::EDGetTokenT<BoostedDoubleSVTagInfoCollection> shallow_tag_info_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> puppi_value_map_token_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> track_builder_token_;

  bool use_puppi_value_map_;
  bool fallback_puppi_weight_;
};

DeepDoubleXTagInfoProducer::DeepDoubleXTagInfoProducer(const edm::ParameterSet& iConfig)
    : jet_radius_(iConfig.getParameter<double>("jet_radius")),
      min_jet_pt_(iConfig.getParameter<double>("min_jet_pt")),
      min_candidate_pt_(iConfig.getParameter<double>("min_candidate_pt")),
      jet_token_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
      shallow_tag_info_token_(
          consumes<BoostedDoubleSVTagInfoCollection>(iConfig.getParameter<edm::InputTag>("shallow_tag_infos"))),
      track_builder_token_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))), 
      use_puppi_value_map_(false),
      fallback_puppi_weight_(iConfig.getParameter<bool>("fallback_puppi_weight"))
      {
  produces<DeepDoubleXTagInfoCollection>();

  const auto& puppi_value_map_tag = iConfig.getParameter<edm::InputTag>("puppi_value_map");
  if (!puppi_value_map_tag.label().empty()) {
    puppi_value_map_token_ = consumes<edm::ValueMap<float>>(puppi_value_map_tag);
    use_puppi_value_map_ = true;
  }
}

DeepDoubleXTagInfoProducer::~DeepDoubleXTagInfoProducer() {}

void DeepDoubleXTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfDeepDoubleXTagInfos
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("shallow_tag_infos", edm::InputTag("pfBoostedDoubleSVAK8TagInfos"));
  desc.add<double>("jet_radius", 0.8);
  desc.add<double>("min_jet_pt", 150);
  desc.add<double>("min_candidate_pt", 0.95);
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("puppi_value_map", edm::InputTag("puppi"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("inclusiveCandidateSecondaryVertices"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak8PFJetsPuppi"));
  desc.add<bool>("fallback_puppi_weight", false);
  descriptions.add("pfDeepDoubleXTagInfos", desc);
}

void DeepDoubleXTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto output_tag_infos = std::make_unique<DeepDoubleXTagInfoCollection>();

  edm::Handle<edm::View<reco::Jet>> jets;
  iEvent.getByToken(jet_token_, jets);

  edm::Handle<VertexCollection> vtxs;
  iEvent.getByToken(vtx_token_, vtxs);
  if (vtxs->empty()) {
    // produce empty TagInfos in case no primary vertex
    iEvent.put(std::move(output_tag_infos));
    return;  // exit event
  }
  // reference to primary vertex
  const auto& pv = vtxs->at(0);

  edm::Handle<SVCollection> svs;
  iEvent.getByToken(sv_token_, svs);

  edm::Handle<BoostedDoubleSVTagInfoCollection> shallow_tag_infos;
  iEvent.getByToken(shallow_tag_info_token_, shallow_tag_infos);

  edm::Handle<edm::ValueMap<float>> puppi_value_map;
  if (use_puppi_value_map_) {
    iEvent.getByToken(puppi_value_map_token_, puppi_value_map);
  }

  edm::ESHandle<TransientTrackBuilder> track_builder = iSetup.getHandle(track_builder_token_);

  for (std::size_t jet_n = 0; jet_n < jets->size(); jet_n++) {
    // create data containing structure
    btagbtvdeep::DeepDoubleXFeatures features;

    // reco jet reference (use as much as possible)
    const auto& jet = jets->at(jet_n);
    const auto* pf_jet = dynamic_cast<const reco::PFJet*>(&jet);
    const auto* pat_jet = dynamic_cast<const pat::Jet*>(&jet);
    if (!pat_jet)
      throw edm::Exception(edm::errors::InvalidReference) << "Input is not a pat::Jet.";

    edm::RefToBase<reco::Jet> jet_ref(jets, jet_n);
    if (jet.pt() > min_jet_pt_) {
      features.filled();
      // TagInfoCollection not in an associative container so search for matchs
      const edm::View<reco::BoostedDoubleSVTagInfo>& taginfos = *shallow_tag_infos;
      edm::Ptr<reco::BoostedDoubleSVTagInfo> match;
      // Try first by 'same index'
      if ((jet_n < taginfos.size()) && (taginfos[jet_n].jet() == jet_ref)) {
        match = taginfos.ptrAt(jet_n);
      } else {
        // otherwise fall back to a simple search
        for (auto itTI = taginfos.begin(), edTI = taginfos.end(); itTI != edTI; ++itTI) {
          if (itTI->jet() == jet_ref) {
            match = taginfos.ptrAt(itTI - taginfos.begin());
            break;
          }
        }
      }
      reco::BoostedDoubleSVTagInfo tag_info;
      if (match.isNonnull()) {
        tag_info = *match;
      }  // will be default values otherwise

      // fill basic jet features
      btagbtvdeep::JetConverter::jetToFeatures(jet, features.jet_features);

      // fill number of pv
      features.npv = vtxs->size();

      // fill features from BoostedDoubleSVTagInfo
      const auto& tag_info_vars = tag_info.taggingVariables();
      btagbtvdeep::doubleBTagToFeatures(tag_info_vars, features.tag_info_features);

      // copy which will be sorted
      auto svs_sorted = *svs;
      // sort by dxy
      std::sort(svs_sorted.begin(), svs_sorted.end(), [&pv](const auto& sva, const auto& svb) {
        return btagbtvdeep::sv_vertex_comparator(sva, svb, pv);
      });
      // fill features from secondary vertices
      for (const auto& sv : svs_sorted) {
        if (reco::deltaR(sv, jet) > jet_radius_)
          continue;
        else {
          features.sv_features.emplace_back();
          // in C++17 could just get from emplace_back output
          auto& sv_features = features.sv_features.back();
          btagbtvdeep::svToFeatures(sv, pv, jet, sv_features);
        }
      }

      // stuff required for dealing with pf candidates
      math::XYZVector jet_dir = jet.momentum().Unit();
      GlobalVector jet_ref_track_dir(jet.px(), jet.py(), jet.pz());

      std::vector<btagbtvdeep::SortingClass<size_t>> c_sorted, n_sorted;
      std::vector<int> n_indexes;

      // to cache the TrackInfo
      std::map<unsigned int, btagbtvdeep::TrackInfoBuilder> trackinfos;

      // unsorted reference to sv
      const auto& svs_unsorted = *svs;
      // fill collection, from DeepTNtuples plus some styling
      // std::vector<const pat::PackedCandidate*> daughters;
      std::vector<const reco::Candidate*> daughters;
      for (unsigned int i = 0; i < jet.numberOfDaughters(); i++) {
        auto const* cand = jet.daughter(i);
        auto packed_cand = dynamic_cast<const pat::PackedCandidate*>(cand);
        auto reco_cand = dynamic_cast<const reco::PFCandidate*>(cand);
        if (packed_cand) {
          if (cand->numberOfDaughters() > 0) {
            for (unsigned int k = 0; k < cand->numberOfDaughters(); k++) {
              daughters.push_back(dynamic_cast<const pat::PackedCandidate*>(cand->daughter(k)));
            }
          } else {
            auto packed_cand = dynamic_cast<const pat::PackedCandidate*>(cand);
            daughters.push_back(packed_cand);
          }
        } else if (reco_cand) {
          // need some edm::Ptr or edm::Ref if reco candidates
          // dynamical casting to pointers, null if not possible
          daughters.push_back(reco_cand);
        }
      }

      std::sort(daughters.begin(), daughters.end(), [](const auto& a, const auto& b) { return a->pt() > b->pt(); });
      for (unsigned int i = 0; i < daughters.size(); i++) {
        auto const* cand = daughters.at(i);
        if (cand) {
          // candidates under 950MeV (configurable) are not considered
          // might change if we use also white-listing
          if (cand->pt() < min_candidate_pt_)
            continue;
          if (cand->charge() != 0) {
            auto& trackinfo = trackinfos.emplace(i, track_builder).first->second;
            trackinfo.buildTrackInfo(cand, jet_dir, jet_ref_track_dir, pv);
            c_sorted.emplace_back(i,
                                  trackinfo.getTrackSip2dSig(),
                                  -btagbtvdeep::mindrsvpfcand(svs_unsorted, cand, jet_radius_),
                                  cand->pt() / jet.pt());
          } else {
            n_sorted.emplace_back(
                i, -1, -btagbtvdeep::mindrsvpfcand(svs_unsorted, cand, jet_radius_), cand->pt() / jet.pt());
            n_indexes.push_back(i);
          }
        }
      }

      // sort collections in added order of priority
      std::sort(c_sorted.begin(), c_sorted.end(), btagbtvdeep::SortingClass<std::size_t>::compareByABCInv);
      std::sort(n_sorted.begin(), n_sorted.end(), btagbtvdeep::SortingClass<std::size_t>::compareByABCInv);

      std::vector<size_t> c_sortedindices, n_sortedindices;

      // this puts 0 everywhere and the right position in ind
      c_sortedindices = btagbtvdeep::invertSortingVector(c_sorted);
      n_sortedindices = btagbtvdeep::invertSortingVector(n_sorted);

      // set right size to vectors
      features.c_pf_features.clear();
      features.c_pf_features.resize(c_sorted.size());
      features.n_pf_features.clear();
      features.n_pf_features.resize(n_sorted.size());

      for (unsigned int i = 0; i < daughters.size(); i++) {
        auto const* cand = daughters.at(i);
        if (cand) {
          // candidates under 950MeV are not considered
          // might change if we use also white-listing
          if (cand->pt() < min_candidate_pt_)
            continue;
          auto packed_cand = dynamic_cast<const pat::PackedCandidate*>(cand);
          auto reco_cand = dynamic_cast<const reco::PFCandidate*>(cand);

          reco::PFCandidatePtr reco_ptr;
          if (pf_jet) {
            reco_ptr = pf_jet->getPFConstituent(i);
          } 

          reco::CandidatePtr cand_ptr;
          if (pat_jet){
            cand_ptr = pat_jet->sourceCandidatePtr(i);
          }

          float puppiw = 1.0;  // fallback value
          if (reco_cand) {
            puppiw = 1.0;  // fallback value for reco_cand
            if (use_puppi_value_map_)
              puppiw = (*puppi_value_map)[reco_ptr];
            else if (!fallback_puppi_weight_) {
              throw edm::Exception(edm::errors::InvalidReference, "PUPPI value map missing")
                << "use fallback_puppi_weight option to use " << puppiw << " for reco_cand as default";
            }
          } 
          else if(packed_cand){
            puppiw = packed_cand->puppiWeight();  // fallback value for packed_cand
            if (use_puppi_value_map_)
              puppiw = (*puppi_value_map)[cand_ptr];
            else if (!fallback_puppi_weight_) {
              throw edm::Exception(edm::errors::InvalidReference, "PUPPI value map missing")
                << "use fallback_puppi_weight option to use puppiWeight() for packed_cand as default";
            }
          }


          float drminpfcandsv = btagbtvdeep::mindrsvpfcand(svs_unsorted, cand, jet_radius_);
          if (cand->charge() != 0) {
            // is charged candidate
            auto entry = c_sortedindices.at(i);
            // get cached track info
            auto& trackinfo = trackinfos.at(i);
            // get_ref to vector element
            auto& c_pf_features = features.c_pf_features.at(entry);
            if (packed_cand) {
              btagbtvdeep::packedCandidateToFeatures(
                  packed_cand, *pat_jet, trackinfo, drminpfcandsv, static_cast<float>(jet_radius_), puppiw, c_pf_features);
            } else if (reco_cand) {
              // get vertex association quality
              int pv_ass_quality = 0;  // fallback value
              // getting the PV as PackedCandidatesProducer
              // but using not the slimmed but original vertices
              auto ctrack = reco_cand->bestTrack();
              int pvi = -1;
              float dist = 1e99;
              for (size_t ii = 0; ii < vtxs->size(); ii++) {
                float dz = (ctrack) ? std::abs(ctrack->dz(((*vtxs)[ii]).position())) : 0;
                if (dz < dist) {
                  pvi = ii;
                  dist = dz;
                }
              }
              auto pv = reco::VertexRef(vtxs, pvi);
              btagbtvdeep::recoCandidateToFeatures(reco_cand,
                                                   jet,
                                                   trackinfo,
                                                   drminpfcandsv,
                                                   static_cast<float>(jet_radius_),
                                                   puppiw,
                                                   pv_ass_quality,
                                                   pv,
                                                   c_pf_features);
            }
          } else {
            // is neutral candidate
            auto entry = n_sortedindices.at(i);
            // get_ref to vector element
            auto& n_pf_features = features.n_pf_features.at(entry);
            // // fill feature structure
            if (packed_cand) {
              btagbtvdeep::packedCandidateToFeatures(
                  packed_cand, *pat_jet, drminpfcandsv, static_cast<float>(jet_radius_), puppiw, n_pf_features);
            } else if (reco_cand) {
              btagbtvdeep::recoCandidateToFeatures(
                  reco_cand, jet, drminpfcandsv, static_cast<float>(jet_radius_), puppiw, n_pf_features);
            }
          }
        }
      }
    }
    output_tag_infos->emplace_back(features, jet_ref);
  }

  iEvent.put(std::move(output_tag_infos));
}

// define this as a plug-in
DEFINE_FWK_MODULE(DeepDoubleXTagInfoProducer);
