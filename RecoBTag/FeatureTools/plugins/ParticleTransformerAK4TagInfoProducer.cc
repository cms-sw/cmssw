#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "DataFormats/BTauReco/interface/ParticleTransformerAK4TagInfo.h"
#include "DataFormats/BTauReco/interface/ParticleTransformerAK4Features.h"

#include "RecoBTag/FeatureTools/interface/JetConverter.h"
#include "RecoBTag/FeatureTools/interface/ShallowTagInfoConverter.h"
#include "RecoBTag/FeatureTools/interface/SecondaryVertexConverter.h"
#include "RecoBTag/FeatureTools/interface/NeutralCandidateConverter.h"
#include "RecoBTag/FeatureTools/interface/ChargedCandidateConverter.h"

#include "RecoBTag/FeatureTools/interface/TrackInfoBuilder.h"
#include "RecoBTag/FeatureTools/interface/sorting_modules.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

#include "RecoBTag/FeatureTools/interface/deep_helpers.h"

#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Common/interface/Provenance.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"

#include "DataFormats/BTauReco/interface/SeedingTrackFeatures.h"
#include "DataFormats/BTauReco/interface/TrackPairFeatures.h"
#include "RecoBTag/FeatureTools/interface/TrackPairInfoBuilder.h"
#include "RecoBTag/FeatureTools/interface/SeedingTrackInfoBuilder.h"
#include "RecoBTag/FeatureTools/interface/SeedingTracksConverter.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "RecoBTag/TrackProbability/interface/HistogramProbabilityEstimator.h"
class HistogramProbabilityEstimator;
#include <memory>

#include <typeinfo>
#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/EventSetupRecordKey.h"

class ParticleTransformerAK4TagInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit ParticleTransformerAK4TagInfoProducer(const edm::ParameterSet&);
  ~ParticleTransformerAK4TagInfoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef std::vector<reco::ParticleTransformerAK4TagInfo> ParticleTransformerAK4TagInfoCollection;
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;
  typedef reco::VertexCollection VertexCollection;

  void beginStream(edm::StreamID) override {}
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override {}

  const double jet_radius_;
  const double min_candidate_pt_;
  const bool flip_;

  edm::EDGetTokenT<edm::View<reco::Jet>> jet_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  edm::EDGetTokenT<SVCollection> sv_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> puppi_value_map_token_;
  edm::EDGetTokenT<edm::ValueMap<int>> pvasq_value_map_token_;
  edm::EDGetTokenT<edm::Association<VertexCollection>> pvas_token_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> candidateToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> track_builder_token_;
  bool use_puppi_value_map_;
  bool use_pvasq_value_map_;

  bool fallback_puppi_weight_;
  bool fallback_vertex_association_;

  bool is_weighted_jet_;

  const double min_jet_pt_;
  const double max_jet_eta_;
};

ParticleTransformerAK4TagInfoProducer::ParticleTransformerAK4TagInfoProducer(const edm::ParameterSet& iConfig)
    : jet_radius_(iConfig.getParameter<double>("jet_radius")),
      min_candidate_pt_(iConfig.getParameter<double>("min_candidate_pt")),
      flip_(iConfig.getParameter<bool>("flip")),
      jet_token_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
      candidateToken_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("candidates"))),
      track_builder_token_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))),
      use_puppi_value_map_(false),
      use_pvasq_value_map_(false),
      fallback_puppi_weight_(iConfig.getParameter<bool>("fallback_puppi_weight")),
      fallback_vertex_association_(iConfig.getParameter<bool>("fallback_vertex_association")),
      is_weighted_jet_(iConfig.getParameter<bool>("is_weighted_jet")),
      min_jet_pt_(iConfig.getParameter<double>("min_jet_pt")),
      max_jet_eta_(iConfig.getParameter<double>("max_jet_eta")) {
  produces<ParticleTransformerAK4TagInfoCollection>();

  const auto& puppi_value_map_tag = iConfig.getParameter<edm::InputTag>("puppi_value_map");
  if (!puppi_value_map_tag.label().empty()) {
    puppi_value_map_token_ = consumes<edm::ValueMap<float>>(puppi_value_map_tag);
    use_puppi_value_map_ = true;
  } else if (is_weighted_jet_) {
    throw edm::Exception(edm::errors::Configuration,
                         "puppi_value_map is not set but jet is weighted. Must set puppi_value_map.");
  }

  const auto& pvas_tag = iConfig.getParameter<edm::InputTag>("vertex_associator");
  if (!pvas_tag.label().empty()) {
    pvasq_value_map_token_ = consumes<edm::ValueMap<int>>(pvas_tag);
    pvas_token_ = consumes<edm::Association<VertexCollection>>(pvas_tag);
    use_pvasq_value_map_ = true;
  }
}

ParticleTransformerAK4TagInfoProducer::~ParticleTransformerAK4TagInfoProducer() {}

void ParticleTransformerAK4TagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfParticleTransformerAK4TagInfos
  edm::ParameterSetDescription desc;
  desc.add<double>("jet_radius", 0.4);
  desc.add<double>("min_candidate_pt", 0.95);
  desc.add<bool>("flip", false);
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("puppi_value_map", edm::InputTag("puppi"));
  desc.add<edm::InputTag>("secondary_vertices", edm::InputTag("inclusiveCandidateSecondaryVertices"));
  desc.add<edm::InputTag>("jets", edm::InputTag("ak4PFJetsCHS"));
  desc.add<edm::InputTag>("candidates", edm::InputTag("packedPFCandidates"));
  desc.add<edm::InputTag>("vertex_associator", edm::InputTag("primaryVertexAssociation", "original"));
  desc.add<bool>("fallback_puppi_weight", false);
  desc.add<bool>("fallback_vertex_association", false);
  desc.add<bool>("is_weighted_jet", false);
  desc.add<double>("min_jet_pt", 15.0);
  desc.add<double>("max_jet_eta", 2.5);
  descriptions.add("pfParticleTransformerAK4TagInfos", desc);
}

void ParticleTransformerAK4TagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto output_tag_infos = std::make_unique<ParticleTransformerAK4TagInfoCollection>();
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

  edm::Handle<edm::View<reco::Candidate>> tracks;
  iEvent.getByToken(candidateToken_, tracks);

  edm::Handle<SVCollection> svs;
  iEvent.getByToken(sv_token_, svs);

  edm::Handle<edm::ValueMap<float>> puppi_value_map;
  if (use_puppi_value_map_) {
    iEvent.getByToken(puppi_value_map_token_, puppi_value_map);
  }

  edm::Handle<edm::ValueMap<int>> pvasq_value_map;
  edm::Handle<edm::Association<VertexCollection>> pvas;
  if (use_pvasq_value_map_) {
    iEvent.getByToken(pvasq_value_map_token_, pvasq_value_map);
    iEvent.getByToken(pvas_token_, pvas);
  }

  edm::ESHandle<TransientTrackBuilder> track_builder = iSetup.getHandle(track_builder_token_);

  for (std::size_t jet_n = 0; jet_n < jets->size(); jet_n++) {
    // create data containing structure
    btagbtvdeep::ParticleTransformerAK4Features features;

    // reco jet reference (use as much as possible)
    const auto& jet = jets->at(jet_n);
    if (jet.pt() < 15.0) {
      features.is_filled = false;
    }
    if (std::abs(jet.eta()) > 2.5) {
      features.is_filled = false;
    }
    // dynamical casting to pointers, null if not possible
    const auto* pf_jet = dynamic_cast<const reco::PFJet*>(&jet);
    const auto* pat_jet = dynamic_cast<const pat::Jet*>(&jet);
    edm::RefToBase<reco::Jet> jet_ref(jets, jet_n);

    if (features.is_filled) {
      math::XYZVector jet_dir = jet.momentum().Unit();
      GlobalVector jet_ref_track_dir(jet.px(), jet.py(), jet.pz());

      // copy which will be sorted
      auto svs_sorted = *svs;
      // sort by dxy
      std::sort(svs_sorted.begin(), svs_sorted.end(), [&pv](const auto& sva, const auto& svb) {
        return btagbtvdeep::sv_vertex_comparator(sva, svb, pv);
      });
      // fill features from secondary vertices
      for (const auto& sv : svs_sorted) {
        if (reco::deltaR2(sv, jet_dir) > (jet_radius_ * jet_radius_))
          continue;
        else {
          features.sv_features.emplace_back();
          // in C++17 could just get from emplace_back output
          auto& sv_features = features.sv_features.back();
          btagbtvdeep::svToFeatures(sv, pv, jet, sv_features, flip_);
        }
      }

      // stuff required for dealing with pf candidates

      std::vector<btagbtvdeep::SortingClass<size_t>> c_sorted, n_sorted;

      // to cache the TrackInfo
      std::map<unsigned int, btagbtvdeep::TrackInfoBuilder> trackinfos;

      // unsorted reference to sv
      const auto& svs_unsorted = *svs;
      // fill collection, from DeepTNtuples plus some styling
      for (unsigned int i = 0; i < jet.numberOfDaughters(); i++) {
        auto cand = jet.daughter(i);
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
                                  -btagbtvdeep::mindrsvpfcand(svs_unsorted, cand),
                                  cand->pt() / jet.pt());
          } else {
            n_sorted.emplace_back(i, -1, -btagbtvdeep::mindrsvpfcand(svs_unsorted, cand), cand->pt() / jet.pt());
          }
        }
      }

      // sort collections (open the black-box if you please)
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

      for (unsigned int i = 0; i < jet.numberOfDaughters(); i++) {
        // get pointer and check that is correct
        auto cand = dynamic_cast<const reco::Candidate*>(jet.daughter(i));
        if (!cand)
          continue;
        // candidates under 950MeV are not considered
        // might change if we use also white-listing
        if (cand->pt() < 0.95)
          continue;

        auto packed_cand = dynamic_cast<const pat::PackedCandidate*>(cand);
        auto reco_cand = dynamic_cast<const reco::PFCandidate*>(cand);

        // need some edm::Ptr or edm::Ref if reco candidates
        reco::PFCandidatePtr reco_ptr;
        if (pf_jet) {
          reco_ptr = pf_jet->getPFConstituent(i);
        } else if (pat_jet && reco_cand) {
          reco_ptr = pat_jet->getPFConstituent(i);
        }

        reco::CandidatePtr cand_ptr;
        if (pat_jet) {
          cand_ptr = pat_jet->sourceCandidatePtr(i);
        }

        //
        // Access puppi weight from ValueMap.
        //
        float puppiw = 1.0;  // Set to fallback value

        if (reco_cand) {
          if (use_puppi_value_map_)
            puppiw = (*puppi_value_map)[reco_ptr];
          else if (!fallback_puppi_weight_) {
            throw edm::Exception(edm::errors::InvalidReference, "PUPPI value map missing")
                << "use fallback_puppi_weight option to use " << puppiw << " for reco_cand as default";
          }
        } else if (packed_cand) {
          if (use_puppi_value_map_)
            puppiw = (*puppi_value_map)[cand_ptr];
          else if (!fallback_puppi_weight_) {
            throw edm::Exception(edm::errors::InvalidReference, "PUPPI value map missing")
                << "use fallback_puppi_weight option to use " << puppiw << " for packed_cand as default";
          }
        } else {
          throw edm::Exception(edm::errors::InvalidReference)
              << "Cannot convert to either reco::PFCandidate or pat::PackedCandidate";
        }

        float drminpfcandsv = btagbtvdeep::mindrsvpfcand(svs_unsorted, cand);
        float distminpfcandsv = 0;
        if (cand->charge() != 0) {
          // is charged candidate
          auto entry = c_sortedindices.at(i);

          // get cached track info
          auto& trackinfo = trackinfos.at(i);

          // get_ref to vector element
          auto& c_pf_features = features.c_pf_features.at(entry);
          // fill feature structure
          if (packed_cand) {
            if (packed_cand->hasTrackDetails()) {
              const reco::Track& PseudoTrack = packed_cand->pseudoTrack();
              reco::TransientTrack transientTrack;
              transientTrack = track_builder->build(PseudoTrack);
              distminpfcandsv = btagbtvdeep::mindistsvpfcand(svs_unsorted, transientTrack);
            }

            btagbtvdeep::packedCandidateToFeatures(packed_cand,
                                                   jet,
                                                   trackinfo,
                                                   is_weighted_jet_,
                                                   drminpfcandsv,
                                                   static_cast<float>(jet_radius_),
                                                   puppiw,
                                                   c_pf_features,
                                                   flip_,
                                                   distminpfcandsv);
          } else if (reco_cand) {
            // get vertex association quality
            int pv_ass_quality = 0;  // fallback value
            if (use_pvasq_value_map_) {
              pv_ass_quality = (*pvasq_value_map)[reco_ptr];
            } else if (!fallback_vertex_association_) {
              throw edm::Exception(edm::errors::InvalidReference, "vertex association missing")
                  << "use fallback_vertex_association option to use" << pv_ass_quality
                  << "as default quality and closest dz PV as criteria";
            }
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
            auto PV = reco::VertexRef(vtxs, pvi);
            if (use_pvasq_value_map_) {
              const reco::VertexRef& PV_orig = (*pvas)[reco_ptr];
              if (PV_orig.isNonnull())
                PV = reco::VertexRef(vtxs, PV_orig.key());
            }
            btagbtvdeep::recoCandidateToFeatures(reco_cand,
                                                 jet,
                                                 trackinfo,
                                                 is_weighted_jet_,
                                                 drminpfcandsv,
                                                 static_cast<float>(jet_radius_),
                                                 puppiw,
                                                 pv_ass_quality,
                                                 PV,
                                                 c_pf_features,
                                                 flip_,
                                                 distminpfcandsv);
          }
        } else {
          // is neutral candidate
          auto entry = n_sortedindices.at(i);
          // get_ref to vector element
          auto& n_pf_features = features.n_pf_features.at(entry);
          // fill feature structure
          if (packed_cand) {
            btagbtvdeep::packedCandidateToFeatures(packed_cand,
                                                   jet,
                                                   is_weighted_jet_,
                                                   drminpfcandsv,
                                                   static_cast<float>(jet_radius_),
                                                   puppiw,
                                                   n_pf_features);
          } else if (reco_cand) {
            btagbtvdeep::recoCandidateToFeatures(
                reco_cand, jet, is_weighted_jet_, drminpfcandsv, static_cast<float>(jet_radius_), puppiw, n_pf_features);
          }
        }
      }
    }
    output_tag_infos->emplace_back(features, jet_ref);
  }
  iEvent.put(std::move(output_tag_infos));
}

//define this as a plug-in
DEFINE_FWK_MODULE(ParticleTransformerAK4TagInfoProducer);
