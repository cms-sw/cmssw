
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

#include "DataFormats/BTauReco/interface/DeepFlavourTagInfo.h"
#include "DataFormats/BTauReco/interface/DeepFlavourFeatures.h"

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

class DeepFlavourTagInfoProducer : public edm::stream::EDProducer<> {
public:
  explicit DeepFlavourTagInfoProducer(const edm::ParameterSet&);
  ~DeepFlavourTagInfoProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  typedef std::vector<reco::DeepFlavourTagInfo> DeepFlavourTagInfoCollection;
  typedef reco::VertexCompositePtrCandidateCollection SVCollection;
  typedef reco::VertexCollection VertexCollection;
  typedef edm::View<reco::ShallowTagInfo> ShallowTagInfoCollection;

  void beginStream(edm::StreamID) override {}
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endStream() override {}

  const double jet_radius_;
  const double min_candidate_pt_;
  const bool flip_;

  edm::EDGetTokenT<edm::View<reco::Jet>> jet_token_;
  edm::EDGetTokenT<VertexCollection> vtx_token_;
  edm::EDGetTokenT<SVCollection> sv_token_;
  edm::EDGetTokenT<ShallowTagInfoCollection> shallow_tag_info_token_;
  edm::EDGetTokenT<edm::ValueMap<float>> puppi_value_map_token_;
  edm::EDGetTokenT<edm::ValueMap<int>> pvasq_value_map_token_;
  edm::EDGetTokenT<edm::Association<VertexCollection>> pvas_token_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> candidateToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> track_builder_token_;
  edm::ESGetToken<TrackProbabilityCalibration, BTagTrackProbability2DRcd> calib2d_token_;
  edm::ESGetToken<TrackProbabilityCalibration, BTagTrackProbability3DRcd> calib3d_token_;

  bool use_puppi_value_map_;
  bool use_pvasq_value_map_;

  bool fallback_puppi_weight_;
  bool fallback_vertex_association_;

  bool run_deepVertex_;

  //TrackProbability
  void checkEventSetup(const edm::EventSetup& iSetup);
  std::unique_ptr<HistogramProbabilityEstimator> probabilityEstimator_;
  bool compute_probabilities_;
  unsigned long long calibrationCacheId2D_;
  unsigned long long calibrationCacheId3D_;

  const double min_jet_pt_;
  const double max_jet_eta_;
};

DeepFlavourTagInfoProducer::DeepFlavourTagInfoProducer(const edm::ParameterSet& iConfig)
    : jet_radius_(iConfig.getParameter<double>("jet_radius")),
      min_candidate_pt_(iConfig.getParameter<double>("min_candidate_pt")),
      flip_(iConfig.getParameter<bool>("flip")),
      jet_token_(consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jets"))),
      vtx_token_(consumes<VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      sv_token_(consumes<SVCollection>(iConfig.getParameter<edm::InputTag>("secondary_vertices"))),
      shallow_tag_info_token_(
          consumes<ShallowTagInfoCollection>(iConfig.getParameter<edm::InputTag>("shallow_tag_infos"))),
      candidateToken_(consumes<edm::View<reco::Candidate>>(iConfig.getParameter<edm::InputTag>("candidates"))),
      track_builder_token_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))),
      use_puppi_value_map_(false),
      use_pvasq_value_map_(false),
      fallback_puppi_weight_(iConfig.getParameter<bool>("fallback_puppi_weight")),
      fallback_vertex_association_(iConfig.getParameter<bool>("fallback_vertex_association")),
      run_deepVertex_(iConfig.getParameter<bool>("run_deepVertex")),
      compute_probabilities_(iConfig.getParameter<bool>("compute_probabilities")),
      min_jet_pt_(iConfig.getParameter<double>("min_jet_pt")),
      max_jet_eta_(iConfig.getParameter<double>("max_jet_eta")) {
  produces<DeepFlavourTagInfoCollection>();

  const auto& puppi_value_map_tag = iConfig.getParameter<edm::InputTag>("puppi_value_map");
  if (!puppi_value_map_tag.label().empty()) {
    puppi_value_map_token_ = consumes<edm::ValueMap<float>>(puppi_value_map_tag);
    use_puppi_value_map_ = true;
  }

  const auto& pvas_tag = iConfig.getParameter<edm::InputTag>("vertex_associator");
  if (!pvas_tag.label().empty()) {
    pvasq_value_map_token_ = consumes<edm::ValueMap<int>>(pvas_tag);
    pvas_token_ = consumes<edm::Association<VertexCollection>>(pvas_tag);
    use_pvasq_value_map_ = true;
  }
  if (compute_probabilities_) {
    calib2d_token_ = esConsumes<TrackProbabilityCalibration, BTagTrackProbability2DRcd>();
    calib3d_token_ = esConsumes<TrackProbabilityCalibration, BTagTrackProbability3DRcd>();
  }
}

DeepFlavourTagInfoProducer::~DeepFlavourTagInfoProducer() {}

void DeepFlavourTagInfoProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // pfDeepFlavourTagInfos
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("shallow_tag_infos", edm::InputTag("pfDeepCSVTagInfos"));
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
  desc.add<bool>("run_deepVertex", false);
  desc.add<bool>("compute_probabilities", false);
  desc.add<double>("min_jet_pt", 15.0);
  desc.add<double>("max_jet_eta", 2.5);
  descriptions.add("pfDeepFlavourTagInfos", desc);
}

void DeepFlavourTagInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto output_tag_infos = std::make_unique<DeepFlavourTagInfoCollection>();
  if (compute_probabilities_)
    checkEventSetup(iSetup);

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

  edm::Handle<ShallowTagInfoCollection> shallow_tag_infos;
  iEvent.getByToken(shallow_tag_info_token_, shallow_tag_infos);
  double negative_cut = 0;  //used only with flip_
  if (flip_) {              //FIXME: Check if can do even less often than once per event
    const edm::Provenance* prov = shallow_tag_infos.provenance();
    const edm::ParameterSet& psetFromProvenance = edm::parameterSet(prov->stable(), iEvent.processHistory());
    negative_cut = ((psetFromProvenance.getParameter<edm::ParameterSet>("computer"))
                        .getParameter<edm::ParameterSet>("trackSelection"))
                       .getParameter<double>("sip3dSigMax");
  }

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

  std::vector<reco::TransientTrack> selectedTracks;
  std::vector<float> masses;

  if (run_deepVertex_)  //make a collection of selected transient tracks for deepvertex outside of the jet loop
  {
    for (typename edm::View<reco::Candidate>::const_iterator track = tracks->begin(); track != tracks->end(); ++track) {
      unsigned int k = track - tracks->begin();
      if (track->bestTrack() != nullptr && track->pt() > 0.5) {
        if (std::fabs(track->vz() - pv.z()) < 0.5) {
          selectedTracks.push_back(track_builder->build(tracks->ptrAt(k)));
          masses.push_back(track->mass());
        }
      }
    }
  }

  for (std::size_t jet_n = 0; jet_n < jets->size(); jet_n++) {
    // create data containing structure
    btagbtvdeep::DeepFlavourFeatures features;

    // reco jet reference (use as much as possible)
    const auto& jet = jets->at(jet_n);
    // dynamical casting to pointers, null if not possible
    const auto* pf_jet = dynamic_cast<const reco::PFJet*>(&jet);
    const auto* pat_jet = dynamic_cast<const pat::Jet*>(&jet);
    edm::RefToBase<reco::Jet> jet_ref(jets, jet_n);
    // TagInfoCollection not in an associative container so search for matchs
    const edm::View<reco::ShallowTagInfo>& taginfos = *shallow_tag_infos;
    edm::Ptr<reco::ShallowTagInfo> match;
    // Try first by 'same index'
    if ((jet_n < taginfos.size()) && (taginfos[jet_n].jet() == jet_ref)) {
      match = taginfos.ptrAt(jet_n);
    } else {
      // otherwise fail back to a simple search
      for (auto itTI = taginfos.begin(), edTI = taginfos.end(); itTI != edTI; ++itTI) {
        if (itTI->jet() == jet_ref) {
          match = taginfos.ptrAt(itTI - taginfos.begin());
          break;
        }
      }
    }
    reco::ShallowTagInfo tag_info;
    if (match.isNonnull()) {
      tag_info = *match;
    }  // will be default values otherwise

    // fill basic jet features
    btagbtvdeep::JetConverter::jetToFeatures(jet, features.jet_features);

    // fill number of pv
    features.npv = vtxs->size();
    math::XYZVector jet_dir = jet.momentum().Unit();
    GlobalVector jet_ref_track_dir(jet.px(), jet.py(), jet.pz());

    // fill features from ShallowTagInfo
    const auto& tag_info_vars = tag_info.taggingVariables();
    btagbtvdeep::bTagToFeatures(tag_info_vars, features.tag_info_features);

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
          c_sorted.emplace_back(
              i, trackinfo.getTrackSip2dSig(), -btagbtvdeep::mindrsvpfcand(svs_unsorted, cand), cand->pt() / jet.pt());
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
      // get PUPPI weight from value map
      float puppiw = 1.0;  // fallback value
      if (reco_cand && use_puppi_value_map_) {
        puppiw = (*puppi_value_map)[reco_ptr];
      } else if (reco_cand && !fallback_puppi_weight_) {
        throw edm::Exception(edm::errors::InvalidReference, "PUPPI value map missing")
            << "use fallback_puppi_weight option to use " << puppiw << "as default";
      }

      float drminpfcandsv = btagbtvdeep::mindrsvpfcand(svs_unsorted, cand);

      if (cand->charge() != 0) {
        // is charged candidate
        auto entry = c_sortedindices.at(i);
        // get cached track info
        auto& trackinfo = trackinfos.at(i);
        if (flip_ && (trackinfo.getTrackSip3dSig() > negative_cut)) {
          continue;
        }
        // get_ref to vector element
        auto& c_pf_features = features.c_pf_features.at(entry);
        // fill feature structure
        if (packed_cand) {
          btagbtvdeep::packedCandidateToFeatures(
              packed_cand, jet, trackinfo, drminpfcandsv, static_cast<float>(jet_radius_), c_pf_features, flip_);
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
                                               drminpfcandsv,
                                               static_cast<float>(jet_radius_),
                                               puppiw,
                                               pv_ass_quality,
                                               PV,
                                               c_pf_features,
                                               flip_);
        }
      } else {
        // is neutral candidate
        auto entry = n_sortedindices.at(i);
        // get_ref to vector element
        auto& n_pf_features = features.n_pf_features.at(entry);
        // fill feature structure
        if (packed_cand) {
          btagbtvdeep::packedCandidateToFeatures(
              packed_cand, jet, drminpfcandsv, static_cast<float>(jet_radius_), n_pf_features);
        } else if (reco_cand) {
          btagbtvdeep::recoCandidateToFeatures(
              reco_cand, jet, drminpfcandsv, static_cast<float>(jet_radius_), puppiw, n_pf_features);
        }
      }
    }

    if (run_deepVertex_) {
      if (jet.pt() > min_jet_pt_ && std::fabs(jet.eta()) < max_jet_eta_)  //jet thresholds
        btagbtvdeep::seedingTracksToFeatures(selectedTracks,
                                             masses,
                                             jet,
                                             pv,
                                             probabilityEstimator_.get(),
                                             compute_probabilities_,
                                             features.seed_features);
    }

    output_tag_infos->emplace_back(features, jet_ref);
  }

  iEvent.put(std::move(output_tag_infos));
}

void DeepFlavourTagInfoProducer::checkEventSetup(const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace edm::eventsetup;

  const EventSetupRecord& re2D = iSetup.get<BTagTrackProbability2DRcd>();
  const EventSetupRecord& re3D = iSetup.get<BTagTrackProbability3DRcd>();
  unsigned long long cacheId2D = re2D.cacheIdentifier();
  unsigned long long cacheId3D = re3D.cacheIdentifier();
  if (cacheId2D != calibrationCacheId2D_ || cacheId3D != calibrationCacheId3D_)  //Calibration changed
  {
    ESHandle<TrackProbabilityCalibration> calib2DHandle = iSetup.getHandle(calib2d_token_);
    ESHandle<TrackProbabilityCalibration> calib3DHandle = iSetup.getHandle(calib3d_token_);
    probabilityEstimator_ =
        std::make_unique<HistogramProbabilityEstimator>(calib3DHandle.product(), calib2DHandle.product());
  }

  calibrationCacheId3D_ = cacheId3D;
  calibrationCacheId2D_ = cacheId2D;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DeepFlavourTagInfoProducer);
