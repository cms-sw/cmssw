#ifndef RecoLocalMuon_GEMCSCSegment_GEMCSCCoincidenceRateAnalyzer_h
#define RecoLocalMuon_GEMCSCSegment_GEMCSCCoincidenceRateAnalyzer_h
/** \class GEMCSCCoincidenceRateAnalyzer
 *
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */
#include <algorithm>
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/GEMRecHit/interface/GEMCSCSegmentCollection.h"
#include "DataFormats/GEMDigi/interface/GEMVFATStatusCollection.h"
#include "DataFormats/GEMDigi/interface/GEMOHStatusCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TTree.h"

class GEMCSCCoincidenceRateAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit GEMCSCCoincidenceRateAnalyzer(const edm::ParameterSet&);
  ~GEMCSCCoincidenceRateAnalyzer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  void resetBranches();
  bool analyzeGEMCSCSegment(const GEMCSCSegment&,
                            const GEMOHStatusCollection*,
                            const GEMVFATStatusCollection*,
                            const std::vector<const CSCSegment*>&);

  bool checkGEMChamberStatus(const GEMDetId& chamber_id, const GEMOHStatusCollection*, const GEMVFATStatusCollection*);
  std::vector<const CSCSegment*> findMuonSegments(const reco::MuonCollection*);
  bool isMuonSegment(const CSCSegment&, const std::vector<const CSCSegment*>);
  bool checkCSCChamberType(const CSCDetId&);

  const edm::EDGetTokenT<GEMCSCSegmentCollection> gem_csc_segment_collection_token_;
  const edm::EDGetTokenT<GEMOHStatusCollection> gem_oh_status_collection_token_;
  const edm::EDGetTokenT<GEMVFATStatusCollection> gem_vfat_status_collection_token_;
  const edm::EDGetTokenT<reco::MuonCollection> muon_collection_token_;
  const std::string log_category_;
  const bool use_gem_daq_status_;
  const std::vector<unsigned int> csc_whitelist_;

  //
  TTree* tree_;
  //// GEMCSCSegment
  int b_region_;
  int b_station_;
  int b_ring_;
  int b_chamber_;
  bool b_gem_chamber_has_error_;
  float b_norm_chi2_;
  //// CSCSegment
  float b_csc_norm_chi2_;
  int b_csc_num_hit_;
  bool b_csc_is_muon_;
  //// GEMRecHit
  int b_gem_num_hit;
  std::vector<int> b_gem_layer_;
  std::vector<int> b_gem_ieta_;
  std::vector<int> b_gem_first_strip_;
  std::vector<int> b_gem_cls_;
};

GEMCSCCoincidenceRateAnalyzer::GEMCSCCoincidenceRateAnalyzer(const edm::ParameterSet& ps)
    : gem_csc_segment_collection_token_(
          consumes<GEMCSCSegmentCollection>(ps.getUntrackedParameter<edm::InputTag>("gemcscSegmentTag"))),
      gem_oh_status_collection_token_(
          consumes<GEMOHStatusCollection>(ps.getUntrackedParameter<edm::InputTag>("ohStatusTag"))),
      gem_vfat_status_collection_token_(
          consumes<GEMVFATStatusCollection>(ps.getUntrackedParameter<edm::InputTag>("vfatStatusTag"))),
      muon_collection_token_(consumes<reco::MuonCollection>(ps.getUntrackedParameter<edm::InputTag>("muonTag"))),
      log_category_(ps.getUntrackedParameter<std::string>("logCategory")),
      use_gem_daq_status_(ps.getUntrackedParameter<bool>("useGEMDAQStatus")),
      csc_whitelist_(ps.getUntrackedParameter<std::vector<unsigned int> >("cscWhitelist")) {
  usesResource(TFileService::kSharedResource);

  edm::Service<TFileService> file_service;
  tree_ = file_service->make<TTree>("gemcsc", "gemcsc");

  tree_->Branch("region", &b_region_);
  tree_->Branch("station", &b_station_);
  tree_->Branch("ring", &b_ring_);
  tree_->Branch("chamber", &b_chamber_);
  tree_->Branch("gem_chamber_has_error", &b_gem_chamber_has_error_, "gem_chamber_has_error/O");

  tree_->Branch("norm_chi2", &b_norm_chi2_);
  tree_->Branch("csc_norm_chi2", &b_csc_norm_chi2_);
  tree_->Branch("csc_num_hit", &b_csc_num_hit_);
  tree_->Branch("csc_is_muon", &b_csc_is_muon_);

  tree_->Branch("gem_num_hit", &b_gem_num_hit);
  tree_->Branch("gem_layer", &b_gem_layer_);
  tree_->Branch("gem_ieta", &b_gem_ieta_);
  tree_->Branch("gem_first_strip", &b_gem_first_strip_);
  tree_->Branch("gem_cls", &b_gem_cls_);
}

void GEMCSCCoincidenceRateAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("ohStatusTag", edm::InputTag("muonGEMDigis", "OHStatus"));
  desc.addUntracked<edm::InputTag>("vfatStatusTag", edm::InputTag("muonGEMDigis", "VFATStatus"));
  desc.addUntracked<std::string>("logCategory", "GEMCSCCoincidenceRateAnalyzer");
  desc.addUntracked<edm::InputTag>("gemcscSegmentTag", edm::InputTag("gemcscSegments"));
  desc.addUntracked<edm::InputTag>("muonTag", edm::InputTag("muons"));
  desc.addUntracked<bool>("useGEMDAQStatus", false);
  desc.addUntracked<std::vector<unsigned int> >("cscWhitelist",
                                                {CSCDetId::iChamberType(1, 1), CSCDetId::iChamberType(2, 1)});
  descriptions.addWithDefaultLabel(desc);
}

void GEMCSCCoincidenceRateAnalyzer::analyze(const edm::Event& event, const edm::EventSetup&) {
  /////////////////////////////////////////////////////////////////////////////
  // get data from edm::Event
  /////////////////////////////////////////////////////////////////////////////
  const GEMCSCSegmentCollection* gem_csc_segment_collection = nullptr;
  if (const edm::Handle<GEMCSCSegmentCollection> handle = event.getHandle(gem_csc_segment_collection_token_)) {
    gem_csc_segment_collection = handle.product();

  } else {
    edm::LogError(log_category_) << "failed to get GEMCSCSegmentCollection";
    return;
  }

  const GEMOHStatusCollection* gem_oh_status_collection = nullptr;
  const GEMVFATStatusCollection* gem_vfat_status_collection = nullptr;
  if (use_gem_daq_status_) {
    if (auto handle = event.getHandle(gem_oh_status_collection_token_)) {
      gem_oh_status_collection = handle.product();

    } else {
      edm::LogError(log_category_) << "failed to get OHVFATStatusCollection";
      return;
    }

    if (auto handle = event.getHandle(gem_vfat_status_collection_token_)) {
      gem_vfat_status_collection = handle.product();

    } else {
      edm::LogError(log_category_) << "failed to get GEMVFATStatusCollection";
      return;
    }
  }

  const reco::MuonCollection* muon_collection = nullptr;
  if (const edm::Handle<reco::MuonCollection> handle = event.getHandle(muon_collection_token_)) {
    muon_collection = handle.product();

  } else {
    edm::LogError(log_category_) << "failed to get reco::MuonCollection";
    return;
  }

  /////////////////////////////////////////////////////////////////////////////
  // analyze
  /////////////////////////////////////////////////////////////////////////////
  const std::vector<const CSCSegment*> muon_segment_vec = findMuonSegments(muon_collection);

  for (edm::OwnVector<GEMCSCSegment>::const_iterator iter = gem_csc_segment_collection->begin();
       iter != gem_csc_segment_collection->end();
       iter++) {
    const GEMCSCSegment& gem_csc_segment = *iter;
    if (not checkCSCChamberType(gem_csc_segment.cscDetId())) {
      continue;
    }
    resetBranches();
    if (analyzeGEMCSCSegment(gem_csc_segment, gem_oh_status_collection, gem_vfat_status_collection, muon_segment_vec)) {
      tree_->Fill();
    }
  }  // GEMCSCSegment
}

void GEMCSCCoincidenceRateAnalyzer::resetBranches() {
  // detector
  b_region_ = 0;
  b_station_ = 0;
  b_ring_ = 0;
  b_chamber_ = 0;
  b_gem_chamber_has_error_ = false;
  // GEMCSCSegment
  b_norm_chi2_ = 0.0f;
  // CSCSegment
  b_csc_norm_chi2_ = 0.0f;
  b_csc_num_hit_ = 0;
  b_csc_is_muon_ = false;
  // GEMRecHit
  b_gem_num_hit = 0;
  b_gem_layer_.clear();
  b_gem_ieta_.clear();
  b_gem_first_strip_.clear();
  b_gem_cls_.clear();
}

// Main function to analyzer GEMCSCSegment.
// Returns a boolean indicating whether the GEMCSCSegment analysis is successful.
bool GEMCSCCoincidenceRateAnalyzer::analyzeGEMCSCSegment(const GEMCSCSegment& gem_csc_segment,
                                                         const GEMOHStatusCollection* gem_oh_status_collection,
                                                         const GEMVFATStatusCollection* gem_vfat_status_collection,
                                                         const std::vector<const CSCSegment*>& muon_segment_vec) {
  const CSCDetId csc_id = gem_csc_segment.cscDetId();
  const CSCSegment csc_segment = gem_csc_segment.cscSegment();
  const std::vector<GEMRecHit>& gem_hit_vec = gem_csc_segment.gemRecHits();
  // GEMDetId(int region, int ring, int station, int layer, int chamber, int ieta)
  const GEMDetId gem_chamber_id{csc_id.zendcap(), 1, csc_id.station(), 0, csc_id.chamber(), 0};

  // detector
  b_region_ = csc_id.zendcap();
  b_station_ = csc_id.station();
  b_ring_ = csc_id.ring();
  b_chamber_ = csc_id.chamber();
  b_gem_chamber_has_error_ =
      use_gem_daq_status_ ? checkGEMChamberStatus(gem_chamber_id, gem_oh_status_collection, gem_vfat_status_collection)
                          : false;
  // GEMCSCSegment
  b_norm_chi2_ = static_cast<float>(gem_csc_segment.chi2() / gem_csc_segment.nRecHits());
  // CSCSegment
  b_csc_norm_chi2_ = static_cast<float>(csc_segment.chi2() / csc_segment.nRecHits());
  b_csc_num_hit_ = csc_segment.nRecHits();
  b_csc_is_muon_ = isMuonSegment(csc_segment, muon_segment_vec);
  // GEMRecHit
  b_gem_num_hit = gem_hit_vec.size();

  b_gem_layer_.reserve(b_gem_num_hit);
  b_gem_ieta_.reserve(b_gem_num_hit);
  b_gem_first_strip_.reserve(b_gem_num_hit);
  b_gem_cls_.reserve(b_gem_num_hit);

  for (const GEMRecHit& gem_hit : gem_hit_vec) {
    const GEMDetId gem_id = gem_hit.gemId();

    b_gem_layer_.push_back(gem_id.layer());
    b_gem_ieta_.push_back(gem_id.ieta());
    b_gem_first_strip_.push_back(gem_hit.firstClusterStrip());
    b_gem_cls_.push_back(gem_hit.clusterSize());
  }  // GEMRecHit

  return true;
}

// Returns a vector of CSCSegments, which are matched with muons.
std::vector<const CSCSegment*> GEMCSCCoincidenceRateAnalyzer::findMuonSegments(
    const reco::MuonCollection* muon_collection) {
  std::vector<const CSCSegment*> muon_segment_vec;

  for (const reco::Muon& muon : *muon_collection) {
    for (const reco::MuonChamberMatch& chamber_match : muon.matches()) {
      if (chamber_match.detector() != MuonSubdetId::CSC) {
        continue;
      }

      if (checkCSCChamberType(chamber_match.id)) {
        for (const reco::MuonSegmentMatch& segment_match : chamber_match.segmentMatches) {
          if (segment_match.isMask(reco::MuonSegmentMatch::BestInStationByDR)) {
            muon_segment_vec.push_back(segment_match.cscSegmentRef.get());
            break;
          }
        }  // MuonSegmentMatch
      }    // checkCSCChamberType
    }      // MuonChamberMatch
  }        // MuonCollection

  return muon_segment_vec;
}

// Returns a boolean indicating whether the given CSCSegment is matched with a
// muon.
// TODO better segment comparison
bool GEMCSCCoincidenceRateAnalyzer::isMuonSegment(const CSCSegment& csc_segment,
                                                  const std::vector<const CSCSegment*> muon_segment_vec) {
  bool found = false;

  const CSCDetId csc_id = csc_segment.cscDetId();
  for (const CSCSegment* muon_segment : muon_segment_vec) {
    if (csc_id != muon_segment->cscDetId())
      continue;
    if (csc_segment.localPosition().x() != muon_segment->localPosition().x())
      continue;
    if (csc_segment.localPosition().y() != muon_segment->localPosition().y())
      continue;
    if (csc_segment.localPosition().z() != muon_segment->localPosition().z())
      continue;
    if (csc_segment.time() != muon_segment->time())
      continue;

    found = true;
  }

  return found;
}

// Returns a boolean indicating whether the GEMChamber has any DAQ error.
bool GEMCSCCoincidenceRateAnalyzer::checkGEMChamberStatus(const GEMDetId& chamber_id,
                                                          const GEMOHStatusCollection* oh_status_collection,
                                                          const GEMVFATStatusCollection* vfat_status_collection) {
  const bool has_error = true;
  if (not use_gem_daq_status_) {
    edm::LogError(log_category_)
        << "useGEMDAQStatus is false but checkGEMChamberStatus is called. gem_chamber_has_error will be set to false";
    return not has_error;
  }

  for (auto iter = oh_status_collection->begin(); iter != oh_status_collection->end(); iter++) {
    const auto [oh_id, range] = (*iter);
    if (chamber_id != oh_id) {
      continue;
    }

    for (auto oh_status = range.first; oh_status != range.second; oh_status++) {
      if (oh_status->isBad()) {
        // GEMOHStatus is bad. Mask this chamber.
        return has_error;
      }  // isBad
    }    // range
  }      // collection

  for (auto iter = vfat_status_collection->begin(); iter != vfat_status_collection->end(); iter++) {
    const auto [vfat_id, range] = (*iter);
    if (chamber_id != vfat_id.chamberId()) {
      continue;
    }
    for (auto vfat_status = range.first; vfat_status != range.second; vfat_status++) {
      if (vfat_status->isBad()) {
        return has_error;
      }
    }  // range
  }    // collection

  return not has_error;
}

// Returns a boolean indicating whether to allow a CSCDetId.
bool GEMCSCCoincidenceRateAnalyzer::checkCSCChamberType(const CSCDetId& csc_id) {
  const auto csc_chamber_type = static_cast<unsigned int>(csc_id.iChamberType());
  return std::find(csc_whitelist_.begin(), csc_whitelist_.end(), csc_chamber_type) != csc_whitelist_.end();
}

DEFINE_FWK_MODULE(GEMCSCCoincidenceRateAnalyzer);
#endif  // RecoLocalMuon_GEMCSCSegment_GEMCSCCoincidenceRateAnalyzer_h
