#ifndef RecoParticleFlow_PFProducer_PFTables_h
#define RecoParticleFlow_PFProducer_PFTables_h

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFProducer/interface/TableDefinitions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm::soa {
  TrackTableVertex makeTrackTableVertex(std::vector<reco::PFBlockElement*> const& targetSet);

  TrackTableExtrapolation makeTrackTable(std::vector<reco::PFBlockElement*> const& targetSet,
                                         reco::PFTrajectoryPoint::LayerType layerType);
  TrackTableExtrapolation makeTrackTable(std::vector<const reco::PFBlockElementGsfTrack*> const& targetSet,
                                         reco::PFTrajectoryPoint::LayerType layerType);
  ConvRefTable makeConvRefTable(const std::vector<reco::ConversionRef>& convrefs);

  RecHitTable makeRecHitTable(std::vector<const reco::PFRecHitFraction*> const& objects);
  SuperClusterRecHitTable makeSuperClusterRecHitTable(std::vector<const std::pair<DetId, float>*> const& objects);
  ClusterTable makeClusterTable(std::vector<const reco::PFBlockElementCluster*> const& objects);
  SuperClusterTable makeSuperClusterTable(std::vector<const reco::PFBlockElementSuperCluster*> const& objects);
  GSFTable makeGSFTable(std::vector<const reco::PFBlockElementGsfTrack*> const& objects);
}  // namespace edm::soa

//this is a transient data structure, made in order to collect the data that PFBlockAlgo needs
//holds all input data to PFBlockAlgo computations

template <class ClusterTable, class RecHitTable>
class PFClusterTables {
public:
  ClusterTable cluster_table_;
  RecHitTable rechit_table_;
  std::vector<size_t> cluster_to_element_;
  std::vector<size_t> element_to_cluster_;
  std::unordered_map<size_t, std::set<size_t>> rechit_to_cluster_;
  std::unordered_map<size_t, std::set<size_t>> cluster_to_rechit_;

  PFClusterTables() = default;
  PFClusterTables(ClusterTable&& _cluster_table,
                  RecHitTable&& _rechit_table,
                  std::vector<size_t>&& _cluster_to_element,
                  std::vector<size_t>&& _element_to_cluster,
                  std::unordered_map<size_t, std::set<size_t>>&& _rechit_to_cluster,
                  std::unordered_map<size_t, std::set<size_t>>&& _cluster_to_rechit)
      : cluster_table_{_cluster_table},
        rechit_table_{_rechit_table},
        cluster_to_element_{_cluster_to_element},
        element_to_cluster_{_element_to_cluster},
        rechit_to_cluster_{_rechit_to_cluster},
        cluster_to_rechit_{_cluster_to_rechit} {};

  void clear() {
    cluster_table_.resize(0);
    rechit_table_.resize(0);
    cluster_to_element_.clear();
    element_to_cluster_.clear();
    rechit_to_cluster_.clear();
    cluster_to_rechit_.clear();
  }
};

class PFTables {
public:
  std::vector<size_t> track_to_element_;
  std::vector<size_t> element_to_track_;
  std::vector<std::vector<size_t>> track_to_convrefs_;

  edm::soa::ConvRefTable convref_table_;
  edm::soa::TrackTableVertex track_table_vertex_;
  edm::soa::TrackTableExtrapolation track_table_ecalshowermax_;
  edm::soa::TrackTableExtrapolation track_table_hcalent_;
  edm::soa::TrackTableExtrapolation track_table_hcalex_;
  edm::soa::TrackTableExtrapolation track_table_vfcalent_;
  edm::soa::TrackTableExtrapolation track_table_ho_;

  edm::soa::GSFTable gsf_table_;
  edm::soa::TrackTableExtrapolation gsf_table_ecalshowermax_;
  edm::soa::TrackTableExtrapolation gsf_table_hcalent_;
  edm::soa::TrackTableExtrapolation gsf_table_hcalex_;
  //gsf_table_ho won't be used, but is needed to pass the correct number of arguments to LinkByRecHit::testTrackAndClusterByRecHit
  edm::soa::TrackTableExtrapolation gsf_table_ho_;
  std::vector<size_t> gsf_to_element_;
  std::vector<size_t> element_to_gsf_;

  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_ps1_;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_ps2_;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_hcal_;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_ecal_;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_ho_;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_hfem_;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_hfhad_;
  PFClusterTables<edm::soa::SuperClusterTable, edm::soa::SuperClusterRecHitTable> clusters_sc_;

  // edm::soa::SuperClusterTable superclusters_table_;
  // std::vector<size_t> element_to_supercluster_;

  void clear() {
    track_to_element_.clear();
    element_to_track_.clear();
    track_to_convrefs_.clear();

    convref_table_.resize(0);
    track_table_vertex_.resize(0);
    track_table_ecalshowermax_.resize(0);
    track_table_hcalent_.resize(0);
    track_table_hcalex_.resize(0);
    track_table_vfcalent_.resize(0);
    track_table_ho_.resize(0);

    gsf_table_.resize(0);
    gsf_table_ecalshowermax_.resize(0);
    gsf_table_hcalent_.resize(0);
    gsf_table_hcalex_.resize(0);
    gsf_table_ho_.resize(0);
    gsf_to_element_.clear();
    element_to_gsf_.clear();

    clusters_ps1_.clear();
    clusters_ps2_.clear();
    clusters_hcal_.clear();
    clusters_ecal_.clear();
    clusters_ho_.clear();
    clusters_hfem_.clear();
    clusters_hfhad_.clear();
    clusters_sc_.clear();

    // superclusters_table_.resize(0);
    // element_to_supercluster_.clear();
  }

  //this is needed since the KDTreeLinkers of different type need to have the same interface, but underneath may need to access different data
  //according to KDTreeLinkerBase::_fieldType and KDTreeLinkerBase::_targetType
  const PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable>& getClusterTable(
      reco::PFBlockElement::Type type) const {
    LogDebug("PFTables") << "getClusterTable " << type;
    if (type == reco::PFBlockElement::HCAL) {
      return clusters_hcal_;
    } else if (type == reco::PFBlockElement::ECAL) {
      return clusters_ecal_;
    } else if (type == reco::PFBlockElement::HFEM) {
      return clusters_hfem_;
    } else if (type == reco::PFBlockElement::HFHAD) {
      return clusters_hfhad_;
    } else if (type == reco::PFBlockElement::HO) {
      return clusters_ho_;
    } else if (type == reco::PFBlockElement::PS1) {
      return clusters_ps1_;
    } else if (type == reco::PFBlockElement::PS2) {
      return clusters_ps2_;
    } else {
      throw cms::Exception("unhandled type in getClusterTable");
    }
  }

  const edm::soa::TrackTableExtrapolation& getTrackTable(reco::PFTrajectoryPoint::LayerType layer) const {
    LogDebug("PFTables") << "getTrackTable " << layer;
    if (layer == reco::PFTrajectoryPoint::HCALEntrance) {
      return track_table_hcalent_;
    } else if (layer == reco::PFTrajectoryPoint::HCALExit) {
      return track_table_hcalex_;
    } else if (layer == reco::PFTrajectoryPoint::VFcalEntrance) {
      return track_table_vfcalent_;
    } else {
      throw cms::Exception("unhandler layer type in getTrackTable");
    }
  }
};

#endif