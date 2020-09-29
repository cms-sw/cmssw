#ifndef RecoParticleFlow_PFProducer_PFTables_h
#define RecoParticleFlow_PFProducer_PFTables_h

#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementTrack.h"
#include "RecoParticleFlow/PFProducer/interface/TableDefinitions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

namespace edm::soa {
  TrackTableVertex makeTrackTableVertex(std::vector<reco::PFBlockElement*> const& targetSet);

  TrackTableExtrapolation makeTrackTable(std::vector<reco::PFBlockElement*> const& objects,
                                         reco::PFTrajectoryPoint::LayerType layerType);
  TrackTableExtrapolation makeTrackTable(std::vector<const reco::PFBlockElementGsfTrack*> const& objects,
                                         reco::PFTrajectoryPoint::LayerType layerType);
  TrackTableExtrapolation makeTrackTable(std::vector<const reco::PFBlockElementBrem*> const& objects,
                                         reco::PFTrajectoryPoint::LayerType layerType);
  template <class RecTrackType>
  TrackTableExtrapolation makeTrackTable(std::vector<RecTrackType> const& objects,
                                         reco::PFTrajectoryPoint::LayerType layerType);

  ConvRefTable makeConvRefTable(const std::vector<reco::ConversionRef>& convrefs);
  ConvBremTable makeConvBremTable(const std::vector<reco::PFRecTrackRef>& convbrems);
  BremTable makeBremTable(const std::vector<const reco::PFBlockElementBrem*>& brems);

  RecHitTable makeRecHitTable(std::vector<const reco::PFRecHitFraction*> const& objects);
  SuperClusterRecHitTable makeSuperClusterRecHitTable(std::vector<const std::pair<DetId, float>*> const& objects);
  ClusterTable makeClusterTable(std::vector<const reco::PFBlockElementCluster*> const& objects);
  SuperClusterTable makeSuperClusterTable(std::vector<const reco::PFBlockElementSuperCluster*> const& objects);
  GSFTable makeGSFTable(std::vector<const reco::PFBlockElementGsfTrack*> const& objects);
}  // namespace edm::soa

//this is a transient data structure, made in order to collect the data that PFBlockAlgo needs
//holds all input data to PFBlockAlgo computations

template <class ClusterTable, class RecHitTable>
struct PFClusterTables {
  ClusterTable cluster_table;
  RecHitTable rechit_table;
  std::vector<size_t> cluster_to_element;
  std::vector<size_t> element_to_cluster;
  std::unordered_map<size_t, std::set<size_t>> rechit_to_cluster;
  std::unordered_map<size_t, std::set<size_t>> cluster_to_rechit;

  PFClusterTables() = default;
  PFClusterTables(ClusterTable&& cluster_table,
                  RecHitTable&& rechit_table,
                  std::vector<size_t>&& cluster_to_element,
                  std::vector<size_t>&& element_to_cluster,
                  std::unordered_map<size_t, std::set<size_t>>&& rechit_to_cluster,
                  std::unordered_map<size_t, std::set<size_t>>&& cluster_to_rechit)
      : cluster_table{cluster_table},
        rechit_table{rechit_table},
        cluster_to_element{cluster_to_element},
        element_to_cluster{element_to_cluster},
        rechit_to_cluster{rechit_to_cluster},
        cluster_to_rechit{cluster_to_rechit} {};
};

struct PFTables {
  std::vector<size_t> track_to_element;
  std::vector<size_t> element_to_track;
  std::vector<std::vector<size_t>> track_to_convrefs;

  edm::soa::ConvRefTable convref_table;
  edm::soa::TrackTableVertex track_table_vertex;
  edm::soa::TrackTableExtrapolation track_table_ecalshowermax;
  edm::soa::TrackTableExtrapolation track_table_hcalent;
  edm::soa::TrackTableExtrapolation track_table_hcalex;
  edm::soa::TrackTableExtrapolation track_table_vfcalent;
  edm::soa::TrackTableExtrapolation track_table_ho;

  edm::soa::GSFTable gsf_table;
  edm::soa::ConvBremTable gsf_convbrem_table;
  edm::soa::TrackTableExtrapolation gsf_table_ecalshowermax;
  edm::soa::TrackTableExtrapolation gsf_table_hcalent;
  edm::soa::TrackTableExtrapolation gsf_table_hcalex;

  std::vector<size_t> gsf_to_element;
  std::vector<size_t> element_to_gsf;
  std::vector<std::vector<size_t>> gsf_to_convbrem;

  std::vector<size_t> element_to_brem;
  edm::soa::BremTable brem_table;
  edm::soa::TrackTableExtrapolation brem_table_ecalshowermax;
  edm::soa::TrackTableExtrapolation brem_table_hcalent;

  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_ps1;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_ps2;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_hcal;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_ecal;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_ho;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_hfem;
  PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable> clusters_hfhad;
  PFClusterTables<edm::soa::SuperClusterTable, edm::soa::SuperClusterRecHitTable> clusters_sc;

  //this is needed since the KDTreeLinkers of different type need to have the same interface, but underneath may need to access different data
  //according to KDTreeLinkerBase::_fieldType and KDTreeLinkerBase::_targetType
  const PFClusterTables<edm::soa::ClusterTable, edm::soa::RecHitTable>& getClusterTable(
      reco::PFBlockElement::Type type) const {
    LogDebug("PFTables") << "getClusterTable " << type;
    if (type == reco::PFBlockElement::HCAL) {
      return clusters_hcal;
    } else if (type == reco::PFBlockElement::ECAL) {
      return clusters_ecal;
    } else if (type == reco::PFBlockElement::HFEM) {
      return clusters_hfem;
    } else if (type == reco::PFBlockElement::HFHAD) {
      return clusters_hfhad;
    } else if (type == reco::PFBlockElement::HO) {
      return clusters_ho;
    } else if (type == reco::PFBlockElement::PS1) {
      return clusters_ps1;
    } else if (type == reco::PFBlockElement::PS2) {
      return clusters_ps2;
    }

    //called with wrong or unimplemented arguments, should terminate the execution
    throw cms::Exception("LogicError") << "unhandled type in getClusterTable";
  }

  const edm::soa::TrackTableExtrapolation& getTrackTable(reco::PFTrajectoryPoint::LayerType layer) const {
    LogDebug("PFTables") << "getTrackTable " << layer;
    if (layer == reco::PFTrajectoryPoint::HCALEntrance) {
      return track_table_hcalent;
    } else if (layer == reco::PFTrajectoryPoint::HCALExit) {
      return track_table_hcalex;
    } else if (layer == reco::PFTrajectoryPoint::VFcalEntrance) {
      return track_table_vfcalent;
    }

    //called with wrong or unimplemented arguments, should terminate the execution
    throw cms::Exception("LogicError") << "unhandled type in getTrackTable";
  }
};

#endif