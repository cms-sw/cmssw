#ifndef RecoParticleFlow_PFClusterTools_LinkByRecHit_h
#define RecoParticleFlow_PFClusterTools_LinkByRecHit_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "RecoParticleFlow/PFProducer/interface/TableDefinitions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//define namespace aliases, but avoid them leaking to other files
namespace {
  namespace cluster = edm::soa::col::pf::cluster;
  namespace track = edm::soa::col::pf::track;
  namespace rechit = edm::soa::col::pf::rechit;
};  // namespace

class LinkByRecHit {
public:
  using TrackTableView = edm::soa::
      TableView<track::ExtrapolationValid, track::Eta, track::Phi, track::Posx, track::Posy, track::Posz, track::PosR>;
  using ClusterXYZTableView = edm::soa::TableView<cluster::Posx, cluster::Posy, cluster::Posz>;
  using RecHitTableView = edm::soa::TableView<rechit::DetIdValue,
                                              rechit::Fraction,
                                              rechit::Eta,
                                              rechit::Phi,
                                              rechit::Posx,
                                              rechit::Posy,
                                              rechit::Posz,
                                              rechit::CornerX,
                                              rechit::CornerY,
                                              rechit::CornerZ,
                                              rechit::CornerEta,
                                              rechit::CornerPhi>;
  LinkByRecHit(){};
  ~LinkByRecHit(){};

  /// computes a chisquare
  static double computeDist(double eta1, double phi1, double eta2, double phi2, bool etaPhi = true);

  static double computeTrackHCALDist(bool checkExit,
                                     size_t itrack,
                                     size_t ihcal,
                                     edm::soa::TableView<cluster::Eta, cluster::Phi> clusterTable,
                                     TrackTableView trackTableEntrance,
                                     TrackTableView trackTableExit);

  //tests association between a track and a cluster by rechit
  static double testTrackAndClusterByRecHit(const reco::PFRecTrack& track,
                                            const reco::PFCluster& cluster,
                                            bool isBrem = false,
                                            bool debug = false);

  static double testTrackAndClusterByRecHit(
      size_t icluster,
      std::set<size_t> cluster_rechits,
      edm::soa::TableView<cluster::Eta, cluster::Phi, cluster::Posz, cluster::Layer, cluster::FracsNbr> cluster_table,

      RecHitTableView rechit_table,

      size_t itrack,
      edm::soa::TableView<track::Pt> tracks_vtx_table,
      TrackTableView tracks_ecal_table,
      TrackTableView tracks_hcalent_table,
      TrackTableView tracks_hcalexit_table,
      TrackTableView tracks_ho_table,
      bool isBrem);

  //tests association between ECAL and PS clusters by rechit
  static double testECALAndPSByRecHit(const reco::PFCluster& clusterECAL,
                                      const reco::PFCluster& clusterPS,
                                      bool debug = false);

  /// test association between HFEM and HFHAD, by rechit
  static double testHFEMAndHFHADByRecHit(const reco::PFCluster& clusterHFEM,
                                         const reco::PFCluster& clusterHFHAD,
                                         bool debug = false);

  static double testHFEMAndHFHADByRecHit(size_t icluster_em,
                                         size_t icluster_had,
                                         ClusterXYZTableView cluster_em_table,
                                         ClusterXYZTableView cluster_had_table);
};

#endif
