#ifndef RecoParticleFlow_PFClusterTools_LinkByRecHit_h
#define RecoParticleFlow_PFClusterTools_LinkByRecHit_h

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"
#include "RecoParticleFlow/PFProducer/interface/TableDefinitions.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class LinkByRecHit {
public:
  LinkByRecHit(){};
  ~LinkByRecHit(){};

  /// computes a chisquare
  static double computeDist(double eta1, double phi1, double eta2, double phi2, bool etaPhi = true);

  static double computeTrackHCALDist(
      bool checkExit,
      size_t itrack,
      size_t ihcal,
      edm::soa::TableView<edm::soa::col::pf::cluster::Eta, edm::soa::col::pf::cluster::Phi> clusterTable,
      edm::soa::TableView<edm::soa::col::pf::track::ExtrapolationValid,
                          edm::soa::col::pf::track::Eta,
                          edm::soa::col::pf::track::Phi,
                          edm::soa::col::pf::track::Posx,
                          edm::soa::col::pf::track::Posy,
                          edm::soa::col::pf::track::Posz,
                          edm::soa::col::pf::track::PosR> trackTableEntrance,
      edm::soa::TableView<edm::soa::col::pf::track::ExtrapolationValid,
                          edm::soa::col::pf::track::Eta,
                          edm::soa::col::pf::track::Phi,
                          edm::soa::col::pf::track::Posx,
                          edm::soa::col::pf::track::Posy,
                          edm::soa::col::pf::track::Posz,
                          edm::soa::col::pf::track::PosR> trackTableExit);

  //tests association between a track and a cluster by rechit
  static double testTrackAndClusterByRecHit(const reco::PFRecTrack& track,
                                            const reco::PFCluster& cluster,
                                            bool isBrem = false,
                                            bool debug = false);

  static double testTrackAndClusterByRecHit(size_t icluster,
                                            std::set<size_t> cluster_rechits,
                                            edm::soa::TableView<edm::soa::col::pf::cluster::Eta,
                                                                edm::soa::col::pf::cluster::Phi,
                                                                edm::soa::col::pf::cluster::Posz,
                                                                edm::soa::col::pf::cluster::Layer,
                                                                edm::soa::col::pf::cluster::FracsNbr> cluster_table,

                                            edm::soa::TableView<edm::soa::col::pf::rechit::DetIdValue,
                                                                edm::soa::col::pf::rechit::Fraction,
                                                                edm::soa::col::pf::rechit::Eta,
                                                                edm::soa::col::pf::rechit::Phi,
                                                                edm::soa::col::pf::rechit::Posx,
                                                                edm::soa::col::pf::rechit::Posy,
                                                                edm::soa::col::pf::rechit::Posz,
                                                                edm::soa::col::pf::rechit::Corner0x,
                                                                edm::soa::col::pf::rechit::Corner0y,
                                                                edm::soa::col::pf::rechit::Corner0z,
                                                                edm::soa::col::pf::rechit::Corner1x,
                                                                edm::soa::col::pf::rechit::Corner1y,
                                                                edm::soa::col::pf::rechit::Corner1z,
                                                                edm::soa::col::pf::rechit::Corner2x,
                                                                edm::soa::col::pf::rechit::Corner2y,
                                                                edm::soa::col::pf::rechit::Corner2z,
                                                                edm::soa::col::pf::rechit::Corner3x,
                                                                edm::soa::col::pf::rechit::Corner3y,
                                                                edm::soa::col::pf::rechit::Corner3z,
                                                                edm::soa::col::pf::rechit::Corner0eta,
                                                                edm::soa::col::pf::rechit::Corner0phi,
                                                                edm::soa::col::pf::rechit::Corner1eta,
                                                                edm::soa::col::pf::rechit::Corner1phi,
                                                                edm::soa::col::pf::rechit::Corner2eta,
                                                                edm::soa::col::pf::rechit::Corner2phi,
                                                                edm::soa::col::pf::rechit::Corner3eta,
                                                                edm::soa::col::pf::rechit::Corner3phi> rechit_table,

                                            size_t itrack,
                                            edm::soa::TableView<edm::soa::col::pf::track::Pt> tracks_vtx_table,
                                            edm::soa::TableView<edm::soa::col::pf::track::ExtrapolationValid,
                                                                edm::soa::col::pf::track::Eta,
                                                                edm::soa::col::pf::track::Phi,
                                                                edm::soa::col::pf::track::Posx,
                                                                edm::soa::col::pf::track::Posy,
                                                                edm::soa::col::pf::track::Posz,
                                                                edm::soa::col::pf::track::PosR> tracks_ecal_table,
                                            edm::soa::TableView<edm::soa::col::pf::track::ExtrapolationValid,
                                                                edm::soa::col::pf::track::Eta,
                                                                edm::soa::col::pf::track::Phi,
                                                                edm::soa::col::pf::track::Posx,
                                                                edm::soa::col::pf::track::Posy,
                                                                edm::soa::col::pf::track::Posz,
                                                                edm::soa::col::pf::track::PosR> tracks_hcalent_table,
                                            edm::soa::TableView<edm::soa::col::pf::track::ExtrapolationValid,
                                                                edm::soa::col::pf::track::Eta,
                                                                edm::soa::col::pf::track::Phi,
                                                                edm::soa::col::pf::track::Posx,
                                                                edm::soa::col::pf::track::Posy,
                                                                edm::soa::col::pf::track::Posz,
                                                                edm::soa::col::pf::track::PosR> tracks_hcalexit_table,
                                            edm::soa::TableView<edm::soa::col::pf::track::ExtrapolationValid,
                                                                edm::soa::col::pf::track::Eta,
                                                                edm::soa::col::pf::track::Phi,
                                                                edm::soa::col::pf::track::Posx,
                                                                edm::soa::col::pf::track::Posy,
                                                                edm::soa::col::pf::track::Posz,
                                                                edm::soa::col::pf::track::PosR> tracks_ho_table,
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
                                         edm::soa::TableView<edm::soa::col::pf::cluster::Posx,
                                                             edm::soa::col::pf::cluster::Posy,
                                                             edm::soa::col::pf::cluster::Posz> cluster_em_table,
                                         edm::soa::TableView<edm::soa::col::pf::cluster::Posx,
                                                             edm::soa::col::pf::cluster::Posy,
                                                             edm::soa::col::pf::cluster::Posz> cluster_had_table);
};

#endif
