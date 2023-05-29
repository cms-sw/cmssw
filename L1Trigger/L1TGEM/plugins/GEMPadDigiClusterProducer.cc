/**
 *  \class GEMPadDigiClusterProducer
 *
 *  Produces GEM pad clusters from at most 8 adjacent GEM pads.
 *  Clusters are used downstream in the CSC local trigger to build
 *  GEM-CSC triggers and in the muon trigger to build EMTF tracks
 *
 *  Based on documentation provided by the GEM firmware architects
 *
 *  \author Sven Dildick (TAMU), updated by Giovanni Mocellin (UCDavis)
 *
 *  *****************************************************
 *  ** Notes on chambers and cluster packing algorithm **
 *  *****************************************************
 *
 *  Based on: https://gitlab.cern.ch/emu/0xbefe/-/tree/devel/gem/hdl/cluster_finding/README.org
 *  (Andrew Peck, 2020/06/26)
 *
 *  GE1/1 chamber has 8 iEta partitions and 1 OH
 *  GE2/1 chamber has 16 iEta partitions and 4 OH (one per module)
 *
 *  Both GE1/1 and GE2/1 have 384 strips = 192 pads per iEta partition
 *
 *  GE1/1 OH has 4 clustering partitions, each covering 2 iEta partitions
 *  GE2/1 OH has 4 clustering partitions, each covering 1 iEta partition
 *
 *  Each clustering partition finds up to 4 clusters per BX, which are
 *  then sent to the sorter. The sorting of the clusters favors lower
 *  eta partitions and lower pad numbers.
 *
 *  The first 8 clusters are selected and sent out through optical fibers.
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

#include <string>
#include <map>
#include <vector>

class GEMPadDigiClusterProducer : public edm::stream::EDProducer<> {
public:
  typedef std::vector<GEMPadDigiCluster> GEMPadDigiClusters;
  typedef std::map<GEMDetId, GEMPadDigiClusters> GEMPadDigiClusterContainer;

  explicit GEMPadDigiClusterProducer(const edm::ParameterSet& ps);

  ~GEMPadDigiClusterProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void buildClusters(const GEMPadDigiCollection& pads, GEMPadDigiClusterContainer& out_clusters) const;
  void selectClusters(const GEMPadDigiClusterContainer& in_clusters, GEMPadDigiClusterCollection& out) const;
  template <class T>
  void checkValid(const T& cluster, const GEMDetId& id) const;

  /// Name of input digi Collection
  edm::EDGetTokenT<GEMPadDigiCollection> pad_token_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> geom_token_;
  edm::InputTag pads_;

  unsigned int nPartitionsGE11_;
  unsigned int nPartitionsGE21_;
  unsigned int maxClustersPartitionGE11_;
  unsigned int maxClustersPartitionGE21_;
  unsigned int nOHGE11_;
  unsigned int nOHGE21_;
  unsigned int maxClustersOHGE11_;
  unsigned int maxClustersOHGE21_;
  unsigned int maxClusterSize_;
  bool sendOverflowClusters_;

  const GEMGeometry* geometry_;
};

GEMPadDigiClusterProducer::GEMPadDigiClusterProducer(const edm::ParameterSet& ps) : geometry_(nullptr) {
  pads_ = ps.getParameter<edm::InputTag>("InputCollection");
  nPartitionsGE11_ = ps.getParameter<unsigned int>("nPartitionsGE11");
  nPartitionsGE21_ = ps.getParameter<unsigned int>("nPartitionsGE21");
  maxClustersPartitionGE11_ = ps.getParameter<unsigned int>("maxClustersPartitionGE11");
  maxClustersPartitionGE21_ = ps.getParameter<unsigned int>("maxClustersPartitionGE21");
  nOHGE11_ = ps.getParameter<unsigned int>("nOHGE11");
  nOHGE21_ = ps.getParameter<unsigned int>("nOHGE21");
  maxClustersOHGE11_ = ps.getParameter<unsigned int>("maxClustersOHGE11");
  maxClustersOHGE21_ = ps.getParameter<unsigned int>("maxClustersOHGE21");
  maxClusterSize_ = ps.getParameter<unsigned int>("maxClusterSize");
  sendOverflowClusters_ = ps.getParameter<bool>("sendOverflowClusters");

  if (sendOverflowClusters_) {
    maxClustersOHGE11_ *= 2;
    maxClustersOHGE21_ *= 2;
  }

  pad_token_ = consumes<GEMPadDigiCollection>(pads_);
  geom_token_ = esConsumes<GEMGeometry, MuonGeometryRecord, edm::Transition::BeginRun>();

  produces<GEMPadDigiClusterCollection>();
  consumes<GEMPadDigiCollection>(pads_);
}

GEMPadDigiClusterProducer::~GEMPadDigiClusterProducer() {}

void GEMPadDigiClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputCollection", edm::InputTag("simMuonGEMPadDigis"));
  desc.add<unsigned int>("nPartitionsGE11", 4);           // Number of clusterizer partitions per OH
  desc.add<unsigned int>("nPartitionsGE21", 4);           // Number of clusterizer partitions per OH
  desc.add<unsigned int>("maxClustersPartitionGE11", 4);  // Maximum number of clusters per clusterizer partition
  desc.add<unsigned int>("maxClustersPartitionGE21", 4);  // Maximum number of clusters per clusterizer partition
  desc.add<unsigned int>("nOHGE11", 1);                   // Number of OH boards per chamber
  desc.add<unsigned int>("nOHGE21", 4);                   // Number of OH boards per chamber
  desc.add<unsigned int>("maxClustersOHGE11", 8);         // Maximum number of clusters per OH
  desc.add<unsigned int>("maxClustersOHGE21", 8);         // Maximum number of clusters per OH
  desc.add<unsigned int>("maxClusterSize", 8);            // Maximum cluster size (number of pads)
  desc.add<bool>("sendOverflowClusters", false);

  descriptions.add("simMuonGEMPadDigiClustersDef", desc);
}

void GEMPadDigiClusterProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  edm::ESHandle<GEMGeometry> hGeom = eventSetup.getHandle(geom_token_);
  geometry_ = &*hGeom;
}

void GEMPadDigiClusterProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Handle<GEMPadDigiCollection> hpads;
  e.getByToken(pad_token_, hpads);

  // Create empty output
  std::unique_ptr<GEMPadDigiClusterCollection> pClusters(new GEMPadDigiClusterCollection());

  // build the proto clusters (per partition)
  GEMPadDigiClusterContainer proto_clusters;
  buildClusters(*(hpads.product()), proto_clusters);

  // sort and select clusters per chamber, per OH, per partition number and per pad number
  selectClusters(proto_clusters, *pClusters);

  // store them in the event
  e.put(std::move(pClusters));
}

void GEMPadDigiClusterProducer::buildClusters(const GEMPadDigiCollection& det_pads,
                                              GEMPadDigiClusterContainer& proto_clusters) const {
  // clear the container
  proto_clusters.clear();

  // construct clusters
  for (const auto& part : geometry_->etaPartitions()) {
    // clusters are not build for ME0
    // -> ignore hits from station 0
    if (part->isME0())
      continue;

    GEMPadDigiClusters all_pad_clusters;

    auto pads = det_pads.get(part->id());
    std::vector<uint16_t> cl;
    int startBX = 99;

    for (auto d = pads.first; d != pads.second; ++d) {
      // check if the input pad is valid
      checkValid(*d, part->id());

      // number of eta partitions
      unsigned nPart = d->nPartitions();

      if (cl.empty()) {
        cl.push_back((*d).pad());
      } else {
        if ((*d).bx() == startBX and            // same bunch crossing
            (*d).pad() == cl.back() + 1         // pad difference is 1
            and cl.size() < maxClusterSize_) {  // max 8 in cluster
          cl.push_back((*d).pad());
        } else {
          // put the current cluster in the proto collection
          GEMPadDigiCluster pad_cluster(cl, startBX, part->subsystem(), nPart);

          // check if the output cluster is valid
          checkValid(pad_cluster, part->id());

          all_pad_clusters.emplace_back(pad_cluster);

          // start a new cluster
          cl.clear();
          cl.push_back((*d).pad());
        }
      }
      startBX = (*d).bx();
    }

    // put the last cluster in the proto collection
    if (pads.first != pads.second) {
      // number of eta partitions
      unsigned nPart = (pads.first)->nPartitions();

      GEMPadDigiCluster pad_cluster(cl, startBX, part->subsystem(), nPart);

      // check if the output cluster is valid
      checkValid(pad_cluster, part->id());

      all_pad_clusters.emplace_back(pad_cluster);
    }
    proto_clusters.emplace(part->id(), all_pad_clusters);

  }  // end of partition loop
}

void GEMPadDigiClusterProducer::selectClusters(const GEMPadDigiClusterContainer& proto_clusters,
                                               GEMPadDigiClusterCollection& out_clusters) const {
  for (const auto& ch : geometry_->chambers()) {
    const unsigned nOH = ch->id().isGE11() ? nOHGE11_ : nOHGE21_;
    const unsigned nPartitions = ch->id().isGE11() ? nPartitionsGE11_ : nPartitionsGE21_;
    const unsigned nEtaPerPartition = ch->nEtaPartitions() / (nPartitions * nOH);
    const unsigned maxClustersPart = ch->id().isGE11() ? maxClustersPartitionGE11_ : maxClustersPartitionGE21_;
    const unsigned maxClustersOH = ch->id().isGE11() ? maxClustersOHGE11_ : maxClustersOHGE21_;

    // loop over OH in this chamber
    for (unsigned int iOH = 0; iOH < nOH; iOH++) {
      unsigned int nClustersOH = 0;  // Up to 8 clusters per OH
                                     // loop over clusterizer partitions
      for (unsigned int iPart = 0; iPart < nPartitions; iPart++) {
        unsigned int nClustersPart = 0;  // Up to 4 clusters per clustizer partition
        // loop over the eta partitions for this clusterizer partition
        for (unsigned iEta = 1; iEta <= nEtaPerPartition; iEta++) {
          // get the clusters for this eta partition
          const GEMDetId& iEtaId =
              ch->etaPartition(iEta + iPart * nEtaPerPartition + iOH * nPartitions * nEtaPerPartition)->id();
          if (proto_clusters.find(iEtaId) != proto_clusters.end()) {
            for (const auto& cluster : proto_clusters.at(iEtaId)) {
              if (nClustersPart < maxClustersPart and nClustersOH < maxClustersOH) {
                checkValid(cluster, iEtaId);
                out_clusters.insertDigi(iEtaId, cluster);
                nClustersPart++;
                nClustersOH++;
              }
            }  // end of loop on clusters in eta
          }
        }  // end of eta partition loop
      }    // end of clusterizer partition loop
    }      // end of OH loop
  }        // end of chamber loop
}

template <class T>
void GEMPadDigiClusterProducer::checkValid(const T& tp, const GEMDetId& id) const {
  // check if the pad/cluster is valid
  // in principle, invalid pads/clusters can appear in the CMS raw data
  if (!tp.isValid()) {
    edm::LogWarning("GEMPadDigiClusterProducer") << "Invalid " << tp << " in " << id;
  }
}

DEFINE_FWK_MODULE(GEMPadDigiClusterProducer);
