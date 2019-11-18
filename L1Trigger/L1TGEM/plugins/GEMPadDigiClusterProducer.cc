/**
 *  \class GEMPadDigiClusterProducer
 *
 *  Produces GEM pad clusters from at most 8 adjacent GEM pads.
 *  Clusters are used downstream in the CSC local trigger to build
 *  GEM-CSC triggers and in the muon trigger to build EMTF tracks
 *
 *  Based on documentation provided by the GEM firmware architects
 *
 *  \author Sven Dildick (TAMU)
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "L1Trigger/L1TMuonEndCap/interface/TrackTools.h"

#include <string>
#include <map>
#include <vector>

class GEMPadDigiClusterProducer : public edm::stream::EDProducer<> {
public:
  // all clusters per eta partition
  typedef std::vector<GEMPadDigiCluster> GEMPadDigiClusters;
  typedef std::map<GEMDetId, GEMPadDigiClusters> GEMPadDigiClusterContainer;

  // all clusters sorted by chamber, by opthohybrid and by eta partition
  typedef std::map<GEMDetId, std::vector<std::vector<std::pair<GEMDetId, GEMPadDigiClusters> > > >
      GEMPadDigiClusterSortedContainer;

  explicit GEMPadDigiClusterProducer(const edm::ParameterSet& ps);

  ~GEMPadDigiClusterProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  void produce(edm::Event&, const edm::EventSetup&) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  /**
   *
   *************************************
   ** Light Cluster Packing Algorithm **
   *************************************

   Based on: https://github.com/cms-gem-daq-project/OptoHybridv3/raw/master/doc/OH_modules.docx
   (Andrew Peck, Thomas Lenzi, Evaldas Juska)

   In the current version of the algorithm, cluster finding is segmented
   into two separate halves of the GE1/1 chambers. Thus, each one of the
   trigger fibers can transmit clusters only from the half of the chamber
   that it corresponds to. For GE2/1, there are four separate quarts of
   the GE2/1 chamber.

   This has the downside of being unable to transmit more than 4 clusters
   when they occur within that side of the chamber, so there will be a
   slightly higher rate of cluster overflow. For GE2/1 each OH can transmit
   up to 5 clusters.

   The benefit, however, is in terms of (1) latency and (2) resource usage.

   The burden of finding clusters on  of the chamber is significantly less,
   and allows the cluster packer to operate in a simple, pipelined architecture
   which returns up to 4 (or 5) clusters per half-chamber per bunch crossing.

   This faster architecture allows the mechanism to operate with only a
   single copy of the cluster finding priority encoder and cluster truncator
   (instead of two multiplexed copies), so the total resource usage of
   these stages is approximately half.

   Further, a second step of cluster merging that is required in the full
   algorithm is avoided, which reduces latency by an additional bunch
   crossing and significantly reduces resource usage as well.

   The sorting of the clusters favors lower eta partitions and lower pad numbers
  */

  void buildClusters(const GEMPadDigiCollection& pads, GEMPadDigiClusterContainer& out_clusters) const;
  void sortClusters(const GEMPadDigiClusterContainer& in_clusters,
                    GEMPadDigiClusterSortedContainer& out_clusters) const;
  void selectClusters(const GEMPadDigiClusterSortedContainer& in,
                      GEMPadDigiClusterCollection& out,
                      GEMPadDigiClusterCollection& out_conv) const;
  void convertCluster(const GEMEtaPartition* part,
                      const GEMPadDigiCluster& in_clusters,
                      GEMPadDigiCluster& out_clusters) const;

  /// Name of input digi Collection
  edm::EDGetTokenT<GEMPadDigiCollection> pad_token_;
  edm::InputTag pads_;

  unsigned int maxClustersOHGE11_;
  unsigned int maxClustersOHGE21_;
  unsigned int nOHGE11_;
  unsigned int nOHGE21_;
  unsigned int maxClusterSize_;

  const GEMGeometry* geometry_;
};

GEMPadDigiClusterProducer::GEMPadDigiClusterProducer(const edm::ParameterSet& ps) : geometry_(nullptr) {
  pads_ = ps.getParameter<edm::InputTag>("InputCollection");
  maxClustersOHGE11_ = ps.getParameter<unsigned int>("maxClustersOHGE11");
  maxClustersOHGE21_ = ps.getParameter<unsigned int>("maxClustersOHGE21");
  nOHGE11_ = ps.getParameter<unsigned int>("nOHGE11");
  nOHGE21_ = ps.getParameter<unsigned int>("nOHGE21");
  maxClusterSize_ = ps.getParameter<unsigned int>("maxClusterSize");

  pad_token_ = consumes<GEMPadDigiCollection>(pads_);

  consumes<GEMPadDigiCollection>(pads_);
  produces<GEMPadDigiClusterCollection>();
  produces<GEMPadDigiClusterCollection>("Converted");
}

GEMPadDigiClusterProducer::~GEMPadDigiClusterProducer() {}

void GEMPadDigiClusterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputCollection", edm::InputTag("simMuonGEMPadDigis"));
  desc.add<unsigned int>("maxClustersOHGE11", 4);
  desc.add<unsigned int>("maxClustersOHGE21", 5);
  desc.add<unsigned int>("nOHGE11", 2);
  desc.add<unsigned int>("nOHGE21", 4);
  desc.add<unsigned int>("maxClusterSize", 8);

  descriptions.add("simMuonGEMPadDigiClustersDef", desc);
}

void GEMPadDigiClusterProducer::beginRun(const edm::Run& run, const edm::EventSetup& eventSetup) {
  edm::ESHandle<GEMGeometry> hGeom;
  eventSetup.get<MuonGeometryRecord>().get(hGeom);
  geometry_ = &*hGeom;
}

void GEMPadDigiClusterProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  edm::Handle<GEMPadDigiCollection> hpads;
  e.getByToken(pad_token_, hpads);

  /// step 1: create clusters

  // build the proto clusters (per partition)
  GEMPadDigiClusterContainer proto_clusters;
  buildClusters(*(hpads.product()), proto_clusters);

  // // sort clusters per chamber, per OH, per partition number and per pad number
  GEMPadDigiClusterSortedContainer sorted_clusters;
  sortClusters(proto_clusters, sorted_clusters);

  // Create empty output
  std::unique_ptr<GEMPadDigiClusterCollection> pClusters(new GEMPadDigiClusterCollection());
  std::unique_ptr<GEMPadDigiClusterCollection> pConvClusters(new GEMPadDigiClusterCollection());

  // step 2: select the clusters from sorted clusters
  // step 3: convert clusters' roll number and pad number into theta,phi
  // this emulates logic in the CTP7/ATCA processor (previously this step was done in the EMTF)
  // Future work: allow for GEM-EMTF alignment corrections!
  // Future work: replace geometry functions with a LUT
  selectClusters(sorted_clusters, *pClusters, *pConvClusters);

  // store them in the event
  e.put(std::move(pClusters));
  e.put(std::move(pConvClusters));
}

void GEMPadDigiClusterProducer::buildClusters(const GEMPadDigiCollection& det_pads,
                                              GEMPadDigiClusterContainer& proto_clusters) const {
  // clear the container
  proto_clusters.clear();

  // construct clusters
  for (const auto& part : geometry_->etaPartitions()) {
    GEMPadDigiClusters all_pad_clusters;

    // only consider partitions in GE1/1 and GE2/1 (not ME0)
    const int station = part->id().station();
    if (!(station == 1 or station == 2))
      continue;

    auto pads = det_pads.get(part->id());
    std::vector<uint16_t> cl;
    int startBX = 99;

    for (auto d = pads.first; d != pads.second; ++d) {
      if (cl.empty()) {
        cl.push_back((*d).pad());
      } else {
        if ((*d).bx() == startBX and            // same bunch crossing
            (*d).pad() == cl.back() + 1         // pad difference is 1
            and cl.size() < maxClusterSize_) {  // max 8 in cluster
          cl.push_back((*d).pad());
        } else {
          // put the current cluster in the proto collection
          GEMPadDigiCluster pad_cluster(cl, startBX);

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
      GEMPadDigiCluster pad_cluster(cl, startBX);
      all_pad_clusters.emplace_back(pad_cluster);
    }
    proto_clusters.emplace(part->id(), all_pad_clusters);

  }  // end of partition loop
}

void GEMPadDigiClusterProducer::sortClusters(const GEMPadDigiClusterContainer& proto_clusters,
                                             GEMPadDigiClusterSortedContainer& sorted_clusters) const {
  // The sorting of the clusters favors lower eta partitions and lower pad numbers
  // By default the eta partitions are sorted by Id

  sorted_clusters.clear();

  for (const auto& ch : geometry_->chambers()) {
    // check the station number
    const int station = ch->id().station();

    // only consider partitions in GE1/1 and GE2/1 (not ME0)
    if (!(station == 1 or station == 2))
      continue;

    const bool isGE11 = (station == 1);
    const unsigned nOH = isGE11 ? nOHGE11_ : nOHGE21_;
    const unsigned nPartOH = ch->nEtaPartitions() / nOH;

    std::vector<std::vector<std::pair<GEMDetId, GEMPadDigiClusters> > > temp_clustersCH;

    for (unsigned int iOH = 0; iOH < nOH; iOH++) {
      // all clusters for a set of eta partitions
      std::vector<std::pair<GEMDetId, GEMPadDigiClusters> > temp_clustersOH;

      // loop over the 4 or 2 eta partitions for this optohybrid
      for (unsigned iPart = 1; iPart <= nPartOH; iPart++) {
        // get the clusters for this eta partition
        const GEMDetId& partId = ch->etaPartition(iPart + iOH * nPartOH)->id();
        if (proto_clusters.find(partId) != proto_clusters.end()) {
          temp_clustersOH.emplace_back(partId, proto_clusters.at(partId));
        }
      }  // end of eta partition loop

      temp_clustersCH.emplace_back(temp_clustersOH);
    }  // end of OH loop

    sorted_clusters.emplace(ch->id(), temp_clustersCH);
  }  // end of chamber loop
}

void GEMPadDigiClusterProducer::selectClusters(const GEMPadDigiClusterSortedContainer& sorted_clusters,
                                               GEMPadDigiClusterCollection& out_clusters,
                                               GEMPadDigiClusterCollection& out_clusters_converted) const {
  // loop over chambers
  for (const auto& ch : geometry_->chambers()) {
    const int station = ch->id().station();

    // only consider partitions in GE1/1 and GE2/1 (not ME0)
    if (!(station == 1 or station == 2))
      continue;

    const bool isGE11 = (station == 1);
    const unsigned maxClustersOH = isGE11 ? maxClustersOHGE11_ : maxClustersOHGE21_;

    // loop over the optohybrids
    for (const auto& optohybrid : sorted_clusters.at(ch->id())) {
      // at most maxClustersOH per OH!
      unsigned nClusters = 0;

      // loop over the eta partitions for this OH
      for (const auto& etapart : optohybrid) {
        const auto& detid(etapart.first);
        const auto& clusters(etapart.second);

        // pick the clusters with lowest pad number
        for (const auto& clus : clusters) {
          if (nClusters < maxClustersOH) {
            // make a new converted cluster
            GEMPadDigiCluster clus_conv = clus;

            // convert the cluster
            convertCluster(geometry_->etaPartition(detid), clus, clus_conv);

            // put the clusters into the collections
            out_clusters.insertDigi(detid, clus);
            out_clusters_converted.insertDigi(detid, clus_conv);

            nClusters++;
          }
        }  // end of cluster loop
      }    // end of eta partition loop
    }      // end of OH loop
  }        // end of chamber loop
}

void GEMPadDigiClusterProducer::convertCluster(const GEMEtaPartition* partition,
                                               const GEMPadDigiCluster& in_cluster,
                                               GEMPadDigiCluster& out_cluster) const {
  // calculate the center of the cluster
  const auto& detid = partition->id();
  int nPads = in_cluster.pads().size();
  float centerOfCluster(nPads % 2 == 0 ? in_cluster.pads().at(nPads / 2 - 1) + 0.5
                                       : in_cluster.pads().at((nPads - 1) / 2));
  edm::LogInfo("GEMPadDigiClusterProducer") << "cluster " << in_cluster << " center " << centerOfCluster << std::endl;

  int triggerSector(CSCTriggerNumbering::triggerSectorFromLabels(detid.station(), 1, detid.chamber()));

  // do the conversion of roll and central pad number into theta and phi
  const LocalPoint& lp = partition->centreOfPad(centerOfCluster);
  const GlobalPoint& gp = partition->toGlobal(lp);

  double glob_phi = emtf::rad_to_deg(gp.phi().value());
  double glob_theta = emtf::rad_to_deg(gp.theta());

  // Use the CSC precision
  int fph = emtf::calc_phi_loc_int(glob_phi, triggerSector);
  int th = emtf::calc_theta_int(glob_theta, detid.region());

  edm::LogInfo("GEMPadDigiClusterProducer") << "glob_phi = " << glob_phi << " fph = " << fph << std::endl;
  edm::LogInfo("GEMPadDigiClusterProducer") << "glob_theta = " << glob_theta << " th = " << th << std::endl;

  if (not(0 <= fph && fph < 5000)) {
    edm::LogError("GEMPadDigiClusterProducer") << "fph = " << fph;
    return;
  }
  if (not(0 <= th && th < 128)) {
    edm::LogError("GEMPadDigiClusterProducer") << "th = " << th;
    return;
  }
  th = (th == 0) ? 1 : th;  // protect against invalid value

  // Output
  out_cluster.setTheta(th);
  out_cluster.setPhi(fph);
}

DEFINE_FWK_MODULE(GEMPadDigiClusterProducer);
