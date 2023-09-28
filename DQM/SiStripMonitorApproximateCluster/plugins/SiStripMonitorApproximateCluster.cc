// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorApproximateCluster
// Class:      SiStripMonitorApproximateCluster
//
/**\class SiStripMonitorApproximateCluster SiStripMonitorApproximateCluster.cc DQM/SiStripMonitorApproximateCluster/plugins/SiStripMonitorApproximateCluster.cc

 Description: Monitor SiStripApproximateClusters and on-demand compare properties with original SiStripClusters

*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 08 Dec 2022 20:51:10 GMT
//
//

#include <string>
// for string manipulations
#include <fmt/printf.h>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetIdVector.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateClusterCollection.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//
// class declaration
//

namespace siStripRawPrime {
  struct monitorApproxCluster {
  public:
    monitorApproxCluster()
        : h_barycenter_{nullptr}, h_width_{nullptr}, h_avgCharge_{nullptr}, h_isSaturated_{nullptr}, isBooked_{false} {}

    void fill(const SiStripApproximateCluster& cluster) {
      h_barycenter_->Fill(cluster.barycenter());
      h_width_->Fill(cluster.width());
      h_charge_->Fill(cluster.width() * cluster.avgCharge());  // see SiStripCluster constructor
      h_avgCharge_->Fill(cluster.avgCharge());
      h_isSaturated_->Fill(cluster.isSaturated() ? 1 : -1);
      h_passFilter_->Fill(cluster.filter() ? 1 : -1);
      h_passPeakFilter_->Fill(cluster.peakFilter() ? 1 : -1);
    }

    void book(dqm::implementation::DQMStore::IBooker& ibook, const std::string& folder) {
      ibook.setCurrentFolder(folder);
      h_barycenter_ =
          ibook.book1D("clusterBarycenter", "cluster barycenter;cluster barycenter;#clusters", 7680., 0., 7680.);
      h_width_ = ibook.book1D("clusterWidth", "cluster width;cluster width;#clusters", 128, -0.5, 127.5);
      h_avgCharge_ = ibook.book1D(
          "clusterAvgCharge", "average strip charge;average strip charge [ADC counts];#clusters", 256, -0.5, 255.5);

      h_charge_ = ibook.book1D(
          "clusterCharge", "total cluster charge;total cluster charge [ADC counts];#clusters", 300, -0.5, 2999.5);

      h_isSaturated_ = ibook.book1D("clusterSaturation", "cluster saturation;is saturated?;#clusters", 3, -1.5, 1.5);
      h_isSaturated_->setBinLabel(1, "Not saturated");
      h_isSaturated_->setBinLabel(3, "Saturated");

      h_passFilter_ = ibook.book1D("filter", "filter;passes filter?;#clusters", 3, -1.5, 1.5);
      h_passFilter_->setBinLabel(1, "No");
      h_passFilter_->setBinLabel(3, "Yes");

      h_passPeakFilter_ = ibook.book1D("peakFilter", "peak filter;passes peak filter?;#clusters", 3, -1.5, 1.5);
      h_passPeakFilter_->setBinLabel(1, "No");
      h_passPeakFilter_->setBinLabel(3, "Yes");

      isBooked_ = true;
    }

    bool isBooked() { return isBooked_; }

  private:
    // approximate clusters
    dqm::reco::MonitorElement* h_barycenter_;
    dqm::reco::MonitorElement* h_width_;
    dqm::reco::MonitorElement* h_avgCharge_;
    dqm::reco::MonitorElement* h_charge_;
    dqm::reco::MonitorElement* h_isSaturated_;
    dqm::reco::MonitorElement* h_passFilter_;
    dqm::reco::MonitorElement* h_passPeakFilter_;

    bool isBooked_;
  };
}  // namespace siStripRawPrime

class SiStripMonitorApproximateCluster : public DQMEDAnalyzer {
public:
  explicit SiStripMonitorApproximateCluster(const edm::ParameterSet&);
  ~SiStripMonitorApproximateCluster() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------
  std::string folder_;
  bool compareClusters_;
  MonitorElement* h_nclusters_;

  siStripRawPrime::monitorApproxCluster allClusters{};
  siStripRawPrime::monitorApproxCluster matchedClusters{};
  siStripRawPrime::monitorApproxCluster unMatchedClusters{};

  // FED errors
  dqm::reco::MonitorElement* h_numberFEDErrors_;

  // for comparisons
  MonitorElement* h_isMatched_{nullptr};
  MonitorElement* h_deltaBarycenter_{nullptr};
  MonitorElement* h_deltaSize_{nullptr};
  MonitorElement* h_deltaCharge_{nullptr};
  MonitorElement* h_deltaFirstStrip_{nullptr};
  MonitorElement* h_deltaEndStrip_{nullptr};

  MonitorElement* h2_CompareBarycenter_{nullptr};
  MonitorElement* h2_CompareSize_{nullptr};
  MonitorElement* h2_CompareCharge_{nullptr};
  MonitorElement* h2_CompareFirstStrip_{nullptr};
  MonitorElement* h2_CompareEndStrip_{nullptr};

  // Event Data
  const edm::EDGetTokenT<SiStripApproximateClusterCollection> approxClustersToken_;
  const edm::EDGetTokenT<DetIdVector> stripFEDErrorsToken_;

  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> stripClustersToken_;
  const edmNew::DetSetVector<SiStripCluster>* stripClusterCollection_;

  // Event Setup Data
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkGeomToken_;
};

//
// constructors and destructor
//
SiStripMonitorApproximateCluster::SiStripMonitorApproximateCluster(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      compareClusters_(iConfig.getParameter<bool>("compareClusters")),
      // Poducer name of input StripClusterCollection
      approxClustersToken_(
          consumes<SiStripApproximateClusterCollection>(iConfig.getParameter<edm::InputTag>("ApproxClustersProducer"))),
      stripFEDErrorsToken_(consumes<DetIdVector>(iConfig.getParameter<edm::InputTag>("SiStripFEDErrorVector"))) {
  tkGeomToken_ = esConsumes();
  if (compareClusters_) {
    stripClustersToken_ =
        consumes<edmNew::DetSetVector<SiStripCluster>>(iConfig.getParameter<edm::InputTag>("ClustersProducer"));
  }
  stripClusterCollection_ = nullptr;
}

//
// member functions
//

// ------------ method called for each event  ------------
void SiStripMonitorApproximateCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const auto& tkGeom = &iSetup.getData(tkGeomToken_);
  const auto tkDets = tkGeom->dets();

  // get SiStripApproximateClusterCollection from Event
  edm::Handle<SiStripApproximateClusterCollection> approx_cluster_detsetvector;
  iEvent.getByToken(approxClustersToken_, approx_cluster_detsetvector);
  if (!approx_cluster_detsetvector.isValid()) {
    edm::LogError("SiStripMonitorApproximateCluster")
        << "SiStripApproximate cluster collection is not valid!" << std::endl;

    // if approximate clusters collection not available, then early return
    return;
  }

  // get the DetIdVector of the SiStrip FED Errors
  edm::Handle<DetIdVector> sistripFEDErrorsVector;
  iEvent.getByToken(stripFEDErrorsToken_, sistripFEDErrorsVector);
  if (!sistripFEDErrorsVector.isValid()) {
    edm::LogError("SiStripMonitorApproximateCluster")
        << " DetIdVector collection of SiStrip FED errors is not valid!" << std::endl;

    // if approximate clusters collection not available, then early return
    return;
  }

  // if requested to perform the comparison
  if (compareClusters_) {
    // get collection of DetSetVector of clusters from Event
    edm::Handle<edmNew::DetSetVector<SiStripCluster>> cluster_detsetvector;
    iEvent.getByToken(stripClustersToken_, cluster_detsetvector);
    if (!cluster_detsetvector.isValid()) {
      edm::LogError("SiStripMonitorApproximateCluster")
          << "Requested to perform comparison, but regular SiStrip cluster collection is not valid!" << std::endl;
      return;
    } else {
      stripClusterCollection_ = cluster_detsetvector.product();
    }
  }

  int nApproxClusters{0};
  const SiStripApproximateClusterCollection* clusterCollection = approx_cluster_detsetvector.product();

  for (const auto& detClusters : *clusterCollection) {
    edmNew::DetSet<SiStripCluster> strip_clusters_detset;
    const auto& detid = detClusters.id();  // get the detid of the current detset

    // starts here comaparison with regular clusters
    if (compareClusters_) {
      edmNew::DetSetVector<SiStripCluster>::const_iterator isearch =
          stripClusterCollection_->find(detid);  // search clusters of same detid

      // protect against a missing match
      if (isearch != stripClusterCollection_->end())
        strip_clusters_detset = (*isearch);
    }

    for (const auto& cluster : detClusters) {
      nApproxClusters++;

      // fill the full cluster collection
      allClusters.fill(cluster);

      if (compareClusters_ && !strip_clusters_detset.empty()) {
        // build the converted cluster for the matching
        uint16_t nStrips{0};
        auto det = std::find_if(tkDets.begin(), tkDets.end(), [detid](auto& elem) -> bool {
          return (elem->geographicalId().rawId() == detid);
        });
        const StripTopology& p = dynamic_cast<const StripGeomDetUnit*>(*det)->specificTopology();
        nStrips = p.nstrips() - 1;

        const auto convertedCluster = SiStripCluster(cluster, nStrips);

        float distance{9999.};
        const SiStripCluster* closestCluster{nullptr};
        for (const auto& stripCluster : strip_clusters_detset) {
          // by construction the approximated cluster width has same
          // size as the original cluster

          if (cluster.width() != stripCluster.size()) {
            continue;
          }

          float deltaBarycenter = convertedCluster.barycenter() - stripCluster.barycenter();
          if (std::abs(deltaBarycenter) < distance) {
            closestCluster = &stripCluster;
            distance = std::abs(deltaBarycenter);
          }
        }

        // Matching criteria:
        // - if exists a closest cluster in the DetId
        // - the size coincides with the original one
        if (closestCluster) {
          // comparisong plots
          h_deltaBarycenter_->Fill(closestCluster->barycenter() - convertedCluster.barycenter());
          h_deltaSize_->Fill(closestCluster->size() - convertedCluster.size());
          h_deltaCharge_->Fill(closestCluster->charge() - convertedCluster.charge());
          h_deltaFirstStrip_->Fill(closestCluster->firstStrip() - convertedCluster.firstStrip());
          h_deltaEndStrip_->Fill(closestCluster->endStrip() - convertedCluster.endStrip());

          h2_CompareBarycenter_->Fill(closestCluster->barycenter(), convertedCluster.barycenter());
          h2_CompareSize_->Fill(closestCluster->size(), convertedCluster.size());
          h2_CompareCharge_->Fill(closestCluster->charge(), convertedCluster.charge());
          h2_CompareFirstStrip_->Fill(closestCluster->firstStrip(), convertedCluster.firstStrip());
          h2_CompareEndStrip_->Fill(closestCluster->endStrip(), convertedCluster.endStrip());

          // monitoring plots
          matchedClusters.fill(cluster);
          h_isMatched_->Fill(1);
        } else {
          // monitoring plots
          unMatchedClusters.fill(cluster);
          h_isMatched_->Fill(-1);
        }
      }  // if we're doing the comparison cluster by cluster

    }  // loop on clusters in a detset
  }    // loop on the detset vector

  h_nclusters_->Fill(nApproxClusters);
  h_numberFEDErrors_->Fill((*sistripFEDErrorsVector).size());
}

void SiStripMonitorApproximateCluster::bookHistograms(DQMStore::IBooker& ibook,
                                                      edm::Run const& run,
                                                      edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);
  h_nclusters_ = ibook.book1D("numberOfClusters", "total N. of clusters;N. of clusters;# events", 500., 0., 500000.);
  h_numberFEDErrors_ = ibook.book1D(
      "numberOfFEDErrors", "number of SiStrip modules in FED Error;N. of Modules in Error; #events", 100, 0, 1000);

  allClusters.book(ibook, folder_);

  //  for comparisons
  if (compareClusters_) {
    // book monitoring for matche and unmatched clusters separately
    matchedClusters.book(ibook, fmt::format("{}/MatchedClusters", folder_));
    unMatchedClusters.book(ibook, fmt::format("{}/UnmatchedClusters", folder_));

    // 1D histograms
    ibook.setCurrentFolder(fmt::format("{}/ClusterComparisons", folder_));
    h_deltaBarycenter_ =
        ibook.book1D("deltaBarycenter", "#Delta barycenter;#Delta barycenter;cluster pairs", 101, -50.5, 50.5);
    h_deltaSize_ = ibook.book1D("deltaSize", "#Delta size;#Delta size;cluster pairs", 201, -100.5, 100.5);
    h_deltaCharge_ = ibook.book1D("deltaCharge", "#Delta charge;#Delta charge;cluster pairs", 401, -200.5, 200.5);

    h_deltaFirstStrip_ =
        ibook.book1D("deltaFirstStrip", "#Delta FirstStrip; #Delta firstStrip;cluster pairs", 101, -50.5, 50.5);
    h_deltaEndStrip_ =
        ibook.book1D("deltaEndStrip", "#Delta EndStrip; #Delta endStrip; cluster pairs", 101, -50.5, 50.5);

    // geometric constants
    constexpr int maxNStrips = 6 * sistrip::STRIPS_PER_APV;
    constexpr float minStrip = -0.5f;
    constexpr float maxStrip = maxNStrips - 0.5f;

    // cluster constants
    constexpr float maxCluSize = 100;
    constexpr float maxCluCharge = 3000;

    // 2D histograms (use TH2I for counts to limit memory allocation)
    std::string toRep = "SiStrip Cluster Barycenter";
    h2_CompareBarycenter_ = ibook.book2I("compareBarycenter",
                                         fmt::sprintf("; %s;Approx %s", toRep, toRep),
                                         maxNStrips,
                                         minStrip,
                                         maxStrip,
                                         maxNStrips,
                                         minStrip,
                                         maxStrip);

    toRep = "SiStrip Cluster Size";
    h2_CompareSize_ = ibook.book2I("compareSize",
                                   fmt::sprintf("; %s;Approx %s", toRep, toRep),
                                   maxCluSize,
                                   -0.5f,
                                   maxCluSize - 0.5f,
                                   maxCluSize,
                                   -0.5f,
                                   maxCluSize - 0.5f);

    toRep = "SiStrip Cluster Charge";
    h2_CompareCharge_ = ibook.book2I("compareCharge",
                                     fmt::sprintf("; %s;Approx %s", toRep, toRep),
                                     (maxCluCharge / 5),
                                     -0.5f,
                                     maxCluCharge - 0.5f,
                                     (maxCluCharge / 5),
                                     -0.5f,
                                     maxCluCharge - 0.5f);

    toRep = "SiStrip Cluster First Strip number";
    h2_CompareFirstStrip_ = ibook.book2I("compareFirstStrip",
                                         fmt::sprintf("; %s;Approx %s", toRep, toRep),
                                         maxNStrips,
                                         minStrip,
                                         maxStrip,
                                         maxNStrips,
                                         minStrip,
                                         maxStrip);

    toRep = "SiStrip Cluster Last Strip number";
    h2_CompareEndStrip_ = ibook.book2I("compareLastStrip",
                                       fmt::sprintf("; %s;Approx %s", toRep, toRep),
                                       maxNStrips,
                                       minStrip,
                                       maxStrip,
                                       maxNStrips,
                                       minStrip,
                                       maxStrip);

    h_isMatched_ = ibook.book1D("isClusterMatched", "cluster matching;is matched?;#clusters", 3, -1.5, 1.5);
    h_isMatched_->setBinLabel(1, "Not matched");
    h_isMatched_->setBinLabel(3, "Matched");
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiStripMonitorApproximateCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Monitor SiStripApproximateCluster collection and compare with regular SiStrip clusters");
  desc.add<bool>("compareClusters", false)->setComment("if true, will compare with regualr Strip clusters");
  desc.add<edm::InputTag>("ApproxClustersProducer", edm::InputTag("hltSiStripClusters2ApproxClusters"))
      ->setComment("approxmate clusters collection");
  desc.add<edm::InputTag>("SiStripFEDErrorVector", edm::InputTag("hltSiStripRawToDigi"))
      ->setComment("SiStrip FED Errors collection");
  desc.add<edm::InputTag>("ClustersProducer", edm::InputTag("hltSiStripClusterizerForRawPrime"))
      ->setComment("regular clusters collection");
  desc.add<std::string>("folder", "SiStripApproximateClusters")->setComment("Top Level Folder");
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorApproximateCluster);
