#include <string>
#include <vector>
#include <map>

#include "DQM/TrackingMonitor/interface/GetLumi.h"
#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

namespace {
  typedef dqm::reco::DQMStore DQMStore;
  typedef dqm::reco::MonitorElement MonitorElement;
  struct MEbinning {
    int nbins;
    double xmin;
    double xmax;
  };

  struct Histograms {
    dqm::reco::MonitorElement* numberOfPixelClustersVsLS;
    dqm::reco::MonitorElement* numberOfPixelClustersVsLumi;
    dqm::reco::MonitorElement* lumiVsLS;
    dqm::reco::MonitorElement* puVsLS;
    dqm::reco::MonitorElement* pixelLumiVsLS;
    dqm::reco::MonitorElement* pixelLumiVsLumi;
  };
}  // namespace

//
// class declaration
//

class LumiMonitor : public DQMGlobalEDAnalyzer<Histograms> {
public:
  LumiMonitor(const edm::ParameterSet&);
  ~LumiMonitor() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static void fillHistoPSetDescription(edm::ParameterSetDescription& pset);
  static void fillHistoLSPSetDescription(edm::ParameterSetDescription& pset);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;
  void dqmAnalyze(edm::Event const&, edm::EventSetup const&, Histograms const&) const override;

  static MEbinning getHistoPSet(const edm::ParameterSet& pset);
  static MEbinning getHistoLSPSet(const edm::ParameterSet& pset);

  std::string folderName_;

  edm::EDGetTokenT<LumiScalersCollection> lumiScalersToken_;
  MEbinning lumi_binning_;
  MEbinning pu_binning_;
  MEbinning ls_binning_;

  bool doPixelLumi_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> pixelClustersToken_;
  bool useBPixLayer1_;
  int minNumberOfPixelsPerCluster_;
  float minPixelClusterCharge_;
  MEbinning pixelCluster_binning_;
  MEbinning pixellumi_binning_;

  edm::EDGetTokenT<LumiSummary> lumiSummaryToken_;

  float lumi_factor_per_bx_;
};

// -----------------------------
//  constructors and destructor
// -----------------------------

LumiMonitor::LumiMonitor(const edm::ParameterSet& config)
    : folderName_(config.getParameter<std::string>("FolderName")),
      lumiScalersToken_(consumes<LumiScalersCollection>(config.getParameter<edm::InputTag>("scalers"))),
      lumi_binning_(getHistoPSet(
          config.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lumiPSet"))),
      pu_binning_(
          getHistoPSet(config.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("puPSet"))),
      ls_binning_(getHistoLSPSet(
          config.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet"))),
      doPixelLumi_(config.getParameter<bool>("doPixelLumi")),
      pixelClustersToken_(doPixelLumi_ ? consumes<edmNew::DetSetVector<SiPixelCluster>>(
                                             config.getParameter<edm::InputTag>("pixelClusters"))
                                       : edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>>()),
      useBPixLayer1_(doPixelLumi_ ? config.getParameter<bool>("useBPixLayer1") : false),
      minNumberOfPixelsPerCluster_(doPixelLumi_ ? config.getParameter<int>("minNumberOfPixelsPerCluster") : -1),
      minPixelClusterCharge_(doPixelLumi_ ? config.getParameter<double>("minPixelClusterCharge") : -1.),
      pixelCluster_binning_(doPixelLumi_ ? getHistoPSet(config.getParameter<edm::ParameterSet>("histoPSet")
                                                            .getParameter<edm::ParameterSet>("pixelClusterPSet"))
                                         : MEbinning{}),
      pixellumi_binning_(doPixelLumi_ ? getHistoPSet(config.getParameter<edm::ParameterSet>("histoPSet")
                                                         .getParameter<edm::ParameterSet>("pixellumiPSet"))
                                      : MEbinning{}) {
  if (useBPixLayer1_) {
    lumi_factor_per_bx_ = GetLumi::FREQ_ORBIT * GetLumi::SECONDS_PER_LS / GetLumi::XSEC_PIXEL_CLUSTER;
  } else {
    lumi_factor_per_bx_ = GetLumi::FREQ_ORBIT * GetLumi::SECONDS_PER_LS / GetLumi::rXSEC_PIXEL_CLUSTER;
  }
}

MEbinning LumiMonitor::getHistoPSet(const edm::ParameterSet& pset) {
  return MEbinning{
      pset.getParameter<int32_t>("nbins"),
      pset.getParameter<double>("xmin"),
      pset.getParameter<double>("xmax"),
  };
}

MEbinning LumiMonitor::getHistoLSPSet(const edm::ParameterSet& pset) {
  return MEbinning{pset.getParameter<int32_t>("nbins"), -0.5, pset.getParameter<int32_t>("nbins") - 0.5};
}

void LumiMonitor::bookHistograms(DQMStore::IBooker& booker,
                                 edm::Run const& run,
                                 edm::EventSetup const& setup,
                                 Histograms& histograms) const {
  booker.setCurrentFolder(folderName_);

  if (doPixelLumi_) {
    auto me = booker.book1D("numberOfPixelClustersVsLS",
                            "number of pixel clusters vs LS",
                            ls_binning_.nbins,
                            ls_binning_.xmin,
                            ls_binning_.xmax);
    me->setAxisTitle("LS", 1);
    me->setAxisTitle("number of pixel clusters", 2);
    histograms.numberOfPixelClustersVsLS = me;

    me = booker.bookProfile("numberOfPixelClustersVsLumi",
                            "number of pixel clusters vs scal lumi",
                            lumi_binning_.nbins,
                            lumi_binning_.xmin,
                            lumi_binning_.xmax,
                            pixelCluster_binning_.xmin,
                            pixelCluster_binning_.xmax);
    me->setAxisTitle("scal inst lumi E30 [Hz cm^{-2}]", 1);
    me->setAxisTitle("number of pixel clusters", 2);
    histograms.numberOfPixelClustersVsLumi = me;

    me = booker.bookProfile("pixelLumiVsLS",
                            "pixel-lumi vs LS",
                            ls_binning_.nbins,
                            ls_binning_.xmin,
                            ls_binning_.xmax,
                            pixellumi_binning_.xmin,
                            pixellumi_binning_.xmax);
    me->setAxisTitle("LS", 1);
    me->setAxisTitle("pixel-based inst lumi E30 [Hz cm^{-2}]", 2);
    histograms.pixelLumiVsLS = me;

    me = booker.bookProfile("pixelLumiVsLumi",
                            "pixel-lumi vs scal lumi",
                            lumi_binning_.nbins,
                            lumi_binning_.xmin,
                            lumi_binning_.xmax,
                            pixellumi_binning_.xmin,
                            lumi_binning_.xmax);
    me->setAxisTitle("scal inst lumi E30 [Hz cm^{-2}]", 1);
    me->setAxisTitle("pixel-based inst lumi E30 [Hz cm^{-2}]", 2);
    histograms.pixelLumiVsLumi = me;
  }

  auto me = booker.bookProfile("lumiVsLS",
                               "scal lumi vs LS",
                               ls_binning_.nbins,
                               ls_binning_.xmin,
                               ls_binning_.xmax,
                               lumi_binning_.xmin,
                               lumi_binning_.xmax);
  me->setAxisTitle("LS", 1);
  me->setAxisTitle("scal inst lumi E30 [Hz cm^{-2}]", 2);
  histograms.lumiVsLS = me;

  me = booker.bookProfile("puVsLS",
                          "scal PU vs LS",
                          ls_binning_.nbins,
                          ls_binning_.xmin,
                          ls_binning_.xmax,
                          pu_binning_.xmin,
                          pu_binning_.xmax);
  me->setAxisTitle("LS", 1);
  me->setAxisTitle("scal PU", 2);
  histograms.puVsLS = me;
}

void LumiMonitor::dqmAnalyze(edm::Event const& event,
                             edm::EventSetup const& setup,
                             Histograms const& histograms) const {
  int ls = event.id().luminosityBlock();

  float scal_lumi = -1.;
  float scal_pu = -1.;
  edm::Handle<LumiScalersCollection> lumiScalers;
  event.getByToken(lumiScalersToken_, lumiScalers);
  if (lumiScalers.isValid() and not lumiScalers->empty()) {
    auto scalit = lumiScalers->begin();
    scal_lumi = scalit->instantLumi();
    scal_pu = scalit->pileup();
  } else {
    scal_lumi = -1.;
    scal_pu = -1.;
  }
  histograms.lumiVsLS->Fill(ls, scal_lumi);
  histograms.puVsLS->Fill(ls, scal_pu);

  if (doPixelLumi_) {
    size_t pixel_clusters = 0;
    float pixel_lumi = -1.;
    edm::Handle<edmNew::DetSetVector<SiPixelCluster>> pixelClusters;
    event.getByToken(pixelClustersToken_, pixelClusters);
    if (pixelClusters.isValid()) {
      auto const& tTopo = edm::get<TrackerTopology, TrackerTopologyRcd>(setup);

      // Count the number of clusters with at least a minimum
      // number of pixels per cluster and at least a minimum charge.
      size_t tot = 0;
      for (auto pixCluDet = pixelClusters->begin(); pixCluDet != pixelClusters->end(); ++pixCluDet) {
        DetId detid = pixCluDet->detId();
        size_t subdetid = detid.subdetId();
        if (subdetid == (int)PixelSubdetector::PixelBarrel) {
          if (tTopo.layer(detid) == 1) {
            continue;
          }
        }

        for (auto pixClu = pixCluDet->begin(); pixClu != pixCluDet->end(); ++pixClu) {
          ++tot;
          if ((pixClu->size() >= minNumberOfPixelsPerCluster_) and (pixClu->charge() >= minPixelClusterCharge_)) {
            ++pixel_clusters;
          }
        }
      }
      pixel_lumi = lumi_factor_per_bx_ * pixel_clusters / GetLumi::CM2_TO_NANOBARN;  // ?!?!
    } else {
      pixel_lumi = -1.;
    }

    histograms.numberOfPixelClustersVsLS->Fill(ls, pixel_clusters);
    histograms.numberOfPixelClustersVsLumi->Fill(scal_lumi, pixel_clusters);
    histograms.pixelLumiVsLS->Fill(ls, pixel_lumi);
    histograms.pixelLumiVsLumi->Fill(scal_lumi, pixel_lumi);
  }
}

void LumiMonitor::fillHistoPSetDescription(edm::ParameterSetDescription& pset) {
  pset.add<int>("nbins");
  pset.add<double>("xmin");
  pset.add<double>("xmax");
}

void LumiMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription& pset) { pset.add<int>("nbins", 2500); }

void LumiMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelClusters", edm::InputTag("hltSiPixelClusters"));
  desc.add<edm::InputTag>("scalers", edm::InputTag("hltScalersRawToDigi"));
  desc.add<std::string>("FolderName", "HLT/LumiMonitoring");
  desc.add<bool>("doPixelLumi", false);
  desc.add<bool>("useBPixLayer1", false);
  desc.add<int>("minNumberOfPixelsPerCluster", 2);  // from DQM/PixelLumi/python/PixelLumiDQM_cfi.py
  desc.add<double>("minPixelClusterCharge", 15000.);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription pixelClusterPSet;
  LumiMonitor::fillHistoPSetDescription(pixelClusterPSet);
  histoPSet.add("pixelClusterPSet", pixelClusterPSet);

  edm::ParameterSetDescription lumiPSet;
  fillHistoPSetDescription(lumiPSet);
  histoPSet.add<edm::ParameterSetDescription>("lumiPSet", lumiPSet);

  edm::ParameterSetDescription puPSet;
  fillHistoPSetDescription(puPSet);
  histoPSet.add<edm::ParameterSetDescription>("puPSet", puPSet);

  edm::ParameterSetDescription pixellumiPSet;
  fillHistoPSetDescription(pixellumiPSet);
  histoPSet.add<edm::ParameterSetDescription>("pixellumiPSet", pixellumiPSet);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("lumiMonitor", desc);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LumiMonitor);
