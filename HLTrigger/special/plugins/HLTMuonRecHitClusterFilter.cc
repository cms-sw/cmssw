#include <vector>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "DataFormats/MuonReco/interface/MuonRecHitCluster.h"

class HLTMuonRecHitClusterFilter : public edm::global::EDFilter<> {
public:
  explicit HLTMuonRecHitClusterFilter(const edm::ParameterSet&);
  ~HLTMuonRecHitClusterFilter() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
  const edm::EDGetTokenT<reco::MuonRecHitClusterCollection> cluster_token_;
  const int min_N_;
  const int min_Size_;
  const int min_SizeMinusMB1_;
  const std::vector<double> min_SizeRegionCutEtas_;
  const std::vector<double> max_SizeRegionCutEtas_;
  const std::vector<int> min_SizeRegionCutNstations_;
  const std::vector<int> max_SizeRegionCutNstations_;
  const std::vector<int> min_SizeRegionCutClusterSize_;
  const int max_nMB1_;
  const int max_nMB2_;
  const int max_nME11_;
  const int max_nME12_;
  const int max_nME41_;
  const int max_nME42_;
  const int min_Nstation_;
  const double min_AvgStation_;
  const double min_Time_;
  const double max_Time_;
  const double min_Eta_;
  const double max_Eta_;
  const double max_TimeSpread_;
};
//
// constructors and destructor
//
HLTMuonRecHitClusterFilter::HLTMuonRecHitClusterFilter(const edm::ParameterSet& iConfig)
    : cluster_token_(consumes<reco::MuonRecHitClusterCollection>(iConfig.getParameter<edm::InputTag>("ClusterTag"))),
      min_N_(iConfig.getParameter<int>("MinN")),
      min_Size_(iConfig.getParameter<int>("MinSize")),
      min_SizeMinusMB1_(iConfig.getParameter<int>("MinSizeMinusMB1")),
      min_SizeRegionCutEtas_(iConfig.getParameter<std::vector<double>>("MinSizeRegionCutEtas")),
      max_SizeRegionCutEtas_(iConfig.getParameter<std::vector<double>>("MaxSizeRegionCutEtas")),
      min_SizeRegionCutNstations_(iConfig.getParameter<std::vector<int>>("MinSizeRegionCutNstations")),
      max_SizeRegionCutNstations_(iConfig.getParameter<std::vector<int>>("MaxSizeRegionCutNstations")),
      min_SizeRegionCutClusterSize_(iConfig.getParameter<std::vector<int>>("MinSizeRegionCutClusterSize")),
      max_nMB1_(iConfig.getParameter<int>("Max_nMB1")),
      max_nMB2_(iConfig.getParameter<int>("Max_nMB2")),
      max_nME11_(iConfig.getParameter<int>("Max_nME11")),
      max_nME12_(iConfig.getParameter<int>("Max_nME12")),
      max_nME41_(iConfig.getParameter<int>("Max_nME41")),
      max_nME42_(iConfig.getParameter<int>("Max_nME42")),
      min_Nstation_(iConfig.getParameter<int>("MinNstation")),
      min_AvgStation_(iConfig.getParameter<double>("MinAvgStation")),
      min_Time_(iConfig.getParameter<double>("MinTime")),
      max_Time_(iConfig.getParameter<double>("MaxTime")),
      min_Eta_(iConfig.getParameter<double>("MinEta")),
      max_Eta_(iConfig.getParameter<double>("MaxEta")),
      max_TimeSpread_(iConfig.getParameter<double>("MaxTimeSpread")) {
  if (!(min_SizeRegionCutEtas_.size() == max_SizeRegionCutEtas_.size() &&
        min_SizeRegionCutEtas_.size() == min_SizeRegionCutNstations_.size() &&
        min_SizeRegionCutEtas_.size() == max_SizeRegionCutNstations_.size() &&
        min_SizeRegionCutEtas_.size() == min_SizeRegionCutClusterSize_.size())) {
    throw cms::Exception("Configuration")
        << "size of \"MinSizeRegionCutEtas\" (" << min_SizeRegionCutEtas_.size() << ") and \"MaxSizeRegionCutEtas\" ("
        << max_SizeRegionCutEtas_.size() << ") and \"MinSizeRegionCutNstations\" ("
        << min_SizeRegionCutNstations_.size() << ") and \"MaxSizeRegionCutNstations\" ("
        << max_SizeRegionCutNstations_.size() << ") and \"MinSizeRegionCutClusterSize\" ("
        << min_SizeRegionCutClusterSize_.size() << ") differ";
  }
}

void HLTMuonRecHitClusterFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ClusterTag", edm::InputTag("hltCSCrechitClusters"));
  desc.add<int>("MinN", 1);
  desc.add<int>("MinSize", 50);
  desc.add<int>("MinSizeMinusMB1", -1);
  desc.add<std::vector<double>>("MinSizeRegionCutEtas", {-1., -1., 1.9, 1.9});
  desc.add<std::vector<double>>("MaxSizeRegionCutEtas", {1.9, 1.9, -1., -1.});
  desc.add<std::vector<int>>("MinSizeRegionCutNstations", {-1, 1, -1, 1});
  desc.add<std::vector<int>>("MaxSizeRegionCutNstations", {1, -1, 1, -1});
  desc.add<std::vector<int>>("MinSizeRegionCutClusterSize", {-1, -1, -1, -1});
  desc.add<int>("Max_nMB1", -1);
  desc.add<int>("Max_nMB2", -1);
  desc.add<int>("Max_nME11", -1);
  desc.add<int>("Max_nME12", -1);
  desc.add<int>("Max_nME41", -1);
  desc.add<int>("Max_nME42", -1);
  desc.add<int>("MinNstation", 0);
  desc.add<double>("MinAvgStation", 0.0);
  desc.add<double>("MinTime", -999);
  desc.add<double>("MaxTime", 999);
  desc.add<double>("MinEta", -1.0);
  desc.add<double>("MaxEta", -1.0);
  desc.add<double>("MaxTimeSpread", -1.0);
  descriptions.addWithDefaultLabel(desc);
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool HLTMuonRecHitClusterFilter::filter(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  int nClusterPassed = 0;

  auto const& rechitClusters = iEvent.get(cluster_token_);

  for (auto const& cluster : rechitClusters) {
    auto passSizeCut = (cluster.size() >= min_Size_ && min_Size_ > 0) ||
                       ((cluster.size() - cluster.nMB1()) >= min_SizeMinusMB1_ && min_SizeMinusMB1_ > 0);
    if (not passSizeCut) {
      for (size_t i = 0; i < min_SizeRegionCutEtas_.size(); ++i) {
        if ((min_SizeRegionCutEtas_[i] < 0. || std::abs(cluster.eta()) > min_SizeRegionCutEtas_[i]) &&
            (max_SizeRegionCutEtas_[i] < 0. || std::abs(cluster.eta()) <= max_SizeRegionCutEtas_[i]) &&
            (min_SizeRegionCutNstations_[i] < 0 || cluster.nStation() > min_SizeRegionCutNstations_[i]) &&
            (max_SizeRegionCutNstations_[i] < 0 || cluster.nStation() <= max_SizeRegionCutNstations_[i]) &&
            (min_SizeRegionCutClusterSize_[i] > 0 && cluster.size() >= min_SizeRegionCutClusterSize_[i])) {
          passSizeCut = true;
          break;
        }
      }
    }
    if (passSizeCut && (max_nMB1_ < 0 || cluster.nMB1() <= max_nMB1_) &&
        (max_nMB2_ < 0 || cluster.nMB2() <= max_nMB2_) && (max_nME11_ < 0 || cluster.nME11() <= max_nME11_) &&
        (max_nME12_ < 0 || cluster.nME12() <= max_nME12_) && (max_nME41_ < 0 || cluster.nME41() <= max_nME41_) &&
        (max_nME42_ < 0 || cluster.nME42() <= max_nME42_) && cluster.nStation() >= min_Nstation_ &&
        cluster.avgStation() >= min_AvgStation_ && (min_Eta_ < 0.0 || std::abs(cluster.eta()) > min_Eta_) &&
        (max_Eta_ < 0.0 || std::abs(cluster.eta()) <= max_Eta_) && cluster.time() > min_Time_ &&
        cluster.time() <= max_Time_ && (max_TimeSpread_ < 0.0 || cluster.timeSpread() <= max_TimeSpread_)) {
      nClusterPassed++;
    }
  }

  return (nClusterPassed >= min_N_);
}

DEFINE_FWK_MODULE(HLTMuonRecHitClusterFilter);
