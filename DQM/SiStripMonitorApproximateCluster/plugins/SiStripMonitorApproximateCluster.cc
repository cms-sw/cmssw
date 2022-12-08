// -*- C++ -*-
//
// Package:    DQM/SiStripMonitorApproximateCluster
// Class:      SiStripMonitorApproximateCluster
//
/**\class SiStripMonitorApproximateCluster SiStripMonitorApproximateCluster.cc DQM/SiStripMonitorApproximateCluster/plugins/SiStripMonitorApproximateCluster.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Thu, 08 Dec 2022 20:51:10 GMT
//
//

#include <string>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripApproximateCluster.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class declaration
//

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
  MonitorElement* h_nclusters;
  MonitorElement* h_barycenter;
  MonitorElement* h_width;
  MonitorElement* h_avgCharge;
  MonitorElement* h_isSaturated;

  edm::EDGetTokenT<edmNew::DetSetVector<SiStripApproximateCluster> > clusterProducerStripToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiStripMonitorApproximateCluster::SiStripMonitorApproximateCluster(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      // Poducer name of input StripClusterCollection
      clusterProducerStripToken_(consumes<edmNew::DetSetVector<SiStripApproximateCluster> >(
          iConfig.getParameter<edm::InputTag>("ClusterProducerStrip"))) {
  // now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called for each event  ------------
void SiStripMonitorApproximateCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // get collection of DetSetVector of clusters from Event
  edm::Handle<edmNew::DetSetVector<SiStripApproximateCluster> > cluster_detsetvector;
  iEvent.getByToken(clusterProducerStripToken_, cluster_detsetvector);

  if (!cluster_detsetvector.isValid()) {
    edm::LogError("SiStripMonitorApproximateCluster")
        << "SiStripApproximate cluster collection is not valid!" << std::endl;
    return;
  }

  int nStripClusters{0};
  const edmNew::DetSetVector<SiStripApproximateCluster>* clusterCollection = cluster_detsetvector.product();

  for (const auto& detClusters : *clusterCollection) {
    for (const auto& cluster : detClusters) {
      nStripClusters++;
      h_barycenter->Fill(cluster.barycenter());
      h_width->Fill(cluster.width());
      h_avgCharge->Fill(cluster.avgCharge());
      h_isSaturated->Fill(cluster.isSaturated() ? 1 : -1);
    }
  }
  h_nclusters->Fill(nStripClusters);
}

void SiStripMonitorApproximateCluster::bookHistograms(DQMStore::IBooker& ibook,
                                                      edm::Run const& run,
                                                      edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);
  h_nclusters = ibook.book1D("numberOfClusters", "total N. of clusters;N. of clusters;#clusters", 500., 0., 500000.);
  h_barycenter = ibook.book1D("clusterBarycenter", "cluster barycenter;cluster barycenter;#clusters", 7680., 0., 7680.);
  h_width = ibook.book1D("clusterWidth", "cluster width;cluster width;#clusters", 128, -0.5, 127.5);
  h_avgCharge =
      ibook.book1D("clusterAvgCharge", "average strip charge;average strip charge;#clusters", 256, -0.5, 255.5);
  h_isSaturated = ibook.book1D("clusterSaturation", "cluster saturation;cluster saturation;is saturated", 3, -1.5, 1.5);
  h_isSaturated->getTH1F()->GetXaxis()->SetBinLabel(1, "Not saturated");
  h_isSaturated->getTH1F()->GetXaxis()->SetBinLabel(3, "Saturated");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void SiStripMonitorApproximateCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("ClusterProducerStrip", edm::InputTag("hltSiStripClusters2ApproxClusters"));
  desc.add<std::string>("folder", "SiStripApproximateClusters");
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(SiStripMonitorApproximateCluster);
