// -*- C++ -*-
///bookLayer
// Package:    SiPixelPhase1MonitorClusterSoA
// Class:      SiPixelPhase1MonitorClusterSoA
//
/**\class SiPixelPhase1MonitorClusterSoA SiPixelPhase1MonitorClusterSoA.cc 
*/
//
// Author: Suvankar Roy Chowdhury
//
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/SiPixelCluster.h/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class SiPixelPhase1MonitorClusterSoA : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1MonitorClusterSoA(const edm::ParameterSet&);
  ~SiPixelPhase1MonitorClusterSoA() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edm::DetSetVector<PixelCluster>> tokenCluster_;
  std::string topFolderName_;
  MonitorElement* hnClusters;
};

//
// constructors
//

SiPixelPhase1MonitorClusterSoA::SiPixelPhase1MonitorClusterSoA(const edm::ParameterSet& iConfig) {
  tokenCluster_ = consumes<edm::DetSetVector<PixelCluster>>(iConfig.getParameter<edm::InputTag>("pixelClusterSrc"));
  topFolderName_ = iConfig.getParameter<std::string>("TopFolderName");  //"SiPixelHeterogeneous/PixelClusterSoA";
}

//
// -- Analyze
//
void SiPixelPhase1MonitorClusterSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::DetSetVector<PixelCluster>> input;
  iEvent.getByToken(tokenCluster_, input);
  if (!input.isValid()){
    edm::LogWarning("SiPixelPhase1MonitorClusterSoA") << "No Clusters found \n returning!" << std::endl;
  }
  else{
    edm::DetSetVector<PixelCluster>::const_iterator it;
    uint32_t nClusters = 0;
    for (it = input->begin(); it != input->end(); ++it) {
      nClusters += it->size();
    }
    edm::LogWarning("SiPixelPhase1MonitorClusterSoA") << "Found "<<nClusters<<" Clusters!" << std::endl;
    hnClusters->Fill(nClusters);
  }
}

//
// -- Book Histograms
//
void SiPixelPhase1MonitorClusterSoA::bookHistograms(DQMStore::IBooker& ibooker,
                                                  edm::Run const& iRun,
                                                  edm::EventSetup const& iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder(topFolderName_);
  hnClusters = ibooker.book1D("nClusters", ";Number of clusters per event;#entries", 1001, -0.5, 10000.5);
}

void SiPixelPhase1MonitorClusterSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelTrackSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelCluster", edm::InputTag("siPixelClusters"));
  desc.add<std::string>("TopFolderName", "SiPixelHeterogeneous/PixelClusterSoA");
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1MonitorClusterSoA);
