// -*- C++ -*-
///bookLayer
// Package:    SiPixelPhase1MonitorDigiSoA
// Class:      SiPixelPhase1MonitorDigiSoA
//
/**\class SiPixelPhase1MonitorDigiSoA SiPixelPhase1MonitorDigiSoA.cc 
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
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigisSoA.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

class SiPixelPhase1MonitorDigiSoA : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1MonitorDigiSoA(const edm::ParameterSet&);
  ~SiPixelPhase1MonitorDigiSoA() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<SiPixelDigisSoA> tokenSoADigi_;
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tokenDigi_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  std::string topFolderName_;
  MonitorElement* hnDigisSoA;
  MonitorElement* hnDigis;
  const TrackerTopology* trackerTopology_;
};

//
// constructors
//

SiPixelPhase1MonitorDigiSoA::SiPixelPhase1MonitorDigiSoA(const edm::ParameterSet& iConfig) {
  tokenSoADigi_ = consumes<SiPixelDigisSoA>(iConfig.getParameter<edm::InputTag>("pixelDigiSrc"));
  tokenDigi_ = consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("pixelDigiSrc2"));
  topFolderName_ = iConfig.getParameter<std::string>("TopFolderName");  //"SiPixelHeterogeneous/PixelTrackSoA";
  trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
}

//
// -- Analyze
//
void SiPixelPhase1MonitorDigiSoA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get geometry
  edm::ESHandle<TrackerGeometry> tracker = iSetup.getHandle(trackerGeomToken_);
  assert(tracker.isValid());
  // TrackerTopology for module informations
  edm::ESHandle<TrackerTopology> trackerTopologyHandle = iSetup.getHandle(trackerTopoToken_);
  trackerTopology_ = trackerTopologyHandle.product();


  const auto& dsoaHandle = iEvent.getHandle(tokenSoADigi_);
  if (!dsoaHandle.isValid()) {
    edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "No Digi SoA found!" << std::endl;
  }
  else{
    auto const& dsoa = iEvent.get(tokenSoADigi_);//*((dsoaHandle.product())->get());
    const uint32_t nDigis = dsoa.size();
    edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "Found "<<nDigis<<" Digis SoA!" << std::endl;
    hnDigisSoA->Fill(nDigis);
    for(uint32_t i=0;i<nDigis;i++){
      DetId id = DetId(dsoa.rawIdArr(i));
      uint32_t subdetid = (id.subdetId());
      if (subdetid == PixelSubdetector::PixelBarrel) {
	edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "SoA PX Barrel:  DetId " <<id.rawId()<<" Layer "<<trackerTopology_->pxbLayer(id)<<std::endl;
      }
      if (subdetid == PixelSubdetector::PixelEndcap) {
	edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "SoA PX Endcaps:  DetId " <<id.rawId()<<" Side "<<trackerTopology_->pxfSide(id)<<" Disk "<<trackerTopology_->pxfDisk(id)<<std::endl;
      }
    }
  }
  edm::Handle<edm::DetSetVector<PixelDigi>> input;
  iEvent.getByToken(tokenDigi_, input);
  if (!input.isValid()){
    edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "No Digi found \n returning!" << std::endl;
  }
  else{
    edm::DetSetVector<PixelDigi>::const_iterator it;
    uint32_t nDigis = 0;
    for (it = input->begin(); it != input->end(); ++it) {
      nDigis += it->size();
      DetId id = it->detId();
      uint32_t subdetid = (id.subdetId());
      if (subdetid == PixelSubdetector::PixelBarrel) {
	edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "PX Barrel:  DetId " <<id.rawId()<<" Layer "<<trackerTopology_->pxbLayer(id)<<std::endl;
      }
      if (subdetid == PixelSubdetector::PixelEndcap) {
	edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "PX Endcaps:  DetId " <<id.rawId()<<" Side "<<trackerTopology_->pxfSide(id)<<" Disk "<<trackerTopology_->pxfDisk(id)<<std::endl;
      }
    }
    edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "Found "<<nDigis<<" Digis!" << std::endl;
    hnDigis->Fill(nDigis);
  }


}

//
// -- Book Histograms
//
void SiPixelPhase1MonitorDigiSoA::bookHistograms(DQMStore::IBooker& ibooker,
                                                  edm::Run const& iRun,
                                                  edm::EventSetup const& iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder(topFolderName_);
  hnDigis = ibooker.book1D("nDigis", ";Number of digis per event;#entries", 1001, -0.5, 10000.5);
  hnDigisSoA = ibooker.book1D("nDigisSoA", ";Number of digis per event;#entries", 1001, -0.5, 10000.5);
}

void SiPixelPhase1MonitorDigiSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelTrackSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelDigiSrc", edm::InputTag("siPixelDigisSoA"));
  desc.add<edm::InputTag>("pixelDigiSrc2", edm::InputTag("siPixelDigis"));
  desc.add<std::string>("TopFolderName", "SiPixelHeterogeneous/PixelDigiSoA");
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1MonitorDigiSoA);
