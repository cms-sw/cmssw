// -*- C++ -*-
///bookLayer
// Package:    SiPixelPhase1MonitorDigis
// Class:      SiPixelPhase1MonitorDigis
//
/**\class SiPixelPhase1MonitorDigis SiPixelPhase1MonitorDigis.cc 
*/
//
// Author: Andrea Piccinelli
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
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

class SiPixelPhase1MonitorDigis : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1MonitorDigis(const edm::ParameterSet&);
  ~SiPixelPhase1MonitorDigis() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> tokenDigi_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  std::string topFolderName_;
  MonitorElement* hnDigis;
  MonitorElement* hnDigisBarrel;
  MonitorElement* hnDigisEndcap;

  MonitorElement* hDigisADC;
  MonitorElement* hDigisBarrelADC;
  MonitorElement* hDigisEndcapADC;

  MonitorElement* hnDigisBarrelLayer1;
  MonitorElement* hnDigisBarrelLayer2;
  MonitorElement* hnDigisBarrelLayer3;
  MonitorElement* hnDigisBarrelLayer4;

  MonitorElement* hDigisBarrelLayer1ADC;
  MonitorElement* hDigisBarrelLayer2ADC;
  MonitorElement* hDigisBarrelLayer3ADC;
  MonitorElement* hDigisBarrelLayer4ADC;

  MonitorElement* hnDigisEndcapDiskm1;
  MonitorElement* hnDigisEndcapDiskm2;
  MonitorElement* hnDigisEndcapDiskm3;
  MonitorElement* hnDigisEndcapDiskp1;
  MonitorElement* hnDigisEndcapDiskp2;
  MonitorElement* hnDigisEndcapDiskp3;

  MonitorElement* hDigisEndcapDiskm1ADC;
  MonitorElement* hDigisEndcapDiskm2ADC;
  MonitorElement* hDigisEndcapDiskm3ADC;
  MonitorElement* hDigisEndcapDiskp1ADC;
  MonitorElement* hDigisEndcapDiskp2ADC;
  MonitorElement* hDigisEndcapDiskp3ADC;

  const TrackerTopology* trackerTopology_;
};

//
// constructors
//

SiPixelPhase1MonitorDigis::SiPixelPhase1MonitorDigi(const edm::ParameterSet& iConfig) {
  tokenDigi_ = consumes<edm::DetSetVector<PixelDigi>>(iConfig.getParameter<edm::InputTag>("pixelDigiSrc"));
  topFolderName_ = iConfig.getParameter<std::string>("TopFolderName");  //"SiPixelHeterogeneous/PixelTrackSoA";
  trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
}

//
// -- Analyze
//
void SiPixelPhase1MonitorDigis::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get geometry
  edm::ESHandle<TrackerGeometry> tracker = iSetup.getHandle(trackerGeomToken_);
  assert(tracker.isValid());
  // TrackerTopology for module informations
  edm::ESHandle<TrackerTopology> trackerTopologyHandle = iSetup.getHandle(trackerTopoToken_);
  trackerTopology_ = trackerTopologyHandle.product();

  edm::Handle<edm::DetSetVector<PixelDigi>> input;
  iEvent.getByToken(tokenDigi_, input);
  if (!input.isValid()){
    edm::LogWarning("SiPixelPhase1MonitorDigis") << "No Digi found \n returning!" << std::endl;
  }
  else{
    edm::DetSetVector<PixelDigi>::const_iterator it;
    uint32_t nDigis = 0;
    uint32_t nDigisBarrel = 0;
    uint32_t nDigisBarrelLayer1 = 0;
    uint32_t nDigisBarrelLayer2 = 0;
    uint32_t nDigisBarrelLayer3 = 0;
    uint32_t nDigisBarrelLayer4 = 0;
    uint32_t nDigisEndcap = 0;
    uint32_t nDigisEndcapDiskm1 = 0;
    uint32_t nDigisEndcapDiskm2 = 0;
    uint32_t nDigisEndcapDiskm3 = 0;
    uint32_t nDigisEndcapDiskp1 = 0;
    uint32_t nDigisEndcapDiskp2 = 0;
    uint32_t nDigisEndcapDiskp3 = 0;
    
    for (it = input->begin(); it != input->end(); ++it) {
      const uint32_t nDigisEv = it->size();
      nDigis += nDigisEv;
      DetId id = it->detId();
      uint32_t subdetid = (id.subdetId());

      if (subdetid == PixelSubdetector::PixelBarrel) {
	nDigisBarrel += nDigisEv;
	const uint32_t nLayer = trackerTopology_->pxbLayer(id);
	if(nLayer == 1){
	  nDigisBarrelLayer1 += nDigisEv;
	  for (PixelDigi const& digi : *it){
	    double digiadc = digi.adc();
	    hDigisADC->Fill(digiadc); 
	    hDigisBarrelADC->Fill(digiadc); 
	    hDigisBarrelLayer1ADC->Fill(digiadc); 
	  }
	}
	else if(nLayer == 2){
	  nDigisBarrelLayer2 += nDigisEv;
	  for (PixelDigi const& digi : *it){
	    double digiadc = digi.adc();
	    hDigisADC->Fill(digiadc); 
	    hDigisBarrelADC->Fill(digiadc); 
	    hDigisBarrelLayer2ADC->Fill(digiadc); 
	  }
	}
	
	else if(nLayer == 3){
	  nDigisBarrelLayer3 += nDigisEv;
	  for (PixelDigi const& digi : *it){
	    double digiadc = digi.adc();
	    hDigisADC->Fill(digiadc); 
	    hDigisBarrelADC->Fill(digiadc); 
	    hDigisBarrelLayer3ADC->Fill(digiadc); 
	  }
	}
	
	else if(nLayer == 4){
	  nDigisBarrelLayer4 += nDigisEv;
	  for (PixelDigi const& digi : *it){
	    double digiadc = digi.adc();
	    hDigisADC->Fill(digiadc); 
	    hDigisBarrelADC->Fill(digiadc); 
	    hDigisBarrelLayer4ADC->Fill(digiadc); 
	  }
	}
	
      }	  

      else if (subdetid == PixelSubdetector::PixelEndcap) {
     
	nDigisEndcap += nDigisEv;
	uint32_t ECside = trackerTopology_->pxfSide(id);
	uint32_t nDisk = trackerTopology_->pxfDisk(id);

	if (ECside == 1){
	  if(nDisk == 1){
	    nDigisEndcapDiskm1 += nDigisEv;
	    for (PixelDigi const& digi : *it){
	      double digiadc = digi.adc();
	      hDigisADC->Fill(digiadc); 
	      hDigisEndcapADC->Fill(digiadc); 
	      hDigisEndcapDiskm1ADC->Fill(digiadc); 
	    }
	  }
	  else if(nDisk == 2){
	    nDigisEndcapDiskm2 += nDigisEv;
	    for (PixelDigi const& digi : *it){
	      double digiadc = digi.adc();
	      hDigisADC->Fill(digiadc); 
	      hDigisEndcapADC->Fill(digiadc); 
	      hDigisEndcapDiskm2ADC->Fill(digiadc); 
	    }
	  }
	  else if(nDisk == 3){
	    nDigisEndcapDiskm3 += nDigisEv;
	    for (PixelDigi const& digi : *it){
	      double digiadc = digi.adc();
	      hDigisADC->Fill(digiadc); 
	      hDigisEndcapADC->Fill(digiadc); 
	      hDigisEndcapDiskm3ADC->Fill(digiadc); 
	    }
	  }
	}
	else if (ECside == 2){
	  if(nDisk == 1){
	    nDigisEndcapDiskp1 += nDigisEv;
	    for (PixelDigi const& digi : *it){
	      double digiadc = digi.adc();
	      hDigisADC->Fill(digiadc); 
	      hDigisEndcapADC->Fill(digiadc); 
	      hDigisEndcapDiskp1ADC->Fill(digiadc); 
	    }
	  }

	  else if(nDisk == 2){
	    nDigisEndcapDiskp2 += nDigisEv;
	    for (PixelDigi const& digi : *it){
	      double digiadc = digi.adc();
	      hDigisADC->Fill(digiadc); 
	      hDigisEndcapADC->Fill(digiadc); 
	      hDigisEndcapDiskp2ADC->Fill(digiadc); 
	    }
	  }

	  else if(nDisk == 3){
	    nDigisEndcapDiskp3 += nDigisEv;
	    for (PixelDigi const& digi : *it){
	      double digiadc = digi.adc();
	      hDigisADC->Fill(digiadc); 
	      hDigisEndcapADC->Fill(digiadc); 
	      hDigisEndcapDiskp3ADC->Fill(digiadc); 
	    }
	  }
	}
      }

    }
    

    hnDigis->Fill(nDigis);
    hnDigisBarrel->Fill(nDigisBarrel);
    hnDigisEndcap->Fill(nDigisEndcap);
    hnDigisBarrelLayer1->Fill(nDigisBarrelLayer1);
    hnDigisBarrelLayer2->Fill(nDigisBarrelLayer2);
    hnDigisBarrelLayer3->Fill(nDigisBarrelLayer3);
    hnDigisBarrelLayer4->Fill(nDigisBarrelLayer4);
    hnDigisEndcapDiskm1->Fill(nDigisEndcapDiskm1);
    hnDigisEndcapDiskm2->Fill(nDigisEndcapDiskm2);
    hnDigisEndcapDiskm3->Fill(nDigisEndcapDiskm3);
    hnDigisEndcapDiskp1->Fill(nDigisEndcapDiskp1);
    hnDigisEndcapDiskp2->Fill(nDigisEndcapDiskp2);
    hnDigisEndcapDiskp3->Fill(nDigisEndcapDiskp3);
  }
}

//
// -- Book Histograms
//
void SiPixelPhase1MonitorDigis::bookHistograms(DQMStore::IBooker& ibooker,
                                                  edm::Run const& iRun,
                                                  edm::EventSetup const& iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder(topFolderName_);
  
  hnDigis = ibooker.book1D("nDigis", ";Number of digis per event;#entries", 1001, -0.5, 10000.5);
  hnDigisBarrel = ibooker.book1D("nDigisBarrel", ";Number of digis per event;#entries", 1001, -0.5, 10000.5);
  hnDigisEndcap = ibooker.book1D("nDigisEndcap", ";Number of digis per event;#entries", 1001, -0.5, 10000.5);

  hnDigisBarrelLayer1 = ibooker.book1D("nDigisBarrelLayer1", ";Number of digis per event;#entries", 151, -0.5, 150.5);
  hnDigisBarrelLayer2 = ibooker.book1D("nDigisBarrelLayer2", ";Number of digis per event;#entries", 151, -0.5, 150.5);
  hnDigisBarrelLayer3 = ibooker.book1D("nDigisBarrelLayer3", ";Number of digis per event;#entries", 151, -0.5, 150.5);
  hnDigisBarrelLayer4 = ibooker.book1D("nDigisBarrelLayer4", ";Number of digis per event;#entries", 151, -0.5, 150.5);

  hnDigisEndcapDiskm1 = ibooker.book1D("nDigisEndcapDiskm1", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskm2 = ibooker.book1D("nDigisEndcapDiskm2", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskm3 = ibooker.book1D("nDigisEndcapDiskm3", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskp1 = ibooker.book1D("nDigisEndcapDiskp1", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskp2 = ibooker.book1D("nDigisEndcapDiskp2", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskp3 = ibooker.book1D("nDigisEndcapDiskp3", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  
  hDigisADC = ibooker.book1D("DigisADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisBarrelADC = ibooker.book1D("DigisBarrelADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapADC = ibooker.book1D("DigisEndcapADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);

  hDigisBarrelLayer1ADC = ibooker.book1D("DigisBarrelLayer1ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisBarrelLayer2ADC = ibooker.book1D("DigisBarrelLayer2ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisBarrelLayer3ADC = ibooker.book1D("DigisBarrelLayer3ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisBarrelLayer4ADC = ibooker.book1D("DigisBarrelLayer4ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);

  hDigisEndcapDiskm1ADC = ibooker.book1D("DigisEndcapDiskm1ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskm2ADC = ibooker.book1D("DigisEndcapDiskm2ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskm3ADC = ibooker.book1D("DigisEndcapDiskm3ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskp1ADC = ibooker.book1D("DigisEndcapDiskp1ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskp2ADC = ibooker.book1D("DigisEndcapDiskp2ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskp3ADC = ibooker.book1D("DigisEndcapDiskp3ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
}

void SiPixelPhase1MonitorDigis::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelDigiSrc", edm::InputTag("siPixelDigis"));
  desc.add<std::string>("TopFolderName", "SiPixelHeterogeneous/PixelDigis");
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1MonitorDigis);
