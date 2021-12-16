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
  MonitorElement* hnDigisSoABarrel;
  MonitorElement* hnDigisBarrel;
  MonitorElement* hnDigisSoAEndcap;
  MonitorElement* hnDigisEndcap;

  MonitorElement* hDigisSoAADC;
  MonitorElement* hDigisADC;
  MonitorElement* hDigisSoABarrelADC;
  MonitorElement* hDigisBarrelADC;
  MonitorElement* hDigisSoAEndcapADC;
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
  //MonitorElement* hnDigisEndcapDiskm4;
  //MonitorElement* hnDigisEndcapDiskm5;
  //MonitorElement* hnDigisEndcapDiskm6;
  MonitorElement* hnDigisEndcapDiskp1;
  MonitorElement* hnDigisEndcapDiskp2;
  MonitorElement* hnDigisEndcapDiskp3;
  //MonitorElement* hnDigisEndcapDiskp4;
  //MonitorElement* hnDigisEndcapDiskp5;
  //MonitorElement* hnDigisEndcapDiskp6;

  MonitorElement* hDigisEndcapDiskm1ADC;
  MonitorElement* hDigisEndcapDiskm2ADC;
  MonitorElement* hDigisEndcapDiskm3ADC;
  //MonitorElement* hDigisEndcapDiskm4ADC;
  //MonitorElement* hDigisEndcapDiskm5ADC;
  //MonitorElement* hDigisEndcapDiskm6ADC;
  MonitorElement* hDigisEndcapDiskp1ADC;
  MonitorElement* hDigisEndcapDiskp2ADC;
  MonitorElement* hDigisEndcapDiskp3ADC;
  //MonitorElement* hDigisEndcapDiskp4ADC;
  //MonitorElement* hDigisEndcapDiskp5ADC;
  //MonitorElement* hDigisEndcapDiskp6ADC;

  MonitorElement* hnDigisSoABarrelLayer1;
  MonitorElement* hnDigisSoABarrelLayer2;
  MonitorElement* hnDigisSoABarrelLayer3;
  MonitorElement* hnDigisSoABarrelLayer4;

  MonitorElement* hDigisSoABarrelLayer1ADC;
  MonitorElement* hDigisSoABarrelLayer2ADC;
  MonitorElement* hDigisSoABarrelLayer3ADC;
  MonitorElement* hDigisSoABarrelLayer4ADC;

  MonitorElement* hnDigisSoAEndcapDiskm1;
  MonitorElement* hnDigisSoAEndcapDiskm2;
  MonitorElement* hnDigisSoAEndcapDiskm3;
  //MonitorElement* hnDigisSoAEndcapDiskm4;
  //MonitorElement* hnDigisSoAEndcapDiskm5;
  //MonitorElement* hnDigisSoAEndcapDiskm6;

  MonitorElement* hnDigisSoAEndcapDiskp1;
  MonitorElement* hnDigisSoAEndcapDiskp2;
  MonitorElement* hnDigisSoAEndcapDiskp3;
  //MonitorElement* hnDigisSoAEndcapDiskp4;
  //MonitorElement* hnDigisSoAEndcapDiskp5;
  //MonitorElement* hnDigisSoAEndcapDiskp6;

  MonitorElement* hDigisSoAEndcapDiskm1ADC;
  MonitorElement* hDigisSoAEndcapDiskm2ADC;
  MonitorElement* hDigisSoAEndcapDiskm3ADC;
  //MonitorElement* hDigisSoAEndcapDiskm4ADC;
  //MonitorElement* hDigisSoAEndcapDiskm5ADC;
  //MonitorElement* hDigisSoAEndcapDiskm6ADC;
  MonitorElement* hDigisSoAEndcapDiskp1ADC;
  MonitorElement* hDigisSoAEndcapDiskp2ADC;
  MonitorElement* hDigisSoAEndcapDiskp3ADC;
  //MonitorElement* hDigisSoAEndcapDiskp4ADC;
  //MonitorElement* hDigisSoAEndcapDiskp5ADC;
  //MonitorElement* hDigisSoAEndcapDiskp6ADC;

  //vector<MonitorElement*> hnDigisBarrelLayers;
  //hnDigisBarrelLayers->push_back(hnDigisBarrelLayer1);
  //hnDigisBarrelLayers->push_back(hnDigisBarrelLayer2);
  //hnDigisBarrelLayers->push_back(hnDigisBarrelLayer3);
  //hnDigisBarrelLayers->push_back(hnDigisBarrelLayer4);

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
    const uint32_t nDigisSoA = dsoa.size();
    edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "Found "<<nDigisSoA<<" DigisSoA SoA!" << std::endl;
    hnDigisSoA->Fill(nDigisSoA);
    uint32_t nDigisSoABarrel = 0;
    uint32_t nDigisSoAEndcap = 0;
    uint32_t nDigisSoABarrelLayer1 = 0;
    uint32_t nDigisSoABarrelLayer2 = 0;
    uint32_t nDigisSoABarrelLayer3 = 0;
    uint32_t nDigisSoABarrelLayer4 = 0;

    uint32_t nDigisSoAEndcapDiskm1 = 0;
    uint32_t nDigisSoAEndcapDiskm2 = 0;
    uint32_t nDigisSoAEndcapDiskm3 = 0;
    //uint32_t nDigisSoAEndcapDiskm4 = 0;
    //uint32_t nDigisSoAEndcapDiskm5 = 0;
    //uint32_t nDigisSoAEndcapDiskm6 = 0;
    uint32_t nDigisSoAEndcapDiskp1 = 0;
    uint32_t nDigisSoAEndcapDiskp2 = 0;
    uint32_t nDigisSoAEndcapDiskp3 = 0;
    //uint32_t nDigisSoAEndcapDiskp4 = 0;
    //uint32_t nDigisSoAEndcapDiskp5 = 0;
    //uint32_t nDigisSoAEndcapDiskp6 = 0;

    for(uint32_t i=0;i<nDigisSoA;i++){
      DetId id = DetId(dsoa.rawIdArr(i));
      uint32_t subdetid = (id.subdetId());
      //edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "DigiSoA adc: " << 
      if (subdetid == PixelSubdetector::PixelBarrel) {
	//edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "SoA PX Barrel:  DetId " <<id.rawId()<<" Layer "<<trackerTopology_->pxbLayer(id)<<std::endl;
	nDigisSoABarrel += 1;

	uint32_t nLayer = trackerTopology_->pxbLayer(id);
	if(nLayer == 1){
	  nDigisSoABarrelLayer1 += 1;
	  hDigisSoAADC->Fill(dsoa.adc(i));
	  hDigisSoABarrelADC->Fill(dsoa.adc(i));
	  hDigisSoABarrelLayer1ADC->Fill(dsoa.adc(i));
	}
	else if(nLayer == 2){
	  nDigisSoABarrelLayer2 += 1;
	  hDigisSoAADC->Fill(dsoa.adc(i));
	  hDigisSoABarrelADC->Fill(dsoa.adc(i));
	  hDigisSoABarrelLayer2ADC->Fill(dsoa.adc(i));
	}
	else if(nLayer == 3){
	  nDigisSoABarrelLayer3 += 1;
	  hDigisSoAADC->Fill(dsoa.adc(i));
	  hDigisSoABarrelADC->Fill(dsoa.adc(i));
	  hDigisSoABarrelLayer3ADC->Fill(dsoa.adc(i));
	}
	else if(nLayer == 4){
	  nDigisSoABarrelLayer4 += 1;
	  hDigisSoAADC->Fill(dsoa.adc(i));
	  hDigisSoABarrelADC->Fill(dsoa.adc(i));
	  hDigisSoABarrelLayer4ADC->Fill(dsoa.adc(i));
	}
      }
      else if (subdetid == PixelSubdetector::PixelEndcap) {
	//edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "SoA PX Endcaps:  DetId " <<id.rawId()<<" Side "<<trackerTopology_->pxfSide(id)<<" Disk "<<trackerTopology_->pxfDisk(id)<<std::endl;
	uint32_t ECside = trackerTopology_->pxfSide(id);
	uint32_t nDisk = trackerTopology_->pxfDisk(id);
	nDigisSoAEndcap += 1;

	if(ECside == 1){
	  if(nDisk == 1){
	    nDigisSoAEndcapDiskm1 += 1;
	    hDigisSoAADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapDiskm1ADC->Fill(dsoa.adc(i));
	  }
	  else if(nDisk == 2){
	    nDigisSoAEndcapDiskm2 += 1;
	    hDigisSoAADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapDiskm2ADC->Fill(dsoa.adc(i));
	  }
	  else if(nDisk == 3){
	    nDigisSoAEndcapDiskm3 += 1;
	    hDigisSoAADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapDiskm3ADC->Fill(dsoa.adc(i));
	  }
	  //else if(nDisk == 4) nDigisSoAEndcapDiskm4 += 1;
	  //else if(nDisk == 5) nDigisSoAEndcapDiskm5 += 1;
	  //else if(nDisk == 6) nDigisSoAEndcapDiskm6 += 1;
	}

	else if(ECside == 2){
	  if(nDisk == 1){
	    nDigisSoAEndcapDiskp1 += 1;
	    hDigisSoAADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapDiskp1ADC->Fill(dsoa.adc(i));
	  }
	  else if(nDisk == 2){
	    nDigisSoAEndcapDiskp2 += 1;
	    hDigisSoAADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapDiskp2ADC->Fill(dsoa.adc(i));
	  }
	  else if(nDisk == 3){
	    nDigisSoAEndcapDiskp3 += 1;
	    hDigisSoAADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapADC->Fill(dsoa.adc(i));
	    hDigisSoAEndcapDiskp3ADC->Fill(dsoa.adc(i));
	  }
	  //else if(nDisk == 4) nDigisSoAEndcapDiskp4 += 1;
	  //else if(nDisk == 5) nDigisSoAEndcapDiskp5 += 1;
	  //else if(nDisk == 6) nDigisSoAEndcapDiskp6 += 1;
	}
      }
    }

    edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "nDigisSoA Endcap" << nDigisSoAEndcap << std::endl;

    hnDigisSoABarrel->Fill(nDigisSoABarrel);
    hnDigisSoAEndcap->Fill(nDigisSoAEndcap);

    hnDigisSoABarrelLayer1->Fill(nDigisSoABarrelLayer1);

    hnDigisSoABarrelLayer2->Fill(nDigisSoABarrelLayer2);
    hnDigisSoABarrelLayer3->Fill(nDigisSoABarrelLayer3);
    hnDigisSoABarrelLayer4->Fill(nDigisSoABarrelLayer4);

    hnDigisSoAEndcapDiskm1->Fill(nDigisSoAEndcapDiskm1);
    hnDigisSoAEndcapDiskm2->Fill(nDigisSoAEndcapDiskm2);
    hnDigisSoAEndcapDiskm3->Fill(nDigisSoAEndcapDiskm3);

    //hnDigisSoAEndcapDiskm4->Fill(nDigisSoAEndcapDiskm4);
    //hnDigisSoAEndcapDiskm5->Fill(nDigisSoAEndcapDiskm5);
    //hnDigisSoAEndcapDiskm6->Fill(nDigisSoAEndcapDiskm6);
    hnDigisSoAEndcapDiskp1->Fill(nDigisSoAEndcapDiskp1);
    hnDigisSoAEndcapDiskp2->Fill(nDigisSoAEndcapDiskp2);
    hnDigisSoAEndcapDiskp3->Fill(nDigisSoAEndcapDiskp3);
    
    //hnDigisSoAEndcapDiskp4->Fill(nDigisSoAEndcapDiskp4);
    //hnDigisSoAEndcapDiskp5->Fill(nDigisSoAEndcapDiskp5);
    //hnDigisSoAEndcapDiskp6->Fill(nDigisSoAEndcapDiskp6);

  }

  edm::Handle<edm::DetSetVector<PixelDigi>> input;
  iEvent.getByToken(tokenDigi_, input);
  if (!input.isValid()){
    edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "No Digi found \n returning!" << std::endl;
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
    //uint32_t nDigisEndcapDiskm4 = 0;
    //uint32_t nDigisEndcapDiskm5 = 0;
    //uint32_t nDigisEndcapDiskm6 = 0;
    uint32_t nDigisEndcapDiskp1 = 0;
    uint32_t nDigisEndcapDiskp2 = 0;
    uint32_t nDigisEndcapDiskp3 = 0;
    //uint32_t nDigisEndcapDiskp4 = 0;
    //uint32_t nDigisEndcapDiskp5 = 0;
    //uint32_t nDigisEndcapDiskp6 = 0;
    
    for (it = input->begin(); it != input->end(); ++it) {
      const uint32_t nDigisEv = it->size();
      nDigis += nDigisEv;
      DetId id = it->detId();
      uint32_t subdetid = (id.subdetId());

      if (subdetid == PixelSubdetector::PixelBarrel) {
	//edm::LogWarning("SiPixelPhase1MonitorDigi") << " PX Barrel:  DetId " <<id.rawId()<<" Layer "<<trackerTopology_->pxbLayer(id)<<std::endl;
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
	//edm::LogWarning("SiPixelPhase1MonitorDigi") << " PX Endcaps:  DetId " <<id.rawId()<<" Side "<<trackerTopology_->pxfSide(id)<<" Disk "<<trackerTopology_->pxfDisk(id)<<std::endl;
     
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

	  //else if(nDisk == 4) nDigisEndcapDiskm4 += nDigisEv;
	  //else if(nDisk == 5) nDigisEndcapDiskm5 += nDigisEv;
	  //else if(nDisk == 6) nDigisEndcapDiskm6 += nDigisEv;
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

	  //else if(nDisk == 4) nDigisEndcapDiskp4 += nDigisEv;
	  //else if(nDisk == 5) nDigisEndcapDiskp5 += nDigisEv;
	  //else if(nDisk == 6) nDigisEndcapDiskp6 += nDigisEv;
	}
      }

      //looping over digis to take adcs
      //edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "nDigis: " << nDigis << "\tnDigisEndcap: " << nDigisEndcap << "\tnDigisBarrel: " << nDigisBarrel << std::endl;
    }
    
    //edm::LogWarning("SiPixelPhase1MonitorDigiSoA") << "End" << std::endl << "nDigis: " << nDigis << "\tnDigisEndcap: " << nDigisEndcap << "\tnDigisBarrel: " << nDigisBarrel << std::endl;

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
    //hnDigisEndcapDiskm4->Fill(nDigisEndcapDiskm4);
    //hnDigisEndcapDiskm5->Fill(nDigisEndcapDiskm5);
    //hnDigisEndcapDiskm6->Fill(nDigisEndcapDiskm6);
    hnDigisEndcapDiskp1->Fill(nDigisEndcapDiskp1);
    hnDigisEndcapDiskp2->Fill(nDigisEndcapDiskp2);
    hnDigisEndcapDiskp3->Fill(nDigisEndcapDiskp3);
    //hnDigisEndcapDiskp4->Fill(nDigisEndcapDiskp4);
    //hnDigisEndcapDiskp5->Fill(nDigisEndcapDiskp5);
    //hnDigisEndcapDiskp6->Fill(nDigisEndcapDiskp6);

    //edm::LogWarning("SiPixelPhase1MonitorDigi") << "Found "<<nDigis<<" Digis!" << std::endl;
    //edm::LogWarning("SiPixelPhase1MonitorDigi") << "Found "<<nDigisBarrel<<" DigisBarrel!" << std::endl;
    //edm::LogWarning("SiPixelPhase1MonitorDigi") << "Found "<<nDigisEndcap<<" DigisEndcap!" << std::endl;

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
  hnDigisBarrel = ibooker.book1D("nDigisBarrel", ";Number of digis per event;#entries", 1001, -0.5, 10000.5);
  hnDigisSoABarrel = ibooker.book1D("nDigisSoABarrel", ";Number of digis per event;#entries", 1001, -0.5, 10000.5);
  hnDigisEndcap = ibooker.book1D("nDigisEndcap", ";Number of digis per event;#entries", 1001, -0.5, 10000.5);
  hnDigisSoAEndcap = ibooker.book1D("nDigisSoAEndcap", ";Number of digis per event;#entries", 1001, -0.5, 10000.5);

  hnDigisBarrelLayer1 = ibooker.book1D("nDigisBarrelLayer1", ";Number of digis per event;#entries", 151, -0.5, 150.5);
  hnDigisBarrelLayer2 = ibooker.book1D("nDigisBarrelLayer2", ";Number of digis per event;#entries", 151, -0.5, 150.5);
  hnDigisBarrelLayer3 = ibooker.book1D("nDigisBarrelLayer3", ";Number of digis per event;#entries", 151, -0.5, 150.5);
  hnDigisBarrelLayer4 = ibooker.book1D("nDigisBarrelLayer4", ";Number of digis per event;#entries", 151, -0.5, 150.5);

  hnDigisEndcapDiskm1 = ibooker.book1D("nDigisEndcapDiskm1", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskm2 = ibooker.book1D("nDigisEndcapDiskm2", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskm3 = ibooker.book1D("nDigisEndcapDiskm3", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisEndcapDiskm4 = ibooker.book1D("nDigisEndcapDiskm4", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisEndcapDiskm5 = ibooker.book1D("nDigisEndcapDiskm5", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisEndcapDiskm6 = ibooker.book1D("nDigisEndcapDiskm6", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskp1 = ibooker.book1D("nDigisEndcapDiskp1", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskp2 = ibooker.book1D("nDigisEndcapDiskp2", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisEndcapDiskp3 = ibooker.book1D("nDigisEndcapDiskp3", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisEndcapDiskp4 = ibooker.book1D("nDigisEndcapDiskp4", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisEndcapDiskp5 = ibooker.book1D("nDigisEndcapDiskp5", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisEndcapDiskp6 = ibooker.book1D("nDigisEndcapDiskp6", ";Number of digis per event;#entries", 16, -0.5, 15.5);

  hnDigisSoABarrelLayer1 = ibooker.book1D("nDigisSoABarrelLayer1", ";Number of digis per event;#entries", 151, -0.5, 150.5);
  hnDigisSoABarrelLayer2 = ibooker.book1D("nDigisSoABarrelLayer2", ";Number of digis per event;#entries", 151, -0.5, 150.5);
  hnDigisSoABarrelLayer3 = ibooker.book1D("nDigisSoABarrelLayer3", ";Number of digis per event;#entries", 151, -0.5, 150.5);
  hnDigisSoABarrelLayer4 = ibooker.book1D("nDigisSoABarrelLayer4", ";Number of digis per event;#entries", 151, -0.5, 150.5);

  hnDigisSoAEndcapDiskm1 = ibooker.book1D("nDigisSoAEndcapDiskm1", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisSoAEndcapDiskm2 = ibooker.book1D("nDigisSoAEndcapDiskm2", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisSoAEndcapDiskm3 = ibooker.book1D("nDigisSoAEndcapDiskm3", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisSoAEndcapDiskm4 = ibooker.book1D("nDigisSoAEndcapDiskm4", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisSoAEndcapDiskm5 = ibooker.book1D("nDigisSoAEndcapDiskm5", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisSoAEndcapDiskm6 = ibooker.book1D("nDigisSoAEndcapDiskm6", ";Number of digis per event;#entries", 16, -0.5, 15.5);

  hnDigisSoAEndcapDiskp1 = ibooker.book1D("nDigisSoAEndcapDiskp1", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisSoAEndcapDiskp2 = ibooker.book1D("nDigisSoAEndcapDiskp2", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  hnDigisSoAEndcapDiskp3 = ibooker.book1D("nDigisSoAEndcapDiskp3", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisSoAEndcapDiskp4 = ibooker.book1D("nDigisSoAEndcapDiskp4", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisSoAEndcapDiskp5 = ibooker.book1D("nDigisSoAEndcapDiskp5", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  //hnDigisSoAEndcapDiskp6 = ibooker.book1D("nDigisSoAEndcapDiskp6", ";Number of digis per event;#entries", 16, -0.5, 15.5);
  
  hDigisADC = ibooker.book1D("DigisADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisBarrelADC = ibooker.book1D("DigisBarrelADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapADC = ibooker.book1D("DigisEndcapADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisSoAADC = ibooker.book1D("DigisSoAADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoABarrelADC = ibooker.book1D("DigisSoABarrelADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoAEndcapADC = ibooker.book1D("DigisSoAEndcapADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);

  hDigisBarrelLayer1ADC = ibooker.book1D("DigisBarrelLayer1ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisBarrelLayer2ADC = ibooker.book1D("DigisBarrelLayer2ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisBarrelLayer3ADC = ibooker.book1D("DigisBarrelLayer3ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisBarrelLayer4ADC = ibooker.book1D("DigisBarrelLayer4ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);

  hDigisEndcapDiskm1ADC = ibooker.book1D("DigisEndcapDiskm1ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskm2ADC = ibooker.book1D("DigisEndcapDiskm2ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskm3ADC = ibooker.book1D("DigisEndcapDiskm3ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  //hDigisEndcapDiskm4ADC = ibooker.book1D("DigisEndcapDiskm4ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  //hDigisEndcapDiskm5ADC = ibooker.book1D("DigisEndcapDiskm5ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  //hDigisEndcapDiskm6ADC = ibooker.book1D("DigisEndcapDiskm6ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskp1ADC = ibooker.book1D("DigisEndcapDiskp1ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskp2ADC = ibooker.book1D("DigisEndcapDiskp2ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  hDigisEndcapDiskp3ADC = ibooker.book1D("DigisEndcapDiskp3ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  //hDigisEndcapDiskp4ADC = ibooker.book1D("DigisEndcapDiskp4ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  //hDigisEndcapDiskp5ADC = ibooker.book1D("DigisEndcapDiskp5ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);
  //hDigisEndcapDiskp6ADC = ibooker.book1D("DigisEndcapDiskp6ADC", ";Digis ADC per event;#entries", 501, -0.5, 500.5);

  hDigisSoABarrelLayer1ADC = ibooker.book1D("DigisSoABarrelLayer1ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoABarrelLayer2ADC = ibooker.book1D("DigisSoABarrelLayer2ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoABarrelLayer3ADC = ibooker.book1D("DigisSoABarrelLayer3ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoABarrelLayer4ADC = ibooker.book1D("DigisSoABarrelLayer4ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);

  hDigisSoAEndcapDiskm1ADC = ibooker.book1D("DigisSoAEndcapDiskm1ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoAEndcapDiskm2ADC = ibooker.book1D("DigisSoAEndcapDiskm2ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoAEndcapDiskm3ADC = ibooker.book1D("DigisSoAEndcapDiskm3ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  //hDigisSoAEndcapDiskm4ADC = ibooker.book1D("DigisSoAEndcapDiskm4ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  //hDigisSoAEndcapDiskm5ADC = ibooker.book1D("DigisSoAEndcapDiskm5ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  //hDigisSoAEndcapDiskm6ADC = ibooker.book1D("DigisSoAEndcapDiskm6ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoAEndcapDiskp1ADC = ibooker.book1D("DigisSoAEndcapDiskp1ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoAEndcapDiskp2ADC = ibooker.book1D("DigisSoAEndcapDiskp2ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  hDigisSoAEndcapDiskp3ADC = ibooker.book1D("DigisSoAEndcapDiskp3ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  //hDigisSoAEndcapDiskp4ADC = ibooker.book1D("DigisSoAEndcapDiskp4ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  //hDigisSoAEndcapDiskp5ADC = ibooker.book1D("DigisSoAEndcapDiskp5ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
  //hDigisSoAEndcapDiskp6ADC = ibooker.book1D("DigisSoAEndcapDiskp6ADC", ";DigisSoA ADC per event;#entries", 50001, -0.5, 50000.5);
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
