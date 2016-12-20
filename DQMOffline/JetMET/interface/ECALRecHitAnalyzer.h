#ifndef ECALRECHITANALYZER_H
#define ECALRECHITANALYZER_H

// author: Bobby Scurlock (The University of Florida)
// date: 11/20/2006

#include <memory>
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

//--egamma Reco stuff--//
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include <memory>
#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <TLorentzVector.h>
#include <string>
#include <map>

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TMath.h>
#include "DQMServices/Core/interface/MonitorElement.h"

class DetId;
//class HcalTopology;
class CaloGeometry;
class CaloSubdetectorGeometry;
//class CaloTowerConstituentsMap;
//class CaloRecHit;


//
// class declaration
//

class ECALRecHitAnalyzer : public DQMEDAnalyzer {
public:

  ECALRecHitAnalyzer(const edm::ParameterSet&);
  //~ECALRecHitAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //  virtual void beginJob(void) ;
  virtual void dqmbeginRun(const edm::Run&, const edm::EventSetup&) ;

  void WriteECALRecHits(const edm::Event&, const edm::EventSetup&);
  void FillGeometry(const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

 private:


  // Inputs from Configuration
  edm::EDGetTokenT<EBRecHitCollection> EBRecHitsLabel_;
  edm::EDGetTokenT<EERecHitCollection> EERecHitsLabel_;

  bool debug_;
  bool finebinning_;
  std::string FolderName_;
  int CurrentEvent;

  //histos
  MonitorElement* hEB_ieta_iphi_etaMap;
  MonitorElement* hEB_ieta_iphi_phiMap;
  MonitorElement* hEB_ieta_detaMap;
  MonitorElement* hEB_ieta_dphiMap;
  
  MonitorElement* hEEpZ_ix_iy_irMap;
  MonitorElement* hEEpZ_ix_iy_xMap;
  MonitorElement* hEEpZ_ix_iy_yMap;
  MonitorElement* hEEpZ_ix_iy_zMap;
  MonitorElement* hEEpZ_ix_iy_dxMap;  
  MonitorElement* hEEpZ_ix_iy_dyMap;
  
  MonitorElement* hEEmZ_ix_iy_irMap;
  MonitorElement* hEEmZ_ix_iy_xMap;
  MonitorElement* hEEmZ_ix_iy_yMap;
  MonitorElement* hEEmZ_ix_iy_zMap;
  MonitorElement* hEEmZ_ix_iy_dxMap;  
  MonitorElement* hEEmZ_ix_iy_dyMap;
  
  MonitorElement* hECAL_Nevents;
  
  MonitorElement* hEEpZ_energy_ix_iy;
  MonitorElement* hEEmZ_energy_ix_iy;
  MonitorElement* hEB_energy_ieta_iphi;
  
  MonitorElement* hEEpZ_Minenergy_ix_iy;
  MonitorElement* hEEmZ_Minenergy_ix_iy;
  MonitorElement* hEB_Minenergy_ieta_iphi;
  
  MonitorElement* hEEpZ_Maxenergy_ix_iy;
  MonitorElement* hEEmZ_Maxenergy_ix_iy;
  MonitorElement* hEB_Maxenergy_ieta_iphi;
  
  MonitorElement* hEEpZ_Occ_ix_iy; 
  MonitorElement* hEEmZ_Occ_ix_iy;
  MonitorElement* hEB_Occ_ieta_iphi;
  
  MonitorElement* hEEpZ_energyvsir;
  MonitorElement* hEEmZ_energyvsir;
  MonitorElement* hEB_energyvsieta;
  
  MonitorElement* hEEpZ_Maxenergyvsir;
  MonitorElement* hEEmZ_Maxenergyvsir;
  MonitorElement* hEB_Maxenergyvsieta;
  
  MonitorElement* hEEpZ_Minenergyvsir;
  MonitorElement* hEEmZ_Minenergyvsir;
  MonitorElement* hEB_Minenergyvsieta;
  
  MonitorElement* hEEpZ_SETvsir;
  MonitorElement* hEEmZ_SETvsir;
  MonitorElement* hEB_SETvsieta;
  
  MonitorElement* hEEpZ_METvsir;
  MonitorElement* hEEmZ_METvsir;
  MonitorElement* hEB_METvsieta;
  
  MonitorElement* hEEpZ_METPhivsir;
  MonitorElement* hEEmZ_METPhivsir;
  MonitorElement* hEB_METPhivsieta;
  
  MonitorElement* hEEpZ_MExvsir;
  MonitorElement* hEEmZ_MExvsir;
  MonitorElement* hEB_MExvsieta;
  
  MonitorElement* hEEpZ_MEyvsir;
  MonitorElement* hEEmZ_MEyvsir;
  MonitorElement* hEB_MEyvsieta;
  
  MonitorElement* hEEpZ_Occvsir;
  MonitorElement* hEEmZ_Occvsir;
  MonitorElement* hEB_Occvsieta;
};

#endif
