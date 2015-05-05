#ifndef _DQMOFFLINE_HCAL_CALOTOWERSANALYZER_H_
#define _DQMOFFLINE_HCAL_CALOTOWERSANALYZER_H_

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>
#include "DQMServices/Core/interface/MonitorElement.h"


class CaloTowersAnalyzer : public DQMEDAnalyzer {
 public:
   CaloTowersAnalyzer(edm::ParameterSet const& conf);
  ~CaloTowersAnalyzer();
  
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun() ;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  virtual void endRun() ;

 private:
  double dR(double eta1, double phi1, double eta2, double phi2);
   
  std::string outputFile_;
  std::string hcalselector_;
  std::string mc_;
  bool        useAllHistos_;

  typedef math::RhoEtaPhiVector Vector;

  edm::EDGetTokenT<CaloTowerCollection> tok_towers_;

  int isub;
  int nevent;

  int imc;

  // eta limits to calcualte MET, SET (not to include HF if not needed)
  double etaMax[3];
  double etaMin[3];

  // ieta scan
  MonitorElement*  emean_vs_ieta_E;
  MonitorElement*  emean_vs_ieta_H;
  MonitorElement*  emean_vs_ieta_EH;

  MonitorElement*  emean_vs_ieta_E1;
  MonitorElement*  emean_vs_ieta_H1;
  MonitorElement*  emean_vs_ieta_EH1;

  MonitorElement* Ntowers_vs_ieta;
  MonitorElement* occupancy_map;
  MonitorElement* occupancy_vs_ieta;

  // Global maps
  MonitorElement*  mapEnergy_E;
  MonitorElement*  mapEnergy_H;
  MonitorElement*  mapEnergy_EH;
  MonitorElement*  mapEnergy_N;

  // for number of bad, recovered and problematic Ecal and Hcal cells
  MonitorElement* numBadCellsEcal_EB;
  MonitorElement* numBadCellsEcal_EE;
  MonitorElement* numRcvCellsEcal_EB;
  MonitorElement* numRcvCellsEcal_EE;
  MonitorElement* numPrbCellsEcal_EB;
  MonitorElement* numPrbCellsEcal_EE;

  MonitorElement* numBadCellsHcal_HB;
  MonitorElement* numBadCellsHcal_HE;
  MonitorElement* numBadCellsHcal_HF;
  MonitorElement* numRcvCellsHcal_HB;
  MonitorElement* numRcvCellsHcal_HE;
  MonitorElement* numRcvCellsHcal_HF;
  MonitorElement* numPrbCellsHcal_HB;
  MonitorElement* numPrbCellsHcal_HE;
  MonitorElement* numPrbCellsHcal_HF; 

  // HB
  MonitorElement* meEnergyHcalvsEcal_HB;
  MonitorElement* meEnergyHO_HB; 
  MonitorElement* meEnergyEcal_HB; 
  MonitorElement* meEnergyHcal_HB; 
  MonitorElement* meNumFiredTowers_HB;

  MonitorElement* meEnergyEcalTower_HB;
  MonitorElement* meEnergyHcalTower_HB;
  MonitorElement* meTotEnergy_HB;

  MonitorElement* mapEnergy_HB;
  MonitorElement* mapEnergyEcal_HB;
  MonitorElement* mapEnergyHcal_HB;
  MonitorElement* MET_HB;
  MonitorElement* SET_HB;
  MonitorElement* phiMET_HB;

  MonitorElement* emTiming_HB;
  MonitorElement* hadTiming_HB;

  MonitorElement* emEnergyTiming_Low_HB;
  MonitorElement* emEnergyTiming_HB;
  MonitorElement* emEnergyTiming_High_HB;
  MonitorElement* emEnergyTiming_profile_Low_HB;
  MonitorElement* emEnergyTiming_profile_HB;
  MonitorElement* emEnergyTiming_profile_High_HB;

  MonitorElement* hadEnergyTiming_Low_HB;
  MonitorElement* hadEnergyTiming_HB;
  MonitorElement* hadEnergyTiming_High_HB;
  MonitorElement* hadEnergyTiming_profile_Low_HB;
  MonitorElement* hadEnergyTiming_profile_HB;
  MonitorElement* hadEnergyTiming_profile_High_HB;

  // HE
  MonitorElement* meEnergyHcalvsEcal_HE;
  MonitorElement* meEnergyHO_HE; 
  MonitorElement* meEnergyEcal_HE; 
  MonitorElement* meEnergyHcal_HE; 
  MonitorElement* meNumFiredTowers_HE;

  MonitorElement* meEnergyEcalTower_HE;
  MonitorElement* meEnergyHcalTower_HE;
  MonitorElement* meTotEnergy_HE;

  MonitorElement* mapEnergy_HE;
  MonitorElement* mapEnergyEcal_HE;
  MonitorElement* mapEnergyHcal_HE;
  MonitorElement* MET_HE;
  MonitorElement* SET_HE;
  MonitorElement* phiMET_HE;

  MonitorElement* emTiming_HE;
  MonitorElement* hadTiming_HE;

  MonitorElement* emEnergyTiming_Low_HE;
  MonitorElement* emEnergyTiming_HE;
  MonitorElement* emEnergyTiming_profile_Low_HE;
  MonitorElement* emEnergyTiming_profile_HE;

  MonitorElement* hadEnergyTiming_Low_HE;
  MonitorElement* hadEnergyTiming_HE;
  MonitorElement* hadEnergyTiming_profile_Low_HE;
  MonitorElement* hadEnergyTiming_profile_HE;

  // HF
  MonitorElement* meEnergyHcalvsEcal_HF;
  MonitorElement* meEnergyHO_HF; 
  MonitorElement* meEnergyEcal_HF; 
  MonitorElement* meEnergyHcal_HF; 
  MonitorElement* meNumFiredTowers_HF;

  MonitorElement* meEnergyEcalTower_HF;
  MonitorElement* meEnergyHcalTower_HF;
  MonitorElement* meTotEnergy_HF;

  MonitorElement* mapEnergy_HF;
  MonitorElement* mapEnergyEcal_HF;
  MonitorElement* mapEnergyHcal_HF;
  MonitorElement* MET_HF;
  MonitorElement* SET_HF;
  MonitorElement* phiMET_HF;

  MonitorElement* emTiming_HF;
  MonitorElement* hadTiming_HF;
  MonitorElement* emEnergyTiming_HF;
  MonitorElement* emEnergyTiming_profile_HF;

  MonitorElement* hadEnergyTiming_Low_HF;
  MonitorElement* hadEnergyTiming_HF;
  MonitorElement* hadEnergyTiming_profile_Low_HF;
  MonitorElement* hadEnergyTiming_profile_HF;

};

#endif
