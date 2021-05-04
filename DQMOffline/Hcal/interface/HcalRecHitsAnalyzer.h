#ifndef _DQMOFFLINE_HCAL_HCALRECHITSANALYZER_H_
#define _DQMOFFLINE_HCAL_HCALRECHITSANALYZER_H_

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include <algorithm>
#include <cmath>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
// channel status
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

// severity level assignment for HCAL
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

// severity level assignment for ECAL
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

class HcalRecHitsAnalyzer : public DQMEDAnalyzer {
public:
  HcalRecHitsAnalyzer(edm::ParameterSet const &conf);

  void analyze(edm::Event const &ev, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  // virtual void beginRun(edm::Run const& run, edm::EventSetup const&)
  // override;
  void dqmBeginRun(const edm::Run &run, const edm::EventSetup &) override;

private:
  virtual void fillRecHitsTmp(int subdet_, edm::Event const &ev);
  double dR(double eta1, double phi1, double eta2, double phi2);
  double phi12(double phi1, double en1, double phi2, double en2);
  double dPhiWsign(double phi1, double phi2);

  std::string topFolderName_;

  std::string outputFile_;
  std::string hcalselector_;
  std::string ecalselector_;
  std::string eventype_;
  std::string sign_;
  bool hep17_;
  std::string mc_;
  bool famos_;

  int maxDepthHB_, maxDepthHE_, maxDepthHO_, maxDepthHF_, maxDepthAll_;

  int nChannels_[5];  // 0:any, 1:HB, 2:HE

  int iphi_bins_;
  float iphi_min_, iphi_max_;

  int ieta_bins_;
  float ieta_min_, ieta_max_;

  // RecHit Collection input tags
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
  edm::EDGetTokenT<EBRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EERecHitCollection> tok_EE_;

  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> hcalDDDRecConstantsToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryRunToken_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryEventToken_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> hcalTopologyToken_;
  edm::ESGetToken<HcalChannelQuality, HcalChannelQualityRcd> hcalChannelQualityToken_;
  edm::ESGetToken<HcalSeverityLevelComputer, HcalSeverityLevelComputerRcd> hcalSeverityLevelComputerToken_;

  // choice of subdetector in config : noise/HB/HE/HO/HF/ALL (0/1/2/3/4/5)
  int subdet_;

  // single/multi-particle sample (1/2)
  int etype_;
  int iz;
  int imc;

  // Hcal topology
  const HcalTopology *theHcalTopology = nullptr;
  // for checking the status of ECAL and HCAL channels stored in the DB
  const HcalChannelQuality *theHcalChStatus = nullptr;
  // calculator of severety level for HCAL
  const HcalSeverityLevelComputer *theHcalSevLvlComputer = nullptr;
  int hcalSevLvl(const CaloRecHit *hit);

  std::vector<int> hcalHBSevLvlVec, hcalHESevLvlVec, hcalHFSevLvlVec, hcalHOSevLvlVec;

  MonitorElement *sevLvl_HB;
  MonitorElement *sevLvl_HE;
  MonitorElement *sevLvl_HF;
  MonitorElement *sevLvl_HO;

  // RecHits counters
  std::vector<MonitorElement *> Nhb;
  std::vector<MonitorElement *> Nhe;
  std::vector<MonitorElement *> Nho;
  std::vector<MonitorElement *> Nhf;

  // In ALL other cases : 2D ieta-iphi maps
  // without and with cuts (a la "Scheme B") on energy
  // - only in the cone around particle for single-part samples (mc = "yes")
  // - for all calls in milti-particle samples (mc = "no")

  MonitorElement *map_ecal;

  std::vector<MonitorElement *> emap;
  MonitorElement *emap_HO;

  std::vector<MonitorElement *> emean_vs_ieta_HB;
  std::vector<MonitorElement *> emean_vs_ieta_HBM0;
  std::vector<MonitorElement *> emean_vs_ieta_HBM3;
  std::vector<MonitorElement *> emean_vs_ieta_HE;
  std::vector<MonitorElement *> emean_vs_ieta_HEM0;
  std::vector<MonitorElement *> emean_vs_ieta_HEM3;
  std::vector<MonitorElement *> emean_vs_ieta_HEP17;
  std::vector<MonitorElement *> emean_vs_ieta_HEP17M0;
  std::vector<MonitorElement *> emean_vs_ieta_HEP17M3;
  std::vector<MonitorElement *> emean_vs_ieta_HF;
  MonitorElement *emean_vs_ieta_HO;

  std::vector<MonitorElement *> occupancy_map_HB;
  std::vector<MonitorElement *> occupancy_map_HE;
  std::vector<MonitorElement *> occupancy_map_HF;
  MonitorElement *occupancy_map_HO;

  std::vector<MonitorElement *> occupancy_vs_ieta_HB;
  std::vector<MonitorElement *> occupancy_vs_ieta_HE;
  std::vector<MonitorElement *> occupancy_vs_ieta_HF;
  MonitorElement *occupancy_vs_ieta_HO;

  std::vector<MonitorElement *> nrechits_vs_iphi_HBP, nrechits_vs_iphi_HBM;
  std::vector<MonitorElement *> nrechits_vs_iphi_HEP, nrechits_vs_iphi_HEM;
  std::vector<MonitorElement *> nrechits_vs_iphi_HFP, nrechits_vs_iphi_HFM;
  MonitorElement *nrechits_vs_iphi_HOP, *nrechits_vs_iphi_HOM;

  // for single monoenergetic particles - cone collection profile vs ieta.
  MonitorElement *meEnConeEtaProfile;
  MonitorElement *meEnConeEtaProfile_E;
  MonitorElement *meEnConeEtaProfile_EH;
  // Single particles - deviation of cluster from MC truth
  MonitorElement *meDeltaPhi;
  MonitorElement *meDeltaEta;

  // time?
  MonitorElement *meTimeHB;
  MonitorElement *meTimeHE;
  MonitorElement *meTimeHO;
  MonitorElement *meTimeHF;

  // energy of rechits
  MonitorElement *meRecHitsEnergyHB;
  MonitorElement *meRecHitsCleanedEnergyHB;
  MonitorElement *meRecHitsEnergyHBM0;
  MonitorElement *meRecHitsEnergyHBM3;
  MonitorElement *meRecHitsEnergyM2vM0HB;
  MonitorElement *meRecHitsEnergyM3vM0HB;
  MonitorElement *meRecHitsEnergyM3vM2HB;
  MonitorElement *meRecHitsM2Chi2HB;

  MonitorElement *meRecHitsEnergyHE;
  MonitorElement *meRecHitsCleanedEnergyHE;
  MonitorElement *meRecHitsEnergyHEM0;
  MonitorElement *meRecHitsEnergyHEM3;
  std::vector<MonitorElement *> meRecHitsEnergyHEP17;
  std::vector<MonitorElement *> meRecHitsEnergyHEP17M0;
  std::vector<MonitorElement *> meRecHitsEnergyHEP17M3;
  MonitorElement *meRecHitsEnergyM2vM0HE;
  MonitorElement *meRecHitsEnergyM3vM0HE;
  MonitorElement *meRecHitsEnergyM3vM2HE;
  MonitorElement *meRecHitsM2Chi2HE;

  MonitorElement *meRecHitsEnergyHO;
  MonitorElement *meRecHitsCleanedEnergyHO;

  MonitorElement *meRecHitsEnergyHF;
  MonitorElement *meRecHitsCleanedEnergyHF;

  MonitorElement *meTE_Low_HB;
  MonitorElement *meTE_HB;
  MonitorElement *meTE_High_HB;
  MonitorElement *meTEprofileHB_Low;
  MonitorElement *meTEprofileHB;
  MonitorElement *meLog10Chi2profileHB;
  MonitorElement *meTEprofileHB_High;

  MonitorElement *meTE_Low_HE;
  MonitorElement *meTE_HE;
  MonitorElement *meTEprofileHE_Low;
  MonitorElement *meTEprofileHE;
  MonitorElement *meLog10Chi2profileHE;

  MonitorElement *meTE_HO;
  MonitorElement *meTE_High_HO;
  MonitorElement *meTEprofileHO;
  MonitorElement *meTEprofileHO_High;

  MonitorElement *meTE_Low_HF;
  MonitorElement *meTE_HF;
  MonitorElement *meTEprofileHF_Low;
  MonitorElement *meTEprofileHF;

  MonitorElement *meSumRecHitsEnergyHB;
  MonitorElement *meSumRecHitsEnergyHE;
  MonitorElement *meSumRecHitsEnergyHO;
  MonitorElement *meSumRecHitsEnergyHF;

  MonitorElement *meSumRecHitsEnergyConeHB;
  MonitorElement *meSumRecHitsEnergyConeHE;
  MonitorElement *meSumRecHitsEnergyConeHO;
  MonitorElement *meSumRecHitsEnergyConeHF;
  MonitorElement *meSumRecHitsEnergyConeHFL;
  MonitorElement *meSumRecHitsEnergyConeHFS;

  MonitorElement *meEcalHcalEnergyHB;
  MonitorElement *meEcalHcalEnergyHE;

  MonitorElement *meEcalHcalEnergyConeHB;
  MonitorElement *meEcalHcalEnergyConeHE;
  MonitorElement *meEcalHcalEnergyConeHO;
  MonitorElement *meEcalHcalEnergyConeHF;

  // 2D plot of sum of RecHits in HCAL as function of ECAL's one
  MonitorElement *meEnergyHcalVsEcalHB;
  MonitorElement *meEnergyHcalVsEcalHE;

  // number of ECAL's rechits in cone 0.3
  MonitorElement *meNumEcalRecHitsConeHB;
  MonitorElement *meNumEcalRecHitsConeHE;

  CaloGeometry const *geometry = nullptr;

  // Status word histos
  MonitorElement *RecHit_StatusWord_HB;
  MonitorElement *RecHit_StatusWord_HE;
  MonitorElement *RecHit_StatusWord_HF;
  MonitorElement *RecHit_StatusWord_HF67;
  MonitorElement *RecHit_StatusWord_HO;

  // Status word correlation
  MonitorElement *RecHit_StatusWordCorr_HB;
  MonitorElement *RecHit_StatusWordCorr_HE;

  // Aux Status word histos
  MonitorElement *RecHit_Aux_StatusWord_HB;
  MonitorElement *RecHit_Aux_StatusWord_HE;
  MonitorElement *RecHit_Aux_StatusWord_HF;
  MonitorElement *RecHit_Aux_StatusWord_HO;

  // Filling vectors with essential RecHits data
  std::vector<int> csub;
  std::vector<int> cieta;
  std::vector<int> ciphi;
  std::vector<int> cdepth;
  std::vector<double> cen;
  std::vector<double> cenM0;
  std::vector<double> cenM3;
  std::vector<double> cchi2;
  std::vector<double> ceta;
  std::vector<double> cphi;
  std::vector<double> ctime;
  std::vector<double> cz;
  std::vector<uint32_t> cstwd;
  std::vector<uint32_t> cauxstwd;
  std::vector<int> csevlev;

  // counter
  int nevtot;
};

#endif
