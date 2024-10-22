#ifndef HCALRECHITANALYZER_H
#define HCALRECHITANALYZER_H

// author: Bobby Scurlock (The University of Florida)
// date: 8/24/2006
// modification: Mike Schmitt
// date: 02.28.2007
// note: code rewrite

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include <string>
#include <map>
class CaloGeometry;
class CaloGeometryRecord;
class HCALRecHitAnalyzer : public DQMEDAnalyzer {
public:
  explicit HCALRecHitAnalyzer(const edm::ParameterSet&);

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  //  virtual void beginJob(void);
  //  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;

private:
  // Inputs from Configuration
  edm::EDGetTokenT<HBHERecHitCollection> hBHERecHitsLabel_;
  edm::EDGetTokenT<HFRecHitCollection> hFRecHitsLabel_;
  edm::EDGetTokenT<HORecHitCollection> hORecHitsLabel_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeomToken_;
  bool debug_;
  bool finebinning_;
  std::string FolderName_;
  // Helper Functions
  void FillGeometry(const edm::EventSetup&);

  int Nevents;

  //histos
  MonitorElement* hHCAL_ieta_iphi_HBMap;
  MonitorElement* hHCAL_ieta_iphi_HEMap;
  MonitorElement* hHCAL_ieta_iphi_HFMap;
  MonitorElement* hHCAL_ieta_iphi_HOMap;
  MonitorElement* hHCAL_ieta_iphi_etaMap;
  MonitorElement* hHCAL_ieta_iphi_phiMap;
  MonitorElement* hHCAL_ieta_detaMap;
  MonitorElement* hHCAL_ieta_dphiMap;

  MonitorElement* hHCAL_Nevents;

  MonitorElement* hHCAL_D1_energy_ieta_iphi;
  MonitorElement* hHCAL_D2_energy_ieta_iphi;
  MonitorElement* hHCAL_D3_energy_ieta_iphi;
  MonitorElement* hHCAL_D4_energy_ieta_iphi;

  MonitorElement* hHCAL_D1_Minenergy_ieta_iphi;
  MonitorElement* hHCAL_D2_Minenergy_ieta_iphi;
  MonitorElement* hHCAL_D3_Minenergy_ieta_iphi;
  MonitorElement* hHCAL_D4_Minenergy_ieta_iphi;

  MonitorElement* hHCAL_D1_Maxenergy_ieta_iphi;
  MonitorElement* hHCAL_D2_Maxenergy_ieta_iphi;
  MonitorElement* hHCAL_D3_Maxenergy_ieta_iphi;
  MonitorElement* hHCAL_D4_Maxenergy_ieta_iphi;

  MonitorElement* hHCAL_D1_Occ_ieta_iphi;
  MonitorElement* hHCAL_D2_Occ_ieta_iphi;
  MonitorElement* hHCAL_D3_Occ_ieta_iphi;
  MonitorElement* hHCAL_D4_Occ_ieta_iphi;

  MonitorElement* hHCAL_D1_energyvsieta;
  MonitorElement* hHCAL_D2_energyvsieta;
  MonitorElement* hHCAL_D3_energyvsieta;
  MonitorElement* hHCAL_D4_energyvsieta;

  MonitorElement* hHCAL_D1_Minenergyvsieta;
  MonitorElement* hHCAL_D2_Minenergyvsieta;
  MonitorElement* hHCAL_D3_Minenergyvsieta;
  MonitorElement* hHCAL_D4_Minenergyvsieta;

  MonitorElement* hHCAL_D1_Maxenergyvsieta;
  MonitorElement* hHCAL_D2_Maxenergyvsieta;
  MonitorElement* hHCAL_D3_Maxenergyvsieta;
  MonitorElement* hHCAL_D4_Maxenergyvsieta;

  MonitorElement* hHCAL_D1_Occvsieta;
  MonitorElement* hHCAL_D2_Occvsieta;
  MonitorElement* hHCAL_D3_Occvsieta;
  MonitorElement* hHCAL_D4_Occvsieta;

  MonitorElement* hHCAL_D1_SETvsieta;
  MonitorElement* hHCAL_D2_SETvsieta;
  MonitorElement* hHCAL_D3_SETvsieta;
  MonitorElement* hHCAL_D4_SETvsieta;

  MonitorElement* hHCAL_D1_METvsieta;
  MonitorElement* hHCAL_D2_METvsieta;
  MonitorElement* hHCAL_D3_METvsieta;
  MonitorElement* hHCAL_D4_METvsieta;

  MonitorElement* hHCAL_D1_METPhivsieta;
  MonitorElement* hHCAL_D2_METPhivsieta;
  MonitorElement* hHCAL_D3_METPhivsieta;
  MonitorElement* hHCAL_D4_METPhivsieta;

  MonitorElement* hHCAL_D1_MExvsieta;
  MonitorElement* hHCAL_D2_MExvsieta;
  MonitorElement* hHCAL_D3_MExvsieta;
  MonitorElement* hHCAL_D4_MExvsieta;

  MonitorElement* hHCAL_D1_MEyvsieta;
  MonitorElement* hHCAL_D2_MEyvsieta;
  MonitorElement* hHCAL_D3_MEyvsieta;
  MonitorElement* hHCAL_D4_MEyvsieta;
};

#endif
