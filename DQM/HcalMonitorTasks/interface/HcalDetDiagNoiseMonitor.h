#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGNOISEMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGNOISEMONITOR_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// forward declarations
class HcalDetDiagNoiseRMSummary; 
class DQMStore;
class MonitorElement;
class HcalDbService;
class HcalLogicalMap;
class HcalLogicalMapGenerator;

// #########################################################################################

/** \class HcalDetDiagNoiseMonitor
  *  
  * $Date: 2010/03/11 08:15:08 $
  * $Revision: 1.4.2.4 $
  * \author D. Vishnevskiy
  */


class HcalDetDiagNoiseMonitor:public HcalBaseDQMonitor {
public:
  HcalDetDiagNoiseMonitor(const edm::ParameterSet& ps); 
  ~HcalDetDiagNoiseMonitor(); 

  void setup();
  void analyze(edm::Event const&e, edm::EventSetup const&s);
  void done();
  void reset();
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void cleanup(); 
  void UpdateHistos();
  int  GetStatistics(){ return ievt_; }
private:
  edm::InputTag digiLabel_;
  edm::InputTag rawDataLabel_;
  
  HcalLogicalMap          *lmap;
  HcalLogicalMapGenerator *gen;
  const HcalDbService* cond_; // cond isn't used for anything!  Remove it?
 
  std::string ReferenceData;
  std::string ReferenceRun;
  std::string OutputFilePath;
  bool IsReference;
  bool UseDB;
  
  double  HPDthresholdHi;
  double  HPDthresholdLo;
  double  SiPMthreshold;
  double  SpikeThreshold;
  int     UpdateEvents;
  
  void SaveReference();
  void LoadReference();
  
  int         ievt_;
  int         run_number;
  int         NoisyEvents;
  MonitorElement *meEVT_;
  MonitorElement *HB_RBXmapRatio;
  MonitorElement *HB_RBXmapRatioCur;
  MonitorElement *HB_RBXmapSpikeCnt;
  MonitorElement *HB_RBXmapSpikeAmp;
  MonitorElement *HE_RBXmapRatio;
  MonitorElement *HE_RBXmapRatioCur;
  MonitorElement *HE_RBXmapSpikeCnt;
  MonitorElement *HE_RBXmapSpikeAmp;
  MonitorElement *HO_RBXmapRatio;
  MonitorElement *HO_RBXmapRatioCur;
  MonitorElement *HO_RBXmapSpikeCnt;
  MonitorElement *HO_RBXmapSpikeAmp;
  
  MonitorElement *PixelMult;
  MonitorElement *HPDEnergy;
  MonitorElement *RBXEnergy;
  
  HcalDetDiagNoiseRMSummary* RBXSummary;
  HcalDetDiagNoiseRMSummary* RBXCurrentSummary;

// #########################################################################################

      MonitorElement *Met;
      MonitorElement *Mephi;
      MonitorElement *Mex;
      MonitorElement *SumEt;
      MonitorElement *HaEtHB;
      MonitorElement *HaEtHE;
      MonitorElement *HaEtHF;
      MonitorElement *EmEtHF;
      MonitorElement *Met_PhysicsCategory;
      MonitorElement *Mephi_PhysicsCategory;
      MonitorElement *Mex_PhysicsCategory;
      MonitorElement *SumEt_PhysicsCategory;
      MonitorElement *HaEtHB_PhysicsCategory;
      MonitorElement *HaEtHE_PhysicsCategory;
      MonitorElement *HaEtHF_PhysicsCategory;
      MonitorElement *EmEtHF_PhysicsCategory;
      MonitorElement *HCALFraction;
      MonitorElement *chargeFraction;
      MonitorElement *HCALFractionVSchargeFraction;
      MonitorElement *JetEt;
      MonitorElement *JetEta;
      MonitorElement *JetPhi;
      MonitorElement *HCALFraction_PhysicsCategory;
      MonitorElement *chargeFraction_PhysicsCategory;
      MonitorElement *HCALFractionVSchargeFraction_PhysicsCategory;
      MonitorElement *JetEt_PhysicsCategory;
      MonitorElement *JetEta_PhysicsCategory;
      MonitorElement *JetPhi_PhysicsCategory;
      MonitorElement *JetEt_TaggedAnomalous;
      MonitorElement *JetEta_TaggedAnomalous;
      MonitorElement *JetPhi_TaggedAnomalous;
      MonitorElement *JetEt_TaggedAnomalous_PhysicsCategory;
      MonitorElement *JetEta_TaggedAnomalous_PhysicsCategory;
      MonitorElement *JetPhi_TaggedAnomalous_PhysicsCategory;
      MonitorElement *HFtowerRatio;
      MonitorElement *HFtowerPt;
      MonitorElement *HFtowerEta;
      MonitorElement *HFtowerPhi;
      MonitorElement *HFtowerRatio_PhysicsCategory;
      MonitorElement *HFtowerPt_PhysicsCategory;
      MonitorElement *HFtowerEta_PhysicsCategory;
      MonitorElement *HFtowerPhi_PhysicsCategory;
      MonitorElement *HFtowerPt_TaggedAnomalous;
      MonitorElement *HFtowerEta_TaggedAnomalous;
      MonitorElement *HFtowerPhi_TaggedAnomalous;
      MonitorElement *HFtowerPt_TaggedAnomalous_PhysicsCategory;
      MonitorElement *HFtowerEta_TaggedAnomalous_PhysicsCategory;
      MonitorElement *HFtowerPhi_TaggedAnomalous_PhysicsCategory;
      MonitorElement *RBXMaxZeros;
      MonitorElement *RBXHitsHighest;
      MonitorElement *RBXE2tsOverE10ts;
      MonitorElement *HPDHitsHighest;
      MonitorElement *HPDE2tsOverE10ts;
      MonitorElement *RBXMaxZeros_PhysicsCategory;
      MonitorElement *RBXHitsHighest_PhysicsCategory;
      MonitorElement *RBXE2tsOverE10ts_PhysicsCategory;
      MonitorElement *HPDHitsHighest_PhysicsCategory;
      MonitorElement *HPDE2tsOverE10ts_PhysicsCategory;
      MonitorElement *Met_TaggedHBHEAnomalous;
      MonitorElement *Mephi_TaggedHBHEAnomalous;
      MonitorElement *Mex_TaggedHBHEAnomalous;
      MonitorElement *SumEt_TaggedHBHEAnomalous;
      MonitorElement *HaEtHB_TaggedHBHEAnomalous;
      MonitorElement *HaEtHE_TaggedHBHEAnomalous;
      MonitorElement *HaEtHF_TaggedHBHEAnomalous;
      MonitorElement *EmEtHF_TaggedHBHEAnomalous;
      MonitorElement *RBXMaxZeros_TaggedHBHEAnomalous;
      MonitorElement *RBXHitsHighest_TaggedHBHEAnomalous;
      MonitorElement *RBXE2tsOverE10ts_TaggedHBHEAnomalous;
      MonitorElement *HPDHitsHighest_TaggedHBHEAnomalous;
      MonitorElement *HPDE2tsOverE10ts_TaggedHBHEAnomalous;
      MonitorElement *Met_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *Mephi_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *Mex_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *SumEt_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *HaEtHB_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *HaEtHE_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *HaEtHF_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *EmEtHF_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *RBXMaxZeros_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *RBXHitsHighest_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *RBXE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *HPDHitsHighest_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *HPDE2tsOverE10ts_TaggedHBHEAnomalous_PhysicsCategory;
      MonitorElement *Met_TaggedHFAnomalous;
      MonitorElement *Mephi_TaggedHFAnomalous;
      MonitorElement *Mex_TaggedHFAnomalous;
      MonitorElement *SumEt_TaggedHFAnomalous;
      MonitorElement *HaEtHB_TaggedHFAnomalous;
      MonitorElement *HaEtHE_TaggedHFAnomalous;
      MonitorElement *HaEtHF_TaggedHFAnomalous;
      MonitorElement *EmEtHF_TaggedHFAnomalous;
      MonitorElement *Met_TaggedHFAnomalous_PhysicsCategory;
      MonitorElement *Mephi_TaggedHFAnomalous_PhysicsCategory;
      MonitorElement *Mex_TaggedHFAnomalous_PhysicsCategory;
      MonitorElement *SumEt_TaggedHFAnomalous_PhysicsCategory;
      MonitorElement *HaEtHB_TaggedHFAnomalous_PhysicsCategory;
      MonitorElement *HaEtHE_TaggedHFAnomalous_PhysicsCategory;
      MonitorElement *HaEtHF_TaggedHFAnomalous_PhysicsCategory;
      MonitorElement *EmEtHF_TaggedHFAnomalous_PhysicsCategory;
      MonitorElement *Met_TaggedHBHEHFAnomalous;
      MonitorElement *Mephi_TaggedHBHEHFAnomalous;
      MonitorElement *Mex_TaggedHBHEHFAnomalous;
      MonitorElement *SumEt_TaggedHBHEHFAnomalous;
      MonitorElement *HaEtHB_TaggedHBHEHFAnomalous;
      MonitorElement *HaEtHE_TaggedHBHEHFAnomalous;
      MonitorElement *HaEtHF_TaggedHBHEHFAnomalous;
      MonitorElement *EmEtHF_TaggedHBHEHFAnomalous;
      MonitorElement *Met_TaggedHBHEHFAnomalous_PhysicsCategory;
      MonitorElement *Mephi_TaggedHBHEHFAnomalous_PhysicsCategory;
      MonitorElement *Mex_TaggedHBHEHFAnomalous_PhysicsCategory;
      MonitorElement *SumEt_TaggedHBHEHFAnomalous_PhysicsCategory;
      MonitorElement *HaEtHB_TaggedHBHEHFAnomalous_PhysicsCategory;
      MonitorElement *HaEtHE_TaggedHBHEHFAnomalous_PhysicsCategory;
      MonitorElement *HaEtHF_TaggedHBHEHFAnomalous_PhysicsCategory;
      MonitorElement *EmEtHF_TaggedHBHEHFAnomalous_PhysicsCategory;
      MonitorElement *NLumiSections;

  edm::InputTag hlTriggerResults_;
  edm::InputTag MetSource_;
  edm::InputTag JetSource_;
  edm::InputTag TrackSource_;
  edm::InputTag VertexSource_;
  std::string rbxCollName_;
  std::string PhysDeclaredRequirement_;
  std::string MonitoringTriggerRequirement_;
  bool UseMonitoringTrigger_;
  bool UseVertexCuts_;
  double JetMinEt_;
  double JetMaxEta_;
  double ConstituentsToJetMatchingDeltaR_;
  double TrackMaxIp_;
  double TrackMinThreshold_;
  double MinJetChargeFraction_;
  double MaxJetHadronicEnergyFraction_;
  edm::InputTag caloTowerCollName_;

  std::vector<unsigned int> lumi;
  std::vector<reco::CaloJet> HcalNoisyJetContainer;

  int numRBXhits;
  double rbxenergy;
  double hpdEnergyHighest;
  double nHitsHighest;
  double totale2ts;
  double totale10ts;
  int numHPDhits;
  double e2ts;
  double e10ts;
  bool isRBXNoisy;
  bool isHPDNoisy;

// #########################################################################################

};

#endif
