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
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
// to retrive trigger information (local runs only)
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"

// forward declarations
class HcalDetDiagNoiseRMSummary; 
class DQMStore;
class MonitorElement;
class HcalDbService;

// #########################################################################################

/** \class HcalDetDiagNoiseMonitor
  *  
  * \author D. Vishnevskiy
  */


class HcalDetDiagNoiseMonitor:public HcalBaseDQMonitor {
public:
  HcalDetDiagNoiseMonitor(const edm::ParameterSet& ps); 
  ~HcalDetDiagNoiseMonitor(); 

  void setup(DQMStore::IBooker &);
  void analyze(edm::Event const&e, edm::EventSetup const&s);
  void done();
  void reset();
  void bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c);
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c);

  void UpdateHistos();
  int  GetStatistics(){ return ievt_; }
private:
  edm::InputTag digiLabel_;

  edm::EDGetTokenT<FEDRawDataCollection> tok_raw_;
  edm::EDGetTokenT<HcalTBTriggerData> tok_tb_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> tok_l1_;
  edm::EDGetTokenT<HBHEDigiCollection> tok_hbhe_;
  edm::EDGetTokenT<HODigiCollection> tok_ho_;


  std::string OutputFilePath;
  bool Overwrite;
  int dataset_seq_number;
 
  bool UseDB;
  bool setupDone_;
  
  double  HPDthresholdHi;
  double  HPDthresholdLo;
  double  SpikeThreshold;
  
  void SaveRates();
  
  int         ievt_;
  bool        newLS;
  int         FirstOrbit;
  int         LastOrbit;
  int         FirstOrbitLS;
  int         LastOrbitLS;
  int         run_number;
  int         NoisyEvents;
  bool        LocalRun; 

  MonitorElement *meEVT_;
  
  MonitorElement *HBP_Rate50;
  MonitorElement *HBM_Rate50;
  MonitorElement *HEP_Rate50;
  MonitorElement *HEM_Rate50;
  MonitorElement *HBP_Rate300;
  MonitorElement *HBM_Rate300;
  MonitorElement *HEP_Rate300;
  MonitorElement *HEM_Rate300;

  MonitorElement *HO0_Rate50;
  MonitorElement *HO1P_Rate50;
  MonitorElement *HO1M_Rate50;
  MonitorElement *HO0_Rate300;
  MonitorElement *HO1P_Rate300;
  MonitorElement *HO1M_Rate300;

  MonitorElement *HB_RBXmapSpikeCnt;
  MonitorElement *HE_RBXmapSpikeCnt;
  MonitorElement *HO_RBXmapSpikeCnt;
  
  MonitorElement *PixelMult;
  MonitorElement *HPDEnergy;
  MonitorElement *RBXEnergy;
  MonitorElement *NZeroes;
  MonitorElement *TriggerBx11;
  MonitorElement *TriggerBx12;
  
  
  HcalDetDiagNoiseRMSummary* RMSummary;

};

#endif
