#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGLEDMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGLEDMONITOR_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

/** \class HcalDetDiagLEDMonitor
  *  
  * $Date: 2010/03/25 11:00:57 $
  * $Revision: 1.6 $
  * \author D. Vishnevskiy
  */


class HcalDetDiagLEDData;
class HcalLogicalMapGenerator;
class HcalLogicalMap;

class HcalDetDiagLEDMonitor:public HcalBaseDQMonitor {
public:
  HcalDetDiagLEDMonitor(const edm::ParameterSet& ps); 
  ~HcalDetDiagLEDMonitor(); 

  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void setup();
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);// const HcalDbService& cond)
  void done();
  void reset();
  void cleanup(); 
  void fillHistos();
  int  GetStatistics(){ return ievt_; }
private:
  HcalLogicalMapGenerator *gen;
  HcalElectronicsMap      emap;
  HcalLogicalMap          *lmap;
  // in principle functions below shold use DB interface (will be modefied when DB will be ready...)  
  void SaveReference();
  void LoadReference();
  void CheckStatus();
  
  HcalDetDiagLEDData* GetCalib(std::string sd,int eta,int phi){
    int SD=0,ETA=0,PHI=0;
    if(sd.compare("HB")==0) SD=1; 
    if(sd.compare("HE")==0) SD=2; 
    if(sd.compare("HO")==0) SD=3; 
    if(sd.compare("HF")==0) SD=4; 
    if(SD==1 || SD==2){
      if(eta>0) ETA=1; else ETA=-1;
      if(phi==71 ||phi==72 || phi==1 || phi==2) PHI=71; else PHI=((phi-3)/4)*4+3;
    }else if(SD==3){
      if(abs(eta)<=4){
	ETA=0;
	if(phi==71 ||phi==72 || phi==1 || phi==2 || phi==3 || phi==4) PHI=71; else PHI=((phi-5)/6)*6+5;
      }else{
	if(abs(eta)>4  && abs(eta)<=10)  ETA=1;
	if(abs(eta)>10 && abs(eta)<=15)  ETA=2;
	if(eta<0) ETA=-ETA;
	if(phi==71 ||phi==72 || (phi>=1 && phi<=10)) PHI=71; else PHI=((phi-11)/12)*12+11;
      }
    }else if(SD==4){
      if(eta>0) ETA=1; else ETA=-1;
      if(phi>=1  && phi<=18) PHI=1;
      if(phi>=19 && phi<=36) PHI=19;
      if(phi>=37 && phi<=54) PHI=37;
      if(phi>=55 && phi<=72) PHI=55;
    }
    return calib_data[SD][ETA+2][PHI-1];
  };
  int         ievt_;
  int         run_number;
  int         dataset_seq_number;
  bool        IsReference;
  
  double      LEDMeanTreshold;
  double      LEDRmsTreshold;
  bool        UseDB;
   
  std::string ReferenceData;
  std::string ReferenceRun;
  std::string OutputFilePath;

  MonitorElement *meEVT_;
  MonitorElement *RefRun_;
  MonitorElement *Energy;
  MonitorElement *Time;
  MonitorElement *EnergyHF;
  MonitorElement *TimeHF;
  MonitorElement *Time2Dhbhehf;
  MonitorElement *Time2Dho;
  MonitorElement *Energy2Dhbhehf;
  MonitorElement *Energy2Dho;
  MonitorElement *EnergyRMS;
  MonitorElement *TimeRMS;
  MonitorElement *EnergyRMSHF;
  MonitorElement *TimeRMSHF;
  MonitorElement *EnergyCorr;
  MonitorElement *HBPphi;
  MonitorElement *HBMphi;
  MonitorElement *HEPphi;
  MonitorElement *HEMphi;
  MonitorElement *HFPphi;
  MonitorElement *HFMphi;
  MonitorElement *HO0phi;
  MonitorElement *HO1Pphi;
  MonitorElement *HO2Pphi;
  MonitorElement *HO1Mphi;
  MonitorElement *HO2Mphi;

  HcalDetDiagLEDData* hb_data[85][72][4];
  HcalDetDiagLEDData* he_data[85][72][4];
  HcalDetDiagLEDData* ho_data[85][72][4];
  HcalDetDiagLEDData* hf_data[85][72][4];
  HcalDetDiagLEDData* calib_data[5][5][72];
  
  EtaPhiHists ChannelsLEDEnergy;
  EtaPhiHists ChannelsLEDEnergyRef;
  EtaPhiHists ChannelStatusMissingChannels;
  EtaPhiHists ChannelStatusUnstableChannels;
  EtaPhiHists ChannelStatusUnstableLEDsignal;
  EtaPhiHists ChannelStatusLEDMean;
  EtaPhiHists ChannelStatusLEDRMS;
  EtaPhiHists ChannelStatusTimeMean;
  EtaPhiHists ChannelStatusTimeRMS;
 
  edm::InputTag digiLabel_;
  edm::InputTag triggerLabel_;
  edm::InputTag calibDigiLabel_;

  void fill_channel_status(std::string subdet,int eta,int phi,int depth,int type,double status);
  void   fill_energy(std::string subdet,int eta,int phi,int depth,double e,int type);
  double get_energy(std::string subdet,int eta,int phi,int depth,int type);
};

#endif
