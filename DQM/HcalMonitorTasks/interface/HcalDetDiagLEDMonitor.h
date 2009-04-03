#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGLEDMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGLEDMONITOR_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <math.h>
using namespace edm;
using namespace std;
// this is to retrieve HCAL digi's
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
// to retrive trigger information (local runs only)
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
// to retrive GMT information, for cosmic runs muon triggers can be used as pedestal (global runs only)
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
// to retrive trigger desition words, to select pedestal (from hcal point of view) triggers (global runs only)
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "CondTools/Hcal/interface/HcalLogicalMapGenerator.h"
#include "CondTools/Hcal/interface/HcalLogicalMap.h"

/** \class HcalDetDiagLEDMonitor
  *  
  * $Date: 2009/01/21 15:01:37 $
  * $Revision: 1.20 $
  * \author D. Vishnevskiy
  */

class HcalDetDiagLEDData{
public: 
   HcalDetDiagLEDData(){ 
	     IsRefetence=false;
	     status=0;
	     reset();
	  }
   void   reset(){
             Xe=XXe=Xt=XXt=n=0;
	     overflow=0;
	     undeflow=0;
          }
   void   add_statistics(double *data,int nTS){
 	     double e=GetEnergy(data,nTS);
	     double t=GetTime(data,nTS);
             if(e<20) undeflow++; else if(e>10000) overflow++; else{
	        n++; Xe+=e; XXe+=e*e; Xt+=t; XXt+=t*t;
	     } 	   
	  }
   void   set_reference(float val,float rms){
             ref_led=val; ref_rms=rms;
	     IsRefetence=true;
          }	  
   void   change_status(int val){
             status|=val;
          }	  
   int    get_status(){
             return status;
          }	  
   bool   get_reference(double *val,double *rms){
             *val=ref_led; *rms=ref_rms;
	     return IsRefetence;
          }	  
   bool   get_average_led(double *ave,double *rms){
	     if(n>0){ *ave=Xe/n; *rms=sqrt(XXe/n-(Xe*Xe)/(n*n));} else return false;
             return true; 
          }
   bool   get_average_time(double *ave,double *rms){
             if(n>0){ *ave=Xt/n; *rms=sqrt(XXt/n-(Xt*Xt)/(n*n));} else return false;
             return true; 
          }
   int    get_statistics(){
	     return (int)n;
	  } 
   int    get_overflow(){
             return overflow;
          }   
   int    get_undeflow(){
             return undeflow;
          }   
private:   
   double GetEnergy(double *data,int n){
             int MaxI=0; double Energy,MaxE=0;
             for(int j=0;j<n;++j) if(MaxE<data[j]){ MaxE=data[j]; MaxI=j; }
             Energy=data[MaxI];
             if(MaxI>0) Energy+=data[MaxI-1];
             if(MaxI>1) Energy+=data[MaxI-2];
             if(MaxI<(n-1)) Energy+=data[MaxI+1];
             if(MaxI<(n-2)) Energy+=data[MaxI+2];
             return Energy;
          }
   double GetTime(double *data,int n=10){
             int MaxI=-100; double Time,SumT=0,MaxT=-10;
             for(int j=0;j<n;++j) if(MaxT<data[j]){ MaxT=data[j]; MaxI=j; }
             Time=MaxI*data[MaxI];
             SumT=data[MaxI];
             if(MaxI>0){ Time+=(MaxI-1)*data[MaxI-1]; SumT+=data[MaxI-1]; }
             if(MaxI<(n-1)){ Time+=(MaxI+1)*data[MaxI+1]; SumT+=data[MaxI+1]; }
	     Time=Time/SumT;
             return Time;
         }      
   int   overflow;
   int   undeflow;
   double Xe,XXe,Xt,XXt,n;
   bool  IsRefetence;
   float ref_led;
   float ref_rms;
   int   status;
};

class HcalDetDiagLEDMonitor:public HcalBaseMonitor {
public:
  HcalDetDiagLEDMonitor(); 
  ~HcalDetDiagLEDMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HcalDbService& cond);
  void done();
  void reset();
  void clearME(); 
  void fillHistos();
  int  GetStatistics(){ return ievt_; }
private:
  const HcalElectronicsMap  *emap;
  // in principle functions below shold use DB interface (will be modefied when DB will be ready...)  
  void SaveReference();
  void LoadReference();
  void CheckStatus();
  
  HcalDetDiagLEDData *GetCalib(char *sd,int eta,int phi){
          int SD=0,ETA=0,PHI=0;
          if(strcmp(sd,"HB")==0)SD=1; 
          if(strcmp(sd,"HE")==0)SD=2; 
          if(strcmp(sd,"HO")==0)SD=3; 
          if(strcmp(sd,"HF")==0)SD=4; 
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
             if(eta>0) ETA=1; else eta=-1;
	     if(phi>=1  && phi<=18) PHI=1;
	     if(phi>=19 && phi<=36) PHI=19;
	     if(phi>=37 && phi<=54) PHI=37;
	     if(phi>=55 && phi<=72) PHI=55;
          }
          return &calib_data[SD][ETA+2][PHI-1];
       };
  int         ievt_;
  int         run_number;
  int         dataset_seq_number;
  bool        IsReference;
  
  double      LEDMeanTreshold;
  double      LEDRmsTreshold;
  bool        UseDB;
   
  std::string ReferenceData;
  std::string OutputFilePath;

  MonitorElement *meEVT_;
  MonitorElement *Energy;
  MonitorElement *Time;
  MonitorElement *Time2Dhbhehf;
  MonitorElement *Time2Dho;
  MonitorElement *Energy2Dhbhehf;
  MonitorElement *Energy2Dho;
  MonitorElement *EnergyRMS;
  MonitorElement *TimeRMS;
  MonitorElement *EnergyCorr;
  
  HcalDetDiagLEDData hb_data[85][72][4];
  HcalDetDiagLEDData he_data[85][72][4];
  HcalDetDiagLEDData ho_data[85][72][4];
  HcalDetDiagLEDData hf_data[85][72][4];
  HcalDetDiagLEDData calib_data[5][5][72];
  
  std::vector<MonitorElement*> ChannelsLEDEnergy;
  std::vector<MonitorElement*> ChannelsLEDEnergyRef;

    
  std::vector<MonitorElement*> ChannelStatusMissingChannels;
  std::vector<MonitorElement*> ChannelStatusUnstableChannels;
  std::vector<MonitorElement*> ChannelStatusUnstableLEDsignal;
  std::vector<MonitorElement*> ChannelStatusLEDMean;
  std::vector<MonitorElement*> ChannelStatusLEDRMS;
  std::vector<MonitorElement*> ChannelStatusTimeMean;
  std::vector<MonitorElement*> ChannelStatusTimeRMS;
 
  void fill_channel_status(char *subdet,int eta,int phi,int depth,int type,double status);
  void fill_energy(char *subdet,int eta,int phi,int depth,double e,int type);
};

#endif
