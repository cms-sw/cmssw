#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGTIMINGMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGTIMINGMONITOR_H

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"


/** \class HcalDetDiagTimingMonitor
  *  
  * $Date: 2010/03/25 11:00:57 $
  * $Revision: 1.2 $
  * \author D. Vishnevskiy
  */

class HcalDetDiagTimingMonitor:public HcalBaseDQMonitor {
public:
  HcalDetDiagTimingMonitor(const edm::ParameterSet& ps); 
  ~HcalDetDiagTimingMonitor(); 

  double GetTime(double *data,int n){
        int MaxI=-100; double Time=0,SumT=0,MaxT=-10;
        for(int j=0;j<n;++j) if(MaxT<data[j]){ MaxT=data[j]; MaxI=j; }
	if(MaxI>=0){
           Time=MaxI*data[MaxI];
           SumT=data[MaxI];
	   if(MaxI>0){ Time+=(MaxI-1)*data[MaxI-1]; SumT+=data[MaxI-1]; }
	   if(MaxI<(n-1)){ Time+=(MaxI+1)*data[MaxI+1]; SumT+=data[MaxI+1]; }
	   Time=Time/SumT;
	}
        return Time;
  }
  bool isSignal(double *data,int n){
        int Imax=-1; double max=-100;
        for(int i=0;i<n;i++) if(data[i]>max){max=data[i]; Imax=i;}
        if(Imax==0 && Imax==(n-1)) return false;
        float sum=data[Imax-1]+data[Imax+1];
        if(data[Imax]>5.5 && sum>(data[Imax]*0.25)) return true;
        return false;
  }
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
  void set_hbhe(int eta,int phi,int depth,int cap,float val){
       HBHE[eta+50][phi][depth][cap]+=val;
       nHBHE[eta+50][phi][depth][cap]+=1.0;
  }   
  void set_ho(int eta,int phi,int depth,int cap,float val){
       HO[eta+50][phi][depth][cap]+=val;
       nHO[eta+50][phi][depth][cap]+=1.0;
  }   
  void set_hf(int eta,int phi,int depth,int cap,float val){
       HF[eta+50][phi][depth][cap]+=val;
       nHF[eta+50][phi][depth][cap]+=1.0;
  }
  double get_ped_hbhe(int eta,int phi,int depth,int cup){
      if(nHBHE[eta+50][phi][depth][cup]<10) return 2.5; 
      if(nHBHE[eta+50][phi][depth][cup]!=0){
         double ped=HBHE[eta+50][phi][depth][cup]/nHBHE[eta+50][phi][depth][cup];
         if(ped>1.5 && ped<4.5) return ped;
      } 
      return 9999; 
  }   
  double get_ped_ho(int eta,int phi,int depth,int cup){
      if(nHO[eta+50][phi][depth][cup]<10) return 2.5; 
      if(nHO[eta+50][phi][depth][cup]!=0){
         double ped=HO[eta+50][phi][depth][cup]/nHO[eta+50][phi][depth][cup];
         if(ped>1.5 && ped<4.5) return ped;
      }
      return 9999; 
  }   
  double get_ped_hf(int eta,int phi,int depth,int cup){
      if(nHF[eta+50][phi][depth][cup]<10) return 2.5; 
      if(nHF[eta+50][phi][depth][cup]!=0){
         double ped=HF[eta+50][phi][depth][cup]/nHF[eta+50][phi][depth][cup];
         if(ped>1.5 && ped<4.5) return ped;
      }
      return 9999; 
  }   
  double HBHE[100][73][5][4];
  double nHBHE[100][73][5][4];
  double HO[100][73][5][4];
  double nHO[100][73][5][4];   
  double HF[100][73][5][4];
  double nHF[100][73][5][4];
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
//   noise/crazy channels masking
  double occHBHE[100][73][5]; 
  double occHO  [100][73][5]; 
  double occHF  [100][73][5]; 
  double occSum;
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////  
  void setup();
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void done();
  void reset();
  void cleanup(); 
  
private:
  edm::InputTag inputLabelDigi_;
  edm::InputTag L1ADataLabel_;
  edm::InputTag FEDRawDataCollection_;
  
  int  GCTTriggerBit1_;
  int  GCTTriggerBit2_;
  int  GCTTriggerBit3_;
  int  GCTTriggerBit4_;
  int  GCTTriggerBit5_; 
  bool CosmicsCorr_; 
  
  MonitorElement *HBTimeDT; 
  MonitorElement *HBTimeRPC; 
  MonitorElement *HBTimeGCT; 
  MonitorElement *HBTimeHO; 
  MonitorElement *HOTimeDT; 
  MonitorElement *HOTimeRPC; 
  MonitorElement *HOTimeGCT; 
  MonitorElement *HOTimeHO; 
  MonitorElement *HETimeCSCp; 
  MonitorElement *HETimeCSCm;
  MonitorElement *HETimeRPCp; 
  MonitorElement *HETimeRPCm;
  MonitorElement *HFTimeCSCp; 
  MonitorElement *HFTimeCSCm;
  MonitorElement *Summary;  
  
  void CheckTiming();
};

#endif
