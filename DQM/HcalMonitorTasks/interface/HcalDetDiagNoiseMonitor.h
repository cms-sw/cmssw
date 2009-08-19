#ifndef DQM_HCALMONITORTASKS_HCALDETDIAGNOISEMONITOR_H
#define DQM_HCALMONITORTASKS_HCALDETDIAGNOISEMONITOR_H

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
// this is to retrieve HCAL LogicalMap
#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

/** \class HcalDetDiagNoiseMonitor
  *  
  * $Date: 2009/07/01 06:09:21 $
  * $Revision: 1.1.2.6 $
  * \author D. Vishnevskiy
  */

class HcalDetDiagNoiseRMData{
public:
  HcalDetDiagNoiseRMData(){
    n_th_hi=n_th_lo=0;
    energy=0;
  };
  int    n_th_hi;
  int    n_th_lo;
  double energy; 
};

class HcalDetDiagNoiseRMSummary{
public:
  HcalDetDiagNoiseRMSummary(){ 
     std::string subdets[11]={"HBM","HBP","HEM","HEP","HO2M","HO1M","HO0","HO1P","HO2P","HFM","HFP"};
     reset(); 
     for(int sd=0;sd<11;sd++) for(int sect=1;sect<=18;sect++) for(int rm=1;rm<=4;rm++){
        std::stringstream tempss;
        tempss << std::setw(2) << std::setfill('0') << sect;
        std::string rbx= subdets[sd]+tempss.str();
        HcalFrontEndId id(rbx,rm,1,1,1,1,1);
        if(id.rawId()==0) continue;
        SubDetIndex[id.rmIndex()]=sd; 
     }
     for(int i=0;i<HcalFrontEndId::maxRmIndex;i++) Ref[i]=0;
  }
  void reset(int subdet=-1){
     if(subdet==-1){
       for(int i=0;i<HcalFrontEndId::maxRmIndex;i++) AboveThHi[i]=0; 
       for(int i=0;i<11;i++) events[i]=0;
     }else{
        std::string subdets[11]={"HBM","HBP","HEM","HEP","HO2M","HO1M","HO0","HO1P","HO2P","HFM","HFP"};
        for(int sect=1;sect<=18;sect++) for(int rm=1;rm<=4;rm++){
	   std::stringstream tempss;
           tempss << std::setw(2) << std::setfill('0') << sect;
           std::string rbx= subdets[subdet]+tempss.str();
           HcalFrontEndId id(rbx,rm,1,1,1,1,1);
           if(id.rawId()==0) continue;
           AboveThHi[id.rmIndex()]=0; 
	   events[subdet]=0;
	}
     }
  }
  void SetReference(int index,double val){
     if(index<0 || index>=HcalFrontEndId::maxRmIndex) return;
     Ref[index]=val;
  } 
  double GetReference(int index){
     if(index<0 || index>=HcalFrontEndId::maxRmIndex) return 0;
     return Ref[index];
  } 
  bool GetRMStatusValue(const std::string& rbx,int rm,double *val){
     int index=GetRMindex(rbx,rm);
     if(index<0 || index>=HcalFrontEndId::maxRmIndex) return false;
     if(events[SubDetIndex[index]]>10){ *val=(double)AboveThHi[index]/(double)events[SubDetIndex[index]]; return true; }
     *val=0; return true; 
  }
  void AddNoiseStat(int rm_index){
     AboveThHi[rm_index]++;
     events[SubDetIndex[rm_index]]++;
  }
  int GetSubDetIndex(const std::string& rbx){
      return SubDetIndex[GetRMindex(rbx,2)];
  }
  
  int GetRMindex(const std::string& rbx,int rm){
      if(rbx.substr(0,3)=="HO0"){
         int sect=atoi(rbx.substr(3,2).c_str());
         if(sect>12) return -1;
	 if(rm==1 && (sect==2  || sect==3 || sect==6 || sect==7 || sect==10 || sect==11)) return -1;
         if(rm==4 && (sect==12 || sect==1 || sect==4 || sect==5 || sect==8  || sect==9 )) return -1;
      }
      if(rbx.substr(0,3)=="HO1" || rbx.substr(0,3)=="HO2"){ 
         int sect=atoi(rbx.substr(4,2).c_str());
	 if(sect>12) return -1;
         if(sect==1 || sect==3 || sect==5 || sect==7 || sect==9 || sect==11) return -1;
      }
      HcalFrontEndId id(rbx,rm,1,1,1,1,1);
      if(id.rawId()==0) return -1;
      return id.rmIndex(); 
  }
  int GetStat(int subdet){ return events[subdet]; }
private:  
  int    AboveThHi  [HcalFrontEndId::maxRmIndex];
  int    SubDetIndex[HcalFrontEndId::maxRmIndex];
  double Ref[HcalFrontEndId::maxRmIndex];
  int    events[11];
};


class HcalDetDiagNoiseMonitor:public HcalBaseMonitor {
public:
  HcalDetDiagNoiseMonitor(); 
  ~HcalDetDiagNoiseMonitor(); 

  void setup(const edm::ParameterSet& ps, DQMStore* dbe);
  void processEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HcalDbService& cond);
  void done();
  void reset();
  void clearME(); 
  void UpdateHistos();
  int  GetStatistics(){ return ievt_; }
private:
  edm::InputTag inputLabelDigi_;
  edm::InputTag FEDRawDataCollection_;
  
  HcalLogicalMapGenerator gen;
  HcalLogicalMap          *lmap;
 
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
  
  HcalDetDiagNoiseRMSummary RBXSummary;
  HcalDetDiagNoiseRMSummary RBXCurrentSummary;
};

#endif
