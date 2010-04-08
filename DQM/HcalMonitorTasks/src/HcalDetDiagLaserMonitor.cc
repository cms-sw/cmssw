// -*- C++ -*-
//
// Package:    HcalDetDiagLaserMonitor
// Class:      HcalDetDiagLaserMonitor
// 
/**\class HcalDetDiagLaserMonitor HcalDetDiagLaserMonitor.cc DQM/HcalDetDiagLaserMonitor/src/HcalDetDiagLaserMonitor.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Dmitry Vishnevskiy,591 R-013,+41227674265,
//         Created:  Wed Mar  3 12:14:16 CET 2010
// $Id: HcalDetDiagLaserMonitor.cc,v 1.13 2010/04/08 10:59:20 dma Exp $
//
//

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
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

// this is to retrieve HCAL digi's
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
// to retrive trigger information (local runs only)
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
 
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

#include <iostream>
#include <fstream>

////////////////////////////////////////////////////////////////////////////////////////////
static const float adc2fC[128]={-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5, 10.5,11.5,12.5,
                   13.5,15.,17.,19.,21.,23.,25.,27.,29.5,32.5,35.5,38.5,42.,46.,50.,54.5,59.5,
		   64.5,59.5,64.5,69.5,74.5,79.5,84.5,89.5,94.5,99.5,104.5,109.5,114.5,119.5,
		   124.5,129.5,137.,147.,157.,167.,177.,187.,197.,209.5,224.5,239.5,254.5,272.,
		   292.,312.,334.5,359.5,384.5,359.5,384.5,409.5,434.5,459.5,484.5,509.5,534.5,
		   559.5,584.5,609.5,634.5,659.5,684.5,709.5,747.,797.,847.,897.,947.,997.,
		   1047.,1109.5,1184.5,1259.5,1334.5,1422.,1522.,1622.,1734.5,1859.5,1984.5,
		   1859.5,1984.5,2109.5,2234.5,2359.5,2484.5,2609.5,2734.5,2859.5,2984.5,
		   3109.5,3234.5,3359.5,3484.5,3609.5,3797.,4047.,4297.,4547.,4797.,5047.,
		   5297.,5609.5,5984.5,6359.5,6734.5,7172.,7672.,8172.,8734.5,9359.5,9984.5};
////////////////////////////////////////////////////////////////////////////////////////////
typedef struct{
int eta;
int phi;
}Raddam_ch;
Raddam_ch RADDAM_CH[56]={{-30,15},{-32,15},{-34,15},{-36,15},{-38,15},{-40,15},{-41,15},
                         {-30,35},{-32,35},{-34,35},{-36,35},{-38,35},{-40,35},{-41,35},
                         {-30,51},{-32,51},{-34,51},{-36,51},{-38,51},{-40,51},{-41,51},
                         {-30,71},{-32,71},{-34,71},{-36,71},{-38,71},{-40,71},{-41,71},
                         {30, 01},{32, 01},{34, 01},{36, 01},{38, 01},{40, 71},{41, 71},
                         {30, 21},{32, 21},{34, 21},{36, 21},{38, 21},{40, 19},{41, 19},
                         {30, 37},{32, 37},{34, 37},{36, 37},{38, 37},{40, 35},{41, 35},
                         {30, 57},{32, 57},{34, 57},{36, 57},{38, 57},{40, 55},{41, 55}};

class HcalDetDiagLaserData{
public: 
   HcalDetDiagLaserData(){ 
	     IsRefetence=false;
             nChecks=0;nBadTime=0;nBadEnergy=0;
	     status=0;
	     reset();
	     reset1();
	  }
   void   reset(){
             Xe=XXe=Xt=XXt=n=0;
	     overflow=0;
	     undeflow=0;
          }
   void   reset1(){
             Xe1=XXe1=Xt1=XXt1=n1=0;
	     overflow1=0;
	     undeflow1=0;
          }
   void   add_statistics(double *data,int nTS){
 	     double e=GetEnergy(data,nTS);
	     double t=GetTime(data,nTS);
             if(e<20){ undeflow++;undeflow1++; }else if(e>10000){ overflow++;overflow1++; }else{
	        n++; Xe+=e; XXe+=e*e; Xt+=t; XXt+=t*t;
	        n1++; Xe1+=e; XXe1+=e*e; Xt1+=t; XXt1+=t*t;
	     } 	   
	  }
   void   set_reference(float val,float rms,float time,float time_rms){
             ref_amp=val; ref_rms=rms;
	     ref_time=time; ref_time_rms=time_rms;
	     IsRefetence=true;
          }	  
   void   change_status(int val){
             status|=val;
          }	  
   int    get_status(){
             return status;
          }	  
   bool   get_reference(double *val,double *rms,double *time,double *time_rms){
             *val=ref_amp; *rms=ref_rms;
             *time=ref_time; *time_rms=ref_time_rms;
	     return IsRefetence;
          }	  
   bool   get_average_amp(double *ave,double *rms){
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
   bool   get_average_amp1(double *ave,double *rms){
	     if(n1>0){ *ave=Xe1/n1; *rms=sqrt(XXe1/n1-(Xe1*Xe1)/(n1*n1));} else return false;
             return true; 
          }
   bool   get_average_time1(double *ave,double *rms){
             if(n1>0){ *ave=Xt1/n1; *rms=sqrt(XXt1/n1-(Xt1*Xt1)/(n1*n1));} else return false;
             return true; 
          }
   int    get_statistics1(){
	     return (int)n1;
	  } 
   int    get_overflow1(){
             return overflow1;
          }   
   int    get_undeflow1(){
             return undeflow1;
          }
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
             int MaxI=-100; double Time=-9999,SumT=0,MaxT=-10;
             for(int j=0;j<n;++j) if(MaxT<data[j]){ MaxT=data[j]; MaxI=j; }
	     if (MaxI>=0) // dummy protection so that compiler doesn't think MaxI=-100
	       {
		 Time=MaxI*data[MaxI];
		 SumT=data[MaxI];
		 if(MaxI>0){ Time+=(MaxI-1)*data[MaxI-1]; SumT+=data[MaxI-1]; }
		 if(MaxI<(n-1)){ Time+=(MaxI+1)*data[MaxI+1]; SumT+=data[MaxI+1]; }
		 Time=Time/SumT;
	       }
             return Time;
         }
   int    overflow;
   int    undeflow;
   int    overflow1;
   int    undeflow1;
   double Xe,XXe,Xt,XXt,n;
   double Xe1,XXe1,Xt1,XXt1,n1;
   bool   IsRefetence;
   float  ref_amp;
   float  ref_rms;
   float  ref_time;
   float  ref_time_rms;
   int    status;
   float    nChecks,nBadTime,nBadEnergy;
};

class HcalDetDiagLaserMonitor : public HcalBaseDQMonitor {
   public:
      explicit HcalDetDiagLaserMonitor(const edm::ParameterSet&);
      ~HcalDetDiagLaserMonitor();

   private:
      void beginRun(const edm::Run& run, const edm::EventSetup& c);  
      void endRun(const edm::Run& run, const edm::EventSetup& c);
      void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c) ;
      void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c);
      void analyze(const edm::Event&, const edm::EventSetup&);

      const HcalElectronicsMap  *emap;
      edm::InputTag inputLabelDigi_;

      void SaveReference();
      void LoadReference();
      bool get_ave_rbx(int sd,int side,int rbx,float *ave,float *rms);
      bool get_ave_subdet(int sd,float *ave_t,float *ave_e,float *ave_t_r,float *ave_e_r);
      void fillHistos(int sd);
      void fillProblems(int sd);
      int  nHBHEchecks,nHOchecks,nHFchecks;
      double LaserTimingThreshold,LaserEnergyThreshold;

      int         ievt_;
      int         run_number;
      int         dataset_seq_number;
      bool        IsReference;
      bool        LocalRun;

      std::string ReferenceData;
      std::string ReferenceRun;
      std::string OutputFilePath;
      std::string XmlFilePath;
      std::string baseFolder_;
      std::string prefixME_;
      bool        Online_;
      bool        Overwrite;

      MonitorElement *meEVT_,*meRUN_;
      MonitorElement *RefRun_;
      MonitorElement *hbheEnergy;
      MonitorElement *hbheTime;
      MonitorElement *hbheEnergyRMS;
      MonitorElement *hbheTimeRMS;
      MonitorElement *hoEnergy;
      MonitorElement *hoTime;
      MonitorElement *hoEnergyRMS;
      MonitorElement *hoTimeRMS;
      MonitorElement *hfEnergy;
      MonitorElement *hfTime;
      MonitorElement *hfEnergyRMS;
      MonitorElement *hfTimeRMS;

      MonitorElement *Time2Dhbhehf;
      MonitorElement *Time2Dho;
      MonitorElement *Energy2Dhbhehf;
      MonitorElement *Energy2Dho;
      MonitorElement *refTime2Dhbhehf;
      MonitorElement *refTime2Dho;
      MonitorElement *refEnergy2Dhbhehf;
      MonitorElement *refEnergy2Dho;

      MonitorElement *hb_time_rbx;
      MonitorElement *he_time_rbx;
      MonitorElement *ho_time_rbx;
      MonitorElement *hf_time_rbx;

      MonitorElement *Raddam[56];
      
      EtaPhiHists* ProblemCellsByDepth_timing;
      EtaPhiHists* ProblemCellsByDepth_energy;
      std::vector<std::string> problemnames_;

      EtaPhiHists* ProblemCellsByDepth_timing_val;
      EtaPhiHists* ProblemCellsByDepth_energy_val;

      HcalDetDiagLaserData hb_data[85][72][4];
      HcalDetDiagLaserData he_data[85][72][4];
      HcalDetDiagLaserData ho_data[85][72][4];
      HcalDetDiagLaserData hf_data[85][72][4];
};

HcalDetDiagLaserMonitor::HcalDetDiagLaserMonitor(const edm::ParameterSet& iConfig){
  ievt_=-1;
  emap=0;
  dataset_seq_number=1;
  run_number=-1;
  IsReference=false;
  LocalRun=false;
  nHBHEchecks=nHOchecks=nHFchecks=0;

  inputLabelDigi_  = iConfig.getUntrackedParameter<edm::InputTag>("digiLabel");
  ReferenceData    = iConfig.getUntrackedParameter<std::string>("LaserReferenceData" ,"");
  OutputFilePath   = iConfig.getUntrackedParameter<std::string>("OutputFilePath", "");
  XmlFilePath      = iConfig.getUntrackedParameter<std::string>("XmlFilePath", "");
  Online_          = iConfig.getUntrackedParameter<bool>  ("online",false);
  Overwrite        = iConfig.getUntrackedParameter<bool>  ("Overwrite",true);
  prefixME_        = iConfig.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.size()>0 && prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_          = iConfig.getUntrackedParameter<std::string>("TaskFolder","DetDiagPedestalMonitor_Hcal/");
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  debug_           = iConfig.getUntrackedParameter<int>("debug",0);

  LaserTimingThreshold = iConfig.getUntrackedParameter<double>("LaserTimingThreshold",0.2);
  LaserEnergyThreshold = iConfig.getUntrackedParameter<double>("LaserEnergyThreshold",0.1);
}
void HcalDetDiagLaserMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c){

  edm::ESHandle<HcalDbService> conditions_;
  c.get<HcalDbRecord>().get(conditions_);
  emap=conditions_->getHcalMapping();
  
  HcalBaseDQMonitor::setup();
  if (!dbe_) return;
    std::string name;
 
  dbe_->setCurrentFolder(subdir_);   
  meEVT_ = dbe_->bookInt("HcalDetDiagLaserMonitor Event Number");
  meRUN_ = dbe_->bookInt("HcalDetDiagLaserMonitor Run Number");

  ProblemCellsByDepth_timing = new EtaPhiHists();
  ProblemCellsByDepth_timing->setup(dbe_," Problem Bad Laser Timing");
  for(unsigned int i=0;i<ProblemCellsByDepth_timing->depth.size();i++)
          problemnames_.push_back(ProblemCellsByDepth_timing->depth[i]->getName());
  ProblemCellsByDepth_energy = new EtaPhiHists();
  ProblemCellsByDepth_energy->setup(dbe_," Problem Bad Laser Energy");
  for(unsigned int i=0;i<ProblemCellsByDepth_energy->depth.size();i++)
          problemnames_.push_back(ProblemCellsByDepth_energy->depth[i]->getName());

  dbe_->setCurrentFolder(subdir_+"Summary Plots");
     
  name="HBHE Laser Energy Distribution";                hbheEnergy        = dbe_->book1D(name,name,200,0,3000);
  name="HBHE Laser Timing Distribution";                hbheTime          = dbe_->book1D(name,name,200,0,10);
  name="HBHE Laser Energy RMS_div_Energy Distribution"; hbheEnergyRMS     = dbe_->book1D(name,name,200,0,0.5);
  name="HBHE Laser Timing RMS Distribution";            hbheTimeRMS       = dbe_->book1D(name,name,200,0,1);
  name="HO Laser Energy Distribution";                  hoEnergy          = dbe_->book1D(name,name,200,0,3000);
  name="HO Laser Timing Distribution";                  hoTime            = dbe_->book1D(name,name,200,0,10);
  name="HO Laser Energy RMS_div_Energy Distribution";   hoEnergyRMS       = dbe_->book1D(name,name,200,0,0.5);
  name="HO Laser Timing RMS Distribution";              hoTimeRMS         = dbe_->book1D(name,name,200,0,1);
  name="HF Laser Energy Distribution";                  hfEnergy          = dbe_->book1D(name,name,200,0,3000);
  name="HF Laser Timing Distribution";                  hfTime            = dbe_->book1D(name,name,200,0,10);
  name="HF Laser Energy RMS_div_Energy Distribution";   hfEnergyRMS       = dbe_->book1D(name,name,200,0,0.7);
  name="HF Laser Timing RMS Distribution";              hfTimeRMS         = dbe_->book1D(name,name,200,0,1);
     
  name="Laser Timing HBHEHF";                           Time2Dhbhehf      = dbe_->book2D(name,name,87,-43,43,74,0,73);
  name="Laser Timing HO";                               Time2Dho          = dbe_->book2D(name,name,33,-16,16,74,0,73);
  name="Laser Energy HBHEHF";                           Energy2Dhbhehf    = dbe_->book2D(name,name,87,-43,43,74,0,73);
  name="Laser Energy HO";                               Energy2Dho        = dbe_->book2D(name,name,33,-16,16,74,0,73);
  name="HBHEHF Laser (Timing-Ref)+1";                   refTime2Dhbhehf   = dbe_->book2D(name,name,87,-43,43,74,0,73);
  name="HO Laser (Timing-Ref)+1";                       refTime2Dho       = dbe_->book2D(name,name,33,-16,16,74,0,73);
  name="HBHEHF Laser Energy_div_Ref";                   refEnergy2Dhbhehf = dbe_->book2D(name,name,87,-43,43,74,0,73);
  name="HO Laser Energy_div_Ref";                       refEnergy2Dho     = dbe_->book2D(name,name,33,-16,16,74,0,73);
     
  name="HB RBX average Time-Ref";                       hb_time_rbx       = dbe_->book1D(name,name,36,0.5,36.5);
  name="HE RBX average Time-Ref";                       he_time_rbx       = dbe_->book1D(name,name,36,0.5,36.5);
  name="HO RBX average Time-Ref";                       ho_time_rbx       = dbe_->book1D(name,name,36,0.5,36.5);
  name="HF RoBox average Time-Ref";                     hf_time_rbx       = dbe_->book1D(name,name,24,0.5,24.5);
  
  char str[200];
  for(int i=1;i<=18;i++){ sprintf(str,"HBM%02i",i);     hb_time_rbx->setBinLabel(i,str);    }
  for(int i=1;i<=18;i++){ sprintf(str,"HBP%02i",i);     hb_time_rbx->setBinLabel(i+18,str); }
  for(int i=1;i<=18;i++){ sprintf(str,"HEM%02i",i);     he_time_rbx->setBinLabel(i,str);    }
  for(int i=1;i<=18;i++){ sprintf(str,"HEP%02i",i);     he_time_rbx->setBinLabel(i+18,str); }
  for(int i=1;i<=12;i++){ sprintf(str,"HFM%02i",i);     hf_time_rbx->setBinLabel(i,str);    }
  for(int i=1;i<=12;i++){ sprintf(str,"HFP%02i",i);     hf_time_rbx->setBinLabel(i+12,str); }
  for(int i=1;i<=6;i++){  sprintf(str,"HO2M%02i",i*2);  ho_time_rbx->setBinLabel(i,str);    }
  for(int i=1;i<=6;i++){  sprintf(str,"HO1M%02i",i*2);  ho_time_rbx->setBinLabel(i+6,str);  }
  for(int i=1;i<=12;i++){ sprintf(str,"HO0%02i",i);     ho_time_rbx->setBinLabel(i+12,str); }
  for(int i=1;i<=6;i++){  sprintf(str,"HO1P%02i",i*2);  ho_time_rbx->setBinLabel(i+24,str); }
  for(int i=1;i<=6;i++){  sprintf(str,"HO2P%02i",i*2);  ho_time_rbx->setBinLabel(i+30,str); }

  Time2Dhbhehf->setAxisTitle("i#eta",1);
  Time2Dhbhehf->setAxisTitle("i#phi",2);
  Time2Dho->setAxisTitle("i#eta",1);
  Time2Dho->setAxisTitle("i#phi",2);
  Energy2Dhbhehf->setAxisTitle("i#eta",1);
  Energy2Dhbhehf->setAxisTitle("i#phi",2);
  Energy2Dho->setAxisTitle("i#eta",1);
  Energy2Dho->setAxisTitle("i#phi",2);
  refTime2Dhbhehf->setAxisTitle("i#eta",1);
  refTime2Dhbhehf->setAxisTitle("i#phi",2);
  refTime2Dho->setAxisTitle("i#eta",1);
  refTime2Dho->setAxisTitle("i#phi",2);
  refEnergy2Dhbhehf->setAxisTitle("i#eta",1);
  refEnergy2Dhbhehf->setAxisTitle("i#phi",2);
  refEnergy2Dho->setAxisTitle("i#eta",1);
  refEnergy2Dho->setAxisTitle("i#phi",2);

  refTime2Dhbhehf->setAxisRange(0,2,3);
  refTime2Dho->setAxisRange(0,2,3);
  refEnergy2Dhbhehf->setAxisRange(0.5,1.5,3);
  refEnergy2Dho->setAxisRange(0.5,1.5,3);

  ReferenceRun="UNKNOWN";
  LoadReference();
  dbe_->setCurrentFolder(subdir_);
  RefRun_= dbe_->bookString("HcalDetDiagLaserMonitor Reference Run",ReferenceRun);

  dbe_->setCurrentFolder(subdir_+"Raddam Plots");
  for(int i=0;i<56;i++){
     sprintf(str,"RADDAM (%i %i)",RADDAM_CH[i].eta,RADDAM_CH[i].phi);                                             
     Raddam[i] = dbe_->book1D(str,str,10,-0.5,9.5);  
  }
  dbe_->setCurrentFolder(subdir_+"Plots for client");
  ProblemCellsByDepth_timing_val = new EtaPhiHists();
  ProblemCellsByDepth_timing_val->setup(dbe_," Laser Timing difference");
  ProblemCellsByDepth_energy_val = new EtaPhiHists();
  ProblemCellsByDepth_energy_val->setup(dbe_," Laser Energy difference");
}


HcalDetDiagLaserMonitor::~HcalDetDiagLaserMonitor(){

}

// ------------ method called to for each event  ------------
void HcalDetDiagLaserMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  HcalBaseDQMonitor::analyze(iEvent,iSetup); // base class increments ievt_, etc. counters

int  eta,phi,depth,nTS;
static bool HBHEseq,HOseq,HFseq;
static int  lastHBHEorbit,lastHOorbit,lastHForbit,nChecksHBHE,nChecksHO,nChecksHF,ievt_hbhe,ievt_ho,ievt_hf;
   if(ievt_==-1){ 
       ievt_=0;HBHEseq=HOseq=HFseq=false; lastHBHEorbit=lastHOorbit=lastHForbit=-1;nChecksHBHE=nChecksHO=nChecksHF=0; 
       ievt_hbhe=0,ievt_ho=0,ievt_hf=0;
   }

   if(!dbe_) return; 
   bool LaserEvent=false;
   bool LaserRaddam=false;
   int orbit=iEvent.orbitNumber();
   meRUN_->Fill(iEvent.id().run());
   // for local runs 
   edm::Handle<HcalTBTriggerData> trigger_data;
   iEvent.getByType(trigger_data);
   if(trigger_data.isValid()){
       if(trigger_data->wasLaserTrigger()) LaserEvent=true;
       LocalRun=true;
   }
   if(!LocalRun && Online_){
      if(HBHEseq && (orbit-lastHBHEorbit)>(11223*10) && ievt_hbhe>40){
         HBHEseq=false;
         fillHistos(HcalBarrel);
         fillProblems(HcalBarrel);
         fillProblems(HcalEndcap);
         nChecksHBHE++;
         ievt_hbhe=0;
         for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++) hb_data[i][j][k].reset();
         for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++) he_data[i][j][k].reset();
      }
      if(HOseq && (orbit-lastHOorbit)>(11223*10) && ievt_ho>40){
         HOseq=false;
         fillHistos(HcalOuter);
         fillProblems(HcalOuter);
         nChecksHO++; 
         ievt_ho=0;
         for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++) ho_data[i][j][k].reset();
      }
      if(HFseq && (orbit-lastHForbit)>(11223*10) && ievt_hf>40){
         HFseq=false;
         fillHistos(HcalForward);
         fillProblems(HcalForward);
         nChecksHF++; 
         ievt_hf=0;
         if(nChecksHF==1 || (nChecksHF>1 && ((nChecksHF-1)%12)==0)){
             SaveReference();
         }
         for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++) hf_data[i][j][k].reset();
      }
   }

   // Abort Gap laser 
   if(LocalRun==false || LaserEvent==false){
       edm::Handle<FEDRawDataCollection> rawdata;
       iEvent.getByType(rawdata);
       //checking FEDs for calibration information
       for (int i=FEDNumbering::MINHCALFEDID;i<=FEDNumbering::MAXHCALFEDID; i++) {
          const FEDRawData& fedData = rawdata->FEDData(i) ;
          if ( fedData.size() < 24 ) continue ;
          int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ;
	  if(value==hc_HBHEHPD){ HBHEseq=true; HOseq=HFseq=false; lastHBHEorbit=orbit; ievt_hbhe++; }
	  if(value==hc_HOHPD){   HOseq=true; HBHEseq=HFseq=false; lastHOorbit=orbit;   ievt_ho++;   }
	  if(value==hc_HFPMT){   HFseq=true; HBHEseq=HOseq=false; lastHForbit=orbit;   ievt_hf++;   }
          
          if(value==hc_HBHEHPD || value==hc_HOHPD || value==hc_HFPMT){ LaserEvent=true; break;}
	  if(value==hc_RADDAM){ LaserEvent=true; LaserRaddam=true; break;} 
       }
   }   
   if(!LaserEvent) return;

   meEVT_->Fill(++ievt_);
   run_number=iEvent.id().run();
   double data[20];
   if(!LaserRaddam){
      edm::Handle<HBHEDigiCollection> hbhe; 
      iEvent.getByLabel(inputLabelDigi_,hbhe);
      if(hbhe.isValid()){
         for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     if(digi->id().subdet()==HcalBarrel){
		for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-2.5;
		hb_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
	     }	 
             if(digi->id().subdet()==HcalEndcap){
		for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-2.5;
		he_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
	     }
         }
      }
      edm::Handle<HODigiCollection> ho; 
      iEvent.getByLabel(inputLabelDigi_,ho);
      if(ho.isValid()){
         for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     if((eta>=11 && eta<=15 && phi>=59 && phi<=70) || (eta>=5 && eta<=10 && phi>=47 && phi<=58)){
	        for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-11.0;
	     }else{
	        for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-2.5;
	     }
             ho_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
         }
      }
      edm::Handle<HFDigiCollection> hf;
      iEvent.getByLabel(inputLabelDigi_,hf);
      if(hf.isValid()){
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-2.5;
	     hf_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
         }   
      }
   }else{ //Raddam
      edm::Handle<HFDigiCollection> hf;
      iEvent.getByLabel(inputLabelDigi_,hf);
      if(hf.isValid()){
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     int N;
	     for(N=0;N<56;N++)if(eta==RADDAM_CH[N].eta && phi==RADDAM_CH[N].phi) break;
	     if(N==56) continue;      
	     for(int i=0;i<nTS;i++) Raddam[N]->Fill(i,adc2fC[digi->sample(i).adc()&0xff]-2.5);
	     
         }   
      }
   }
}
bool HcalDetDiagLaserMonitor::get_ave_subdet(int sd,float *ave_t,float *ave_e,float *ave_t_r,float *ave_e_r){
double T=0,nT=0,E=0,nE=0,Tr=0,nTr=0,Er=0,nEr=0;
   if(sd==HcalBarrel) for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++){ 
      for(int depth=1;depth<=2;depth++){
         if(hb_data[eta+42][phi-1][depth-1].get_statistics()>10){
	    double ave=0,rms=0,time=0,time_rms=0;
	    hb_data[eta+42][phi-1][depth-1].get_average_amp(&ave,&rms);
	    hb_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
            T+=time; nT++; E+=ave; nE++;
            if(hb_data[eta+42][phi-1][depth-1].get_reference(&ave,&rms,&time,&time_rms)){
              Tr+=time; nTr++; Er+=ave; nEr++;}
         }
      } 
   } 
   // HE histograms
   if(sd==HcalEndcap) for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++){
      for(int depth=1;depth<=3;depth++){
         if(he_data[eta+42][phi-1][depth-1].get_statistics()>10){
            double ave=0; double rms=0; double time=0; double time_rms=0;
	    he_data[eta+42][phi-1][depth-1].get_average_amp(&ave,&rms);
	    he_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
            T+=time; nT++; E+=ave; nE++;
            if(he_data[eta+42][phi-1][depth-1].get_reference(&ave,&rms,&time,&time_rms)){
              Tr+=time; nTr++; Er+=ave; nEr++;}
        }
      }
   } 
   // HF histograms
   if(sd==HcalForward) for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++){
      for(int depth=1;depth<=2;depth++){
         if(hf_data[eta+42][phi-1][depth-1].get_statistics()>10){
            double ave=0; double rms=0; double time=0; double time_rms=0;
	    hf_data[eta+42][phi-1][depth-1].get_average_amp(&ave,&rms);
	    hf_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    T+=time; nT++; E+=ave; nE++;
            if(hf_data[eta+42][phi-1][depth-1].get_reference(&ave,&rms,&time,&time_rms)){
              Tr+=time; nTr++; Er+=ave; nEr++;}
         }
      }	
   } 
   // HO histograms
   if(sd==HcalOuter) for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++){
      for(int depth=4;depth<=4;depth++){
         if(ho_data[eta+42][phi-1][depth-1].get_statistics()>10){
            double ave=0; double rms=0; double time=0; double time_rms=0;
	    ho_data[eta+42][phi-1][depth-1].get_average_amp(&ave,&rms);
	    ho_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    T+=time; nT++; E+=ave; nE++;
            if(ho_data[eta+42][phi-1][depth-1].get_reference(&ave,&rms,&time,&time_rms)){
              Tr+=time; nTr++; Er+=ave; nEr++;}
         }
      }
   } 
   if(nT<200 || nE<200 || nTr<200 || nEr<200) return false;
   *ave_t=T/nT;
   *ave_e=E/nE;
   *ave_t_r=Tr/nTr;
   *ave_e_r=Er/nEr;
   return true;
}

void HcalDetDiagLaserMonitor::fillProblems(int sd){
float ave_t,ave_e,ave_t_r,ave_e_r;
   if(!get_ave_subdet(sd,&ave_t,&ave_e,&ave_t_r,&ave_e_r)) return;

   for(int i=0;i<4;i++){
      ProblemCellsByDepth_energy->depth[i]->Reset();
      ProblemCellsByDepth_timing->depth[i]->Reset();
   }

   std::vector <HcalElectronicsId> AllElIds = emap->allElectronicsIdPrecision();
   for (std::vector <HcalElectronicsId>::iterator eid = AllElIds.begin(); eid != AllElIds.end(); eid++){
     DetId detid=emap->lookup(*eid);
     if (detid.det()!=DetId::Hcal) continue;
     HcalGenericDetId gid(emap->lookup(*eid));
     if (gid.null()) 
       continue;
     if (gid.genericSubdet()!=HcalGenericDetId::HcalGenBarrel &&
	 gid.genericSubdet()!=HcalGenericDetId::HcalGenEndcap  &&
	 gid.genericSubdet()!=HcalGenericDetId::HcalGenForward &&
	 gid.genericSubdet()!=HcalGenericDetId::HcalGenOuter)
       continue;
     int eta=0,phi=0,depth=0;
     HcalDetId hid(detid);
     eta=hid.ieta();
     phi=hid.iphi();
     depth=hid.depth();

     int e=CalcEtaBin(sd,eta,depth)+1;
     if(detid.subdetId()==HcalBarrel && sd==HcalBarrel){
	double val=0,rms=0,time=0,time_rms=0,VAL=0,RMS=0,TIME=0,TIME_RMS=0;
	if(!hb_data[eta+42][phi-1][depth-1].get_reference(&val,&rms,&time,&time_rms)) continue;
        if(!hb_data[eta+42][phi-1][depth-1].get_average_amp(&VAL,&RMS)) continue;
	if(!hb_data[eta+42][phi-1][depth-1].get_average_time(&TIME,&TIME_RMS)) continue;
        hb_data[eta+42][phi-1][depth-1].nChecks++;
        float diff_t=(TIME-ave_t)-(time-ave_t_r); if(diff_t<0) diff_t=-diff_t;
        if(diff_t>LaserTimingThreshold){
             hb_data[eta+42][phi-1][depth-1].nBadTime++; 
             ProblemCellsByDepth_timing_val->depth[depth-1]->setBinContent(e,phi,(TIME-ave_t)-(time-ave_t_r));
        }else ProblemCellsByDepth_timing_val->depth[depth-1]->setBinContent(e,phi,0); 
        if(VAL!=0 && val!=0 && ave_e!=0 && ave_e_r!=0){
          float diff_e=((VAL/ave_e))/(val/ave_e_r);
          if(diff_e>(1+LaserEnergyThreshold) ||diff_e<(1-LaserEnergyThreshold) ){
               hb_data[eta+42][phi-1][depth-1].nBadEnergy++;
               ProblemCellsByDepth_energy_val->depth[depth-1]->setBinContent(e,phi,((VAL/ave_e))/(val/ave_e_r));
          }else ProblemCellsByDepth_energy_val->depth[depth-1]->setBinContent(e,phi,0);
        }
     }
     if(detid.subdetId()==HcalEndcap && sd==HcalEndcap){
	double val=0,rms=0,time=0,time_rms=0,VAL=0,RMS=0,TIME=0,TIME_RMS=0;
	if(!he_data[eta+42][phi-1][depth-1].get_reference(&val,&rms,&time,&time_rms)) continue;
        if(!he_data[eta+42][phi-1][depth-1].get_average_amp(&VAL,&RMS)) continue;
	if(!he_data[eta+42][phi-1][depth-1].get_average_time(&TIME,&TIME_RMS)) continue;
        he_data[eta+42][phi-1][depth-1].nChecks++;
        float diff_t=(TIME-ave_t)-(time-ave_t_r); if(diff_t<0) diff_t=-diff_t;
        if(diff_t>LaserTimingThreshold){
          he_data[eta+42][phi-1][depth-1].nBadTime++;
          ProblemCellsByDepth_timing_val->depth[depth-1]->setBinContent(e,phi,(TIME-ave_t)-(time-ave_t_r));
        }else ProblemCellsByDepth_timing_val->depth[depth-1]->setBinContent(e,phi,0); 
        if(VAL!=0 && val!=0 && ave_e!=0 && ave_e_r!=0){
          float diff_e=((VAL/ave_e))/(val/ave_e_r);
          if(diff_e>(1+LaserEnergyThreshold) ||diff_e<(1-LaserEnergyThreshold) ){
            he_data[eta+42][phi-1][depth-1].nBadEnergy++;
            ProblemCellsByDepth_energy_val->depth[depth-1]->setBinContent(e,phi,((VAL/ave_e))/(val/ave_e_r));
          }else ProblemCellsByDepth_energy_val->depth[depth-1]->setBinContent(e,phi,0);
        }
     }
     if(detid.subdetId()==HcalOuter && sd==HcalOuter){
	double val=0,rms=0,time=0,time_rms=0,VAL=0,RMS=0,TIME=0,TIME_RMS=0;
	if(!ho_data[eta+42][phi-1][depth-1].get_reference(&val,&rms,&time,&time_rms)) continue;
        if(!ho_data[eta+42][phi-1][depth-1].get_average_amp(&VAL,&RMS)) continue;
	if(!ho_data[eta+42][phi-1][depth-1].get_average_time(&TIME,&TIME_RMS)) continue;
        ho_data[eta+42][phi-1][depth-1].nChecks++;
        float diff_t=(TIME-ave_t)-(time-ave_t_r); if(diff_t<0) diff_t=-diff_t;
        if(diff_t>LaserTimingThreshold){
           ho_data[eta+42][phi-1][depth-1].nBadTime++;
           ProblemCellsByDepth_timing_val->depth[depth-1]->setBinContent(e,phi,(TIME-ave_t)-(time-ave_t_r));
        }else ProblemCellsByDepth_timing_val->depth[depth-1]->setBinContent(e,phi,0); 
        if(VAL!=0 && val!=0 && ave_e!=0 && ave_e_r!=0){
          float diff_e=((VAL/ave_e))/(val/ave_e_r);
          if(diff_e>(1+LaserEnergyThreshold) ||diff_e<(1-LaserEnergyThreshold) ){
            ho_data[eta+42][phi-1][depth-1].nBadEnergy++;
            ProblemCellsByDepth_energy_val->depth[depth-1]->setBinContent(e,phi,((VAL/ave_e))/(val/ave_e_r));
          }else ProblemCellsByDepth_energy_val->depth[depth-1]->setBinContent(e,phi,0);
        }
     }
     if(detid.subdetId()==HcalForward && sd==HcalForward){
	double val=0,rms=0,time=0,time_rms=0,VAL=0,RMS=0,TIME=0,TIME_RMS=0;
	if(!hf_data[eta+42][phi-1][depth-1].get_reference(&val,&rms,&time,&time_rms)) continue;
        if(!hf_data[eta+42][phi-1][depth-1].get_average_amp(&VAL,&RMS)) continue;
	if(!hf_data[eta+42][phi-1][depth-1].get_average_time(&TIME,&TIME_RMS)) continue;
        hf_data[eta+42][phi-1][depth-1].nChecks++;
        float diff_t=(TIME-ave_t)-(time-ave_t_r); if(diff_t<0) diff_t=-diff_t;
        if(diff_t>LaserTimingThreshold){
           hf_data[eta+42][phi-1][depth-1].nBadTime++;
           ProblemCellsByDepth_timing_val->depth[depth-1]->setBinContent(e,phi,(TIME-ave_t)-(time-ave_t_r));
        }else ProblemCellsByDepth_timing_val->depth[depth-1]->setBinContent(e,phi,0); 
        if(VAL!=0 && val!=0 && ave_e!=0 && ave_e_r!=0){
          float diff_e=((VAL/ave_e))/(val/ave_e_r);
          if(diff_e>(1+LaserEnergyThreshold) ||diff_e<(1-LaserEnergyThreshold) ){
            hf_data[eta+42][phi-1][depth-1].nBadEnergy++;
            ProblemCellsByDepth_energy_val->depth[depth-1]->setBinContent(e,phi,((VAL/ave_e))/(val/ave_e_r));
          }else ProblemCellsByDepth_energy_val->depth[depth-1]->setBinContent(e,phi,0);
        }
     }
   }
   for (std::vector <HcalElectronicsId>::iterator eid = AllElIds.begin(); eid != AllElIds.end(); eid++){
     DetId detid=emap->lookup(*eid);
     if (detid.det()!=DetId::Hcal) continue;
     HcalGenericDetId gid(emap->lookup(*eid));
     if (gid.null()) 
       continue;
     if (gid.genericSubdet()!=HcalGenericDetId::HcalGenBarrel &&
	 gid.genericSubdet()!=HcalGenericDetId::HcalGenEndcap  &&
	 gid.genericSubdet()!=HcalGenericDetId::HcalGenForward &&
	 gid.genericSubdet()!=HcalGenericDetId::HcalGenOuter)
       continue;

     int eta=0,phi=0,depth=0;
     HcalDetId hid(detid);
     eta=hid.ieta();
     phi=hid.iphi();
     depth=hid.depth();
   
     if(detid.subdetId()==HcalBarrel){
        if(hb_data[eta+42][phi-1][depth-1].nBadTime>0){
           int e=CalcEtaBin(HcalBarrel,eta,depth)+1; 
           double val=hb_data[eta+42][phi-1][depth-1].nBadTime/hb_data[eta+42][phi-1][depth-1].nChecks;
           ProblemCellsByDepth_timing->depth[depth-1]->setBinContent(e,phi,val);
        } 
        if(hb_data[eta+42][phi-1][depth-1].nBadEnergy>0){
           int e=CalcEtaBin(HcalBarrel,eta,depth)+1; 
           double val=hb_data[eta+42][phi-1][depth-1].nBadEnergy/hb_data[eta+42][phi-1][depth-1].nChecks;
           ProblemCellsByDepth_energy->depth[depth-1]->setBinContent(e,phi,val);
        } 
     }
     if(detid.subdetId()==HcalEndcap){
        if(he_data[eta+42][phi-1][depth-1].nBadTime>0){
           int e=CalcEtaBin(HcalEndcap,eta,depth)+1; 
           double val=he_data[eta+42][phi-1][depth-1].nBadTime/he_data[eta+42][phi-1][depth-1].nChecks;
           ProblemCellsByDepth_timing->depth[depth-1]->setBinContent(e,phi,val);
        } 
        if(he_data[eta+42][phi-1][depth-1].nBadEnergy>0){
           int e=CalcEtaBin(HcalEndcap,eta,depth)+1; 
           double val=he_data[eta+42][phi-1][depth-1].nBadEnergy/he_data[eta+42][phi-1][depth-1].nChecks;
           ProblemCellsByDepth_energy->depth[depth-1]->setBinContent(e,phi,val);
        } 
     }
     if(detid.subdetId()==HcalOuter){
        if(ho_data[eta+42][phi-1][depth-1].nBadTime>0){
           int e=CalcEtaBin(HcalOuter,eta,depth)+1; 
           double val=ho_data[eta+42][phi-1][depth-1].nBadTime/ho_data[eta+42][phi-1][depth-1].nChecks;
           ProblemCellsByDepth_timing->depth[depth-1]->setBinContent(e,phi,val);
        } 
        if(ho_data[eta+42][phi-1][depth-1].nBadEnergy>0){
           int e=CalcEtaBin(HcalOuter,eta,depth)+1; 
           double val=ho_data[eta+42][phi-1][depth-1].nBadEnergy/ho_data[eta+42][phi-1][depth-1].nChecks;
           ProblemCellsByDepth_energy->depth[depth-1]->setBinContent(e,phi,val); 
        } 
     }
     if(detid.subdetId()==HcalForward){
        if(hf_data[eta+42][phi-1][depth-1].nBadTime>0){
           int e=CalcEtaBin(HcalForward,eta,depth)+1; 
           double val=hf_data[eta+42][phi-1][depth-1].nBadTime/hf_data[eta+42][phi-1][depth-1].nChecks;
           ProblemCellsByDepth_timing->depth[depth-1]->setBinContent(e,phi,val);
        } 
        if(hf_data[eta+42][phi-1][depth-1].nBadEnergy>0){
           int e=CalcEtaBin(HcalForward,eta,depth)+1; 
           double val=hf_data[eta+42][phi-1][depth-1].nBadEnergy/hf_data[eta+42][phi-1][depth-1].nChecks;
           ProblemCellsByDepth_energy->depth[depth-1]->setBinContent(e,phi,val);
        } 
     }
   }
}

bool HcalDetDiagLaserMonitor::get_ave_rbx(int sd,int side,int rbx,float *ave,float *rms){
   double xt=0,xxt=0; 
   int eta_min=0,eta_max=0,n=0;
   if(sd==HcalBarrel){
      if(side>0){eta_min=1; eta_max=29;}
      if(side<0){eta_min=-29; eta_max=-1;}
      if(rbx==1){
         for(int i=eta_min;i<=eta_max;i++) for(int j=71;j<=72;j++)for(int k=1;k<=3;k++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!hb_data[i+42][j-1][k-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!hb_data[i+42][j-1][k-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	 }
         for(int i=eta_min;i<=eta_max;i++) for(int j=1;j<=2;j++)for(int k=1;k<=3;k++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!hb_data[i+42][j-1][k-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!hb_data[i+42][j-1][k-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	 }
      
      }else{
         for(int i=eta_min;i<=eta_max;i++) for(int j=((rbx-1)*4-1);j<=((rbx-1)*4+2);j++)for(int k=1;k<=3;k++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!hb_data[i+42][j-1][k-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!hb_data[i+42][j-1][k-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	 }
      }
   }
   if(sd==HcalEndcap){
      if(side>0){eta_min=1; eta_max=29;}
      if(side<0){eta_min=-29; eta_max=-1;}
      if(rbx==1){
         for(int i=eta_min;i<=eta_max;i++) for(int j=71;j<=72;j++)for(int k=1;k<=3;k++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!he_data[i+42][j-1][k-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!he_data[i+42][j-1][k-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	 }
         for(int i=eta_min;i<=eta_max;i++) for(int j=1;j<=2;j++)for(int k=1;k<=3;k++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!he_data[i+42][j-1][k-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!he_data[i+42][j-1][k-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	 }
      
      }else{
         for(int i=eta_min;i<=eta_max;i++) for(int j=((rbx-1)*4-1);j<=((rbx-1)*4+2);j++)for(int k=1;k<=3;k++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!he_data[i+42][j-1][k-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!he_data[i+42][j-1][k-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	 }
      }
   }  
   if(sd==HcalForward){
      if(side>0){eta_min=29; eta_max=40;}
      if(side<0){eta_min=-40; eta_max=-29;}
      for(int i=eta_min;i<=eta_max;i++) for(int j=((rbx-1)*6+1);j<=((rbx-1)*6+6);j++)for(int k=1;k<=2;k++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!hf_data[i+42][j-1][k-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!hf_data[i+42][j-1][k-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
      }
   }   
   if(sd==HcalOuter){
      if(side==0){
        eta_min=-4,eta_max=4;
        if(rbx==1){
          for(int i=eta_min;i<=eta_max;i++) for(int j=71;j<=72;j++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!ho_data[i+42][j-1][4-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!ho_data[i+42][j-1][4-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	  }
          for(int i=eta_min;i<=eta_max;i++) for(int j=1;j<=4;j++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!ho_data[i+42][j-1][4-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!ho_data[i+42][j-1][4-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	  }
	
	}else{
          for(int i=eta_min;i<=eta_max;i++) for(int j=((rbx-1)*6-1);j<=((rbx-1)*6+4);j++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!ho_data[i+42][j-1][4-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!ho_data[i+42][j-1][4-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	  }
	}
      }
      if(side==-1){ eta_min=-10,eta_max=-5;} 
      if(side==-2){ eta_min=-15,eta_max=-11;} 
      if(side==1) { eta_min=5,  eta_max=10;} 
      if(side==2) { eta_min=11, eta_max=15;}
      if(side!=0){
        if(rbx==1){
          for(int i=eta_min;i<=eta_max;i++) for(int j=71;j<=72;j++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!ho_data[i+42][j-1][4-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!ho_data[i+42][j-1][4-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	  }
          for(int i=eta_min;i<=eta_max;i++) for(int j=1;j<=10;j++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!ho_data[i+42][j-1][4-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!ho_data[i+42][j-1][4-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	  }
	
	}else{
          for(int i=eta_min;i<=eta_max;i++) for(int j=((rbx-1)*12-1);j<=((rbx-1)*12+10);j++){
	   double val,rms,time,time_rms;
	   double TIME,TIME_RMS;
	   if(!ho_data[i+42][j-1][4-1].get_reference(&val,&rms,&time,&time_rms)) continue;
	   if(!ho_data[i+42][j-1][4-1].get_average_time(&TIME,&TIME_RMS)) continue;
           xt+=TIME-time; xxt+=(TIME-time)*(TIME-time); n++;
	  }
	}      
      } 
   }
   if(n<10) return false;
   *ave=xt/n;
   *rms=sqrt(xxt/n-(xt*xt)/(n*n));
   return true; 
}


void HcalDetDiagLaserMonitor::fillHistos(int sd){
   if(sd==HcalBarrel || sd==HcalEndcap){
      hbheEnergy->Reset();
      hbheTime->Reset();
      hbheEnergyRMS->Reset();
      hbheTimeRMS->Reset();
      hb_time_rbx->Reset();
      he_time_rbx->Reset();
   }
   if(sd==HcalOuter){
      hoEnergy->Reset();
      hoTime->Reset();
      hoEnergyRMS->Reset();
      hoTimeRMS->Reset();
      Time2Dho->Reset();
      Energy2Dho->Reset();
      refTime2Dho->Reset(); 
      refEnergy2Dho->Reset();
      ho_time_rbx->Reset();
   }
   if(sd==HcalForward){
      hfEnergy->Reset();
      hfTime->Reset();
      hfEnergyRMS->Reset();
      hfTimeRMS->Reset();
      hf_time_rbx->Reset();
   }
   if(sd==HcalBarrel || sd==HcalEndcap){
     // HB histograms
     for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++){ 
        double T=0,nT=0,E=0,nE=0;
        for(int depth=1;depth<=2;depth++){
           if(hb_data[eta+42][phi-1][depth-1].get_statistics()>10){
	      double ave=0;
	      double rms=0;
	      double time=0; 
	      double time_rms=0;
	      hb_data[eta+42][phi-1][depth-1].get_average_amp(&ave,&rms);
	      hb_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	      hbheEnergy->Fill(ave);
	      if(ave>0)hbheEnergyRMS->Fill(rms/ave);
	      hbheTime->Fill(time);
	      hbheTimeRMS->Fill(time_rms);
	      T+=time; nT++; E+=ave; nE++;
           }
        } 
        if(nT>0){Time2Dhbhehf->setBinContent(eta+44,phi+1,T/nT);Energy2Dhbhehf->setBinContent(eta+44,phi+1,E/nE); }
     } 
     // HE histograms
     for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++){
        double T=0,nT=0,E=0,nE=0;
        for(int depth=1;depth<=3;depth++){
           if(he_data[eta+42][phi-1][depth-1].get_statistics()>10){
              double ave=0; double rms=0; double time=0; double time_rms=0;
	      he_data[eta+42][phi-1][depth-1].get_average_amp(&ave,&rms);
	      he_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	      hbheEnergy->Fill(ave);
	      if(ave>0)hbheEnergyRMS->Fill(rms/ave);
	      hbheTime->Fill(time);
	      hbheTimeRMS->Fill(time_rms);
	      T+=time; nT++; E+=ave; nE++;
           }
        }
        if(nT>0 && abs(eta)>16 ){Time2Dhbhehf->setBinContent(eta+44,phi+1,T/nT); Energy2Dhbhehf->setBinContent(eta+44,phi+1,E/nE); }	 
        if(nT>0 && abs(eta)>20 ){Time2Dhbhehf->setBinContent(eta+44,phi+2,T/nT); Energy2Dhbhehf->setBinContent(eta+44,phi+2,E/nE);}	 
     } 
   }
   if(sd==HcalForward){
     // HF histograms
     for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++){
        double T=0,nT=0,E=0,nE=0;
        for(int depth=1;depth<=2;depth++){
           if(hf_data[eta+42][phi-1][depth-1].get_statistics()>10){
              double ave=0; double rms=0; double time=0; double time_rms=0;
	      hf_data[eta+42][phi-1][depth-1].get_average_amp(&ave,&rms);
	      hf_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	      hfEnergy->Fill(ave);
	      if(ave>0)hfEnergyRMS->Fill(rms/ave);
	      hfTime->Fill(time);
	      T+=time; nT++; E+=ave; nE++;
	      hfTimeRMS->Fill(time_rms);
           }
        }	
        if(nT>0 && abs(eta)>29 ){ Time2Dhbhehf->setBinContent(eta+44,phi+1,T/nT);   Time2Dhbhehf->setBinContent(eta+44,phi+2,T/nT);}	 
        if(nT>0 && abs(eta)>29 ){ Energy2Dhbhehf->setBinContent(eta+44,phi+1,E/nE); Energy2Dhbhehf->setBinContent(eta+44,phi+2,E/nE);}	 
     }
   } 
   if(sd==HcalOuter){
     // HO histograms
     for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++){
        double T=0,nT=0,E=0,nE=0;
        for(int depth=4;depth<=4;depth++){
           if(ho_data[eta+42][phi-1][depth-1].get_statistics()>10){
              double ave=0; double rms=0; double time=0; double time_rms=0;
	      ho_data[eta+42][phi-1][depth-1].get_average_amp(&ave,&rms);
	      ho_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	      hoEnergy->Fill(ave);
	      if(ave>0)hoEnergyRMS->Fill(rms/ave);
  	      hoTime->Fill(time);
	      T+=time; nT++; E+=ave; nE++;
	      hoTimeRMS->Fill(time_rms);
           }
        }
        if(nT>0){ Time2Dho->Fill(eta,phi,T/nT); Energy2Dho->Fill(eta,phi+1,E/nE) ;}
     } 
   }

   // compare with reference...
   if(sd==HcalBarrel || sd==HcalEndcap){
     for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++){ 
        double T=0,nT=0,E=0,nE=0;
        for(int depth=1;depth<=2;depth++){
           if(hb_data[eta+42][phi-1][depth-1].get_statistics()>10){
	     double val=0,rms=0,time=0,time_rms=0;
	     double VAL=0,RMS=0,TIME=0,TIME_RMS=0;
	     if(!hb_data[eta+42][phi-1][depth-1].get_reference(&val,&rms,&time,&time_rms)) continue;
             if(!hb_data[eta+42][phi-1][depth-1].get_average_amp(&VAL,&RMS)) continue;
	     if(!hb_data[eta+42][phi-1][depth-1].get_average_time(&TIME,&TIME_RMS)) continue;
	     E+=VAL/val; nE++;
	     T+=TIME-time; nT++;
	   }  
        }
        if(nE>0) refEnergy2Dhbhehf->setBinContent(eta+44,phi+1,E/nE);  
        if(nT>0){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01;  refTime2Dhbhehf->setBinContent(eta+44,phi+1,TTT); } 
     } 
     for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++){
        double T=0,nT=0,E=0,nE=0;
        for(int depth=1;depth<=3;depth++){
           if(he_data[eta+42][phi-1][depth-1].get_statistics()>10){
	     double val=0,rms=0,time=0,time_rms=0;
	     double VAL=0,RMS=0,TIME=0,TIME_RMS=0;
	     if(!he_data[eta+42][phi-1][depth-1].get_reference(&val,&rms,&time,&time_rms)) continue;
             if(!he_data[eta+42][phi-1][depth-1].get_average_amp(&VAL,&RMS)) continue;
	     if(!he_data[eta+42][phi-1][depth-1].get_average_time(&TIME,&TIME_RMS)) continue;
	     E+=VAL/val; nE++;
	     T+=TIME-time; nT++;
	   }  
        }
        if(nE>0 && abs(eta)>16) refEnergy2Dhbhehf->setBinContent(eta+44,phi+1,E/nE);  
        if(nT>0 && abs(eta)>16){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01;  refTime2Dhbhehf->setBinContent(eta+44,phi+1,TTT); } 
        if(nE>0 && abs(eta)>20) refEnergy2Dhbhehf->setBinContent(eta+44,phi+2,E/nE);  
        if(nT>0 && abs(eta)>20){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01;  refTime2Dhbhehf->setBinContent(eta+44,phi+2,TTT); }  
     } 
   }
   if(sd==HcalForward){
     for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++){
        double T=0,nT=0,E=0,nE=0;
        for(int depth=1;depth<=2;depth++){
           if(hf_data[eta+42][phi-1][depth-1].get_statistics()>10){
	     double val=0,rms=0,time=0,time_rms=0;
	     double VAL=0,RMS=0,TIME=0,TIME_RMS=0;
	     if(!hf_data[eta+42][phi-1][depth-1].get_reference(&val,&rms,&time,&time_rms)) continue;
             if(!hf_data[eta+42][phi-1][depth-1].get_average_amp(&VAL,&RMS)) continue;
	     if(!hf_data[eta+42][phi-1][depth-1].get_average_time(&TIME,&TIME_RMS)) continue;
	     E+=VAL/val; nE++;
	     T+=TIME-time; nT++;
	   }  
        }
        if(nE>0 && abs(eta)>29) refEnergy2Dhbhehf->setBinContent(eta+44,phi+1,E/nE);  
        if(nT>0 && abs(eta)>29){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01; refTime2Dhbhehf->setBinContent(eta+44,phi+1,TTT); }
        if(nE>0 && abs(eta)>29) refEnergy2Dhbhehf->setBinContent(eta+44,phi+2,E/nE);  
        if(nT>0 && abs(eta)>29){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01; refTime2Dhbhehf->setBinContent(eta+44,phi+2,TTT); } 
     }
   } 
   if(sd==HcalOuter){
     for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++){
        double T=0,nT=0,E=0,nE=0;
        for(int depth=4;depth<=4;depth++){
           if(ho_data[eta+42][phi-1][depth-1].get_statistics()>10){
	     double val=0,rms=0,time=0,time_rms=0;
	     double VAL=0,RMS=0,TIME=0,TIME_RMS=0;
	     if(!ho_data[eta+42][phi-1][depth-1].get_reference(&val,&rms,&time,&time_rms)) continue;
             if(!ho_data[eta+42][phi-1][depth-1].get_average_amp(&VAL,&RMS)) continue;
	     if(!ho_data[eta+42][phi-1][depth-1].get_average_time(&TIME,&TIME_RMS)) continue;
	     E+=VAL/val; nE++;
	     T+=TIME-time; nT++;
	   }  
         }
        if(nE>0) refEnergy2Dho->Fill(eta,phi,E/nE);  
        if(nT>0){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01; refTime2Dho->Fill(eta,phi,TTT);}  
     } 
   }
 //////////////////////////////////////////////////////////////////////
  float min,max;
  if(sd==HcalBarrel || sd==HcalEndcap){
    min=100;max=-100;
    for(int i=1;i<=18;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalBarrel,-1,i,&ave,&rms)){
        hb_time_rbx->setBinContent(i,ave);
        hb_time_rbx->setBinError(i,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    for(int i=1;i<=18;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalBarrel,1,i,&ave,&rms)){
        hb_time_rbx->setBinContent(i+18,ave);
        hb_time_rbx->setBinError(i+18,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    if(max>-100)hb_time_rbx->setAxisRange(min-1,max+1,2);
    min=100;max=-100;
    for(int i=1;i<=18;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalEndcap,-1,i,&ave,&rms)){
        he_time_rbx->setBinContent(i,ave);
        he_time_rbx->setBinError(i,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    for(int i=1;i<=18;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalEndcap,1,i,&ave,&rms)){
        he_time_rbx->setBinContent(i+18,ave);
        he_time_rbx->setBinError(i+18,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    if(max>-100)he_time_rbx->setAxisRange(min-1,max+1,2);
  }
  //////////////////////////////////////////////////////////////////////
  if(sd==HcalOuter){
    min=100;max=-100;
    for(int i=1;i<=6;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalOuter,-2,i,&ave,&rms)){
        ho_time_rbx->setBinContent(i,ave);
        ho_time_rbx->setBinError(i,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    for(int i=1;i<=6;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalOuter,-1,i,&ave,&rms)){
        ho_time_rbx->setBinContent(i+6,ave);
        ho_time_rbx->setBinError(i+6,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    for(int i=1;i<=12;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalOuter,0,i,&ave,&rms)){
        ho_time_rbx->setBinContent(i+12,ave);
        ho_time_rbx->setBinError(i+12,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    for(int i=1;i<=6;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalOuter,1,i,&ave,&rms)){
        ho_time_rbx->setBinContent(i+24,ave);
        ho_time_rbx->setBinError(i+24,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    for(int i=1;i<=6;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalOuter,2,i,&ave,&rms)){
        ho_time_rbx->setBinContent(i+30,ave);
        ho_time_rbx->setBinError(i+30,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    if(max>-100)ho_time_rbx->setAxisRange(min-1,max+1,2);
  }
  //////////////////////////////////////////////////////////////////////
  if(sd==HcalForward){
    min=100;max=-100;
    for(int i=1;i<=12;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalForward,-1,i,&ave,&rms)){
        hf_time_rbx->setBinContent(i,ave);
        hf_time_rbx->setBinError(i,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    for(int i=1;i<=12;i++){
      float ave=-10,rms=-10;
      if(get_ave_rbx(HcalForward,1,i,&ave,&rms)){
        hf_time_rbx->setBinContent(i+12,ave);
        hf_time_rbx->setBinError(i+12,rms);
        if(ave<min) min=ave;
        if(ave>max) max=ave;
      }
    }
    if(max>-100)hf_time_rbx->setAxisRange(min-1,max+1,2);
  }
} 

void HcalDetDiagLaserMonitor::SaveReference(){
double amp,rms,Time,time_rms;
int    Eta,Phi,Depth,Statistic,Status=0;
char   Subdet[10],str[500];
   if(OutputFilePath.size()>0){
       if(!Overwrite){
          sprintf(str,"%sHcalDetDiagLaserData_run%06i_%i.root",OutputFilePath.c_str(),run_number,dataset_seq_number);
       }else{
          sprintf(str,"%sHcalDetDiagLaserData.root",OutputFilePath.c_str());
       }
       TFile *theFile = new TFile(str, "RECREATE");
       if(!theFile->IsOpen()) return;
       theFile->cd();
       sprintf(str,"%d",run_number); TObjString run(str);    run.Write("run number");
       sprintf(str,"%d",ievt_);      TObjString events(str); events.Write("Total events processed");
       sprintf(str,"%d",dataset_seq_number);      TObjString dsnum(str);  dsnum.Write("Dataset number");
       Long_t t; t=time(0); strftime(str,30,"%F %T",localtime(&t)); TObjString tm(str);  tm.Write("Dataset creation time");

       TTree *tree   =new TTree("HCAL Laser data","HCAL Laser data");
       if(tree==0)   return;
       tree->Branch("Subdet",   &Subdet,         "Subdet/C");
       tree->Branch("eta",      &Eta,            "Eta/I");
       tree->Branch("phi",      &Phi,            "Phi/I");
       tree->Branch("depth",    &Depth,          "Depth/I");
       tree->Branch("statistic",&Statistic,      "Statistic/I");
       tree->Branch("status",   &Status,         "Status/I");
       tree->Branch("amp",      &amp,            "amp/D");
       tree->Branch("rms",      &rms,            "rms/D");
       tree->Branch("time",     &Time,           "time/D");
       tree->Branch("time_rms", &time_rms,       "time_rms/D");
       sprintf(Subdet,"HB");
       for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
          if((Statistic=hb_data[eta+42][phi-1][depth-1].get_statistics1())>10){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hb_data[eta+42][phi-1][depth-1].get_status();
	     hb_data[eta+42][phi-1][depth-1].get_average_amp1(&amp,&rms);
	     hb_data[eta+42][phi-1][depth-1].get_average_time1(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"HE");
       for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=3;depth++){
         if((Statistic=he_data[eta+42][phi-1][depth-1].get_statistics1())>10){
            Eta=eta; Phi=phi; Depth=depth;
	    Status=he_data[eta+42][phi-1][depth-1].get_status();
	    he_data[eta+42][phi-1][depth-1].get_average_amp1(&amp,&rms);
	    he_data[eta+42][phi-1][depth-1].get_average_time1(&Time,&time_rms);
	    tree->Fill();
         }
       } 
       sprintf(Subdet,"HO");
       for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++) for(int depth=4;depth<=4;depth++){
         if((Statistic=ho_data[eta+42][phi-1][depth-1].get_statistics1())>10){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=ho_data[eta+42][phi-1][depth-1].get_status();
	     ho_data[eta+42][phi-1][depth-1].get_average_amp1(&amp,&rms);
	     ho_data[eta+42][phi-1][depth-1].get_average_time1(&Time,&time_rms);
	     tree->Fill();
         }
       } 
       sprintf(Subdet,"HF");
       for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
         if((Statistic=hf_data[eta+42][phi-1][depth-1].get_statistics1())>10){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hf_data[eta+42][phi-1][depth-1].get_status();
	     hf_data[eta+42][phi-1][depth-1].get_average_amp1(&amp,&rms);
	     hf_data[eta+42][phi-1][depth-1].get_average_time1(&Time,&time_rms);
	     tree->Fill();
         }
       }
       theFile->Write();
       theFile->Close();
   }
   if(XmlFilePath.size()>0){
      //create XML file
      if(!Overwrite){
         sprintf(str,"HcalDetDiagLaser_%i_%i.xml",run_number,dataset_seq_number);
      }else{
         sprintf(str,"HcalDetDiagLaser.xml");
      }
      std::string xmlName=str;
      ofstream xmlFile;
      xmlFile.open(xmlName.c_str());

      xmlFile<<"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
      xmlFile<<"<ROOT>\n";
      xmlFile<<"  <HEADER>\n";
      xmlFile<<"    <HINTS mode='only-det-root'/>\n";
      xmlFile<<"    <TYPE>\n";
      xmlFile<<"      <EXTENSION_TABLE_NAME>HCAL_DETMON_LED_LASER_V1</EXTENSION_TABLE_NAME>\n";
      xmlFile<<"      <NAME>HCAL Laser HBHE HPD [abort gap global]</NAME>\n";
      xmlFile<<"    </TYPE>\n";
      xmlFile<<"    <!-- run details -->\n";
      xmlFile<<"    <RUN>\n";
      xmlFile<<"      <RUN_TYPE>GLOBAL-RUN</RUN_TYPE>\n";
      xmlFile<<"      <RUN_NUMBER>"<<run_number<<"</RUN_NUMBER>\n";
      xmlFile<<"      <RUN_BEGIN_TIMESTAMP>2009-01-01 00:00:00</RUN_BEGIN_TIMESTAMP>\n";
      xmlFile<<"      <COMMENT_DESCRIPTION>hcal laser data</COMMENT_DESCRIPTION>\n";
      xmlFile<<"      <LOCATION>P5</LOCATION>\n";
      xmlFile<<"      <INITIATED_BY_USER>dma</INITIATED_BY_USER>\n";
      xmlFile<<"    </RUN>\n";
      xmlFile<<"  </HEADER>\n";
      xmlFile<<"  <DATA_SET>\n";
      xmlFile<<"     <!-- optional dataset metadata -->\n\n";
      xmlFile<<"     <SET_NUMBER>"<<dataset_seq_number<<"</SET_NUMBER>\n";
      xmlFile<<"     <SET_BEGIN_TIMESTAMP>2009-01-01 00:00:00</SET_BEGIN_TIMESTAMP>\n";
      xmlFile<<"     <SET_END_TIMESTAMP>2009-01-01 00:00:00</SET_END_TIMESTAMP>\n";
      xmlFile<<"     <NUMBER_OF_EVENTS_IN_SET>"<<ievt_<<"</NUMBER_OF_EVENTS_IN_SET>\n";
      xmlFile<<"     <COMMENT_DESCRIPTION>Automatic DQM output</COMMENT_DESCRIPTION>\n";
      xmlFile<<"     <DATA_FILE_NAME>"<< xmlName <<"</DATA_FILE_NAME>\n";
      xmlFile<<"     <IMAGE_FILE_NAME>data plot url or file path</IMAGE_FILE_NAME>\n";
      xmlFile<<"     <!-- who and when created this dataset-->\n\n";
      Long_t t; t=time(0); strftime(str,30,"%F %T",localtime(&t));
      xmlFile<<"     <CREATE_TIMESTAMP>"<<str<<"</CREATE_TIMESTAMP>\n";
      xmlFile<<"     <CREATED_BY_USER>dma</CREATED_BY_USER>\n";
      xmlFile<<"     <!-- version (string) and subversion (number) -->\n";
      xmlFile<<"     <!-- fields are used to read data back from the database -->\n\n";
      xmlFile<<"     <VERSION>"<<run_number<<dataset_seq_number<<"</VERSION>\n";
      xmlFile<<"     <SUBVERSION>1</SUBVERSION>\n";
      xmlFile<<"     <!--  Assign predefined dataset attributes -->\n\n";
      xmlFile<<"     <PREDEFINED_ATTRIBUTES>\n";
      xmlFile<<"        <ATTRIBUTE>\n";
      xmlFile<<"           <NAME>HCAL Dataset Status</NAME>\n";
      xmlFile<<"           <VALUE>VALID</VALUE>\n";
      xmlFile<<"        </ATTRIBUTE>\n";
      xmlFile<<"     </PREDEFINED_ATTRIBUTES>\n";
      xmlFile<<"     <!-- multiple data block records -->\n\n";

      std::vector <HcalElectronicsId> AllElIds = emap->allElectronicsIdPrecision();
      for(std::vector <HcalElectronicsId>::iterator eid = AllElIds.begin(); eid != AllElIds.end(); eid++){
         DetId detid=emap->lookup(*eid);
	 if (detid.det()!=DetId::Hcal) continue;
         HcalGenericDetId gid(emap->lookup(*eid));
	 if (gid.null()) 
	   continue;
	 if (gid.genericSubdet()!=HcalGenericDetId::HcalGenBarrel &&
	     gid.genericSubdet()!=HcalGenericDetId::HcalGenEndcap  &&
	     gid.genericSubdet()!=HcalGenericDetId::HcalGenForward &&
	     gid.genericSubdet()!=HcalGenericDetId::HcalGenOuter)
	   continue;
         int eta,phi,depth; 
         std::string subdet="";
	 HcalDetId hid(detid);
	 eta=hid.ieta();
	 phi=hid.iphi();
	 depth=hid.depth(); 
         
         double e=0,e_rms=0,t=0,t_rms=0;
         if(detid.subdetId()==HcalBarrel){
             subdet="HB";
             Statistic=hb_data[eta+42][phi-1][depth-1].get_statistics1();
	     Status   =hb_data[eta+42][phi-1][depth-1].get_status();
	     hb_data[eta+42][phi-1][depth-1].get_average_amp1(&e,&e_rms);
	     hb_data[eta+42][phi-1][depth-1].get_average_time1(&t,&t_rms);
         }else if(detid.subdetId()==HcalEndcap){
             subdet="HE";
             Statistic=he_data[eta+42][phi-1][depth-1].get_statistics1();
	     Status   =he_data[eta+42][phi-1][depth-1].get_status();
	     he_data[eta+42][phi-1][depth-1].get_average_amp1(&e,&e_rms);
	     he_data[eta+42][phi-1][depth-1].get_average_time1(&t,&t_rms);
	 }else if(detid.subdetId()==HcalForward){
             subdet="HF";
             Statistic=hf_data[eta+42][phi-1][depth-1].get_statistics1();
	     Status   =hf_data[eta+42][phi-1][depth-1].get_status();
	     hf_data[eta+42][phi-1][depth-1].get_average_amp1(&e,&e_rms);
	     hf_data[eta+42][phi-1][depth-1].get_average_time1(&t,&t_rms);
	 }else if(detid.subdetId()==HcalOuter){
             subdet="HO";
             Statistic=ho_data[eta+42][phi-1][depth-1].get_statistics1();
	     Status   =ho_data[eta+42][phi-1][depth-1].get_status();
	     ho_data[eta+42][phi-1][depth-1].get_average_amp1(&e,&e_rms);
	     ho_data[eta+42][phi-1][depth-1].get_average_time1(&t,&t_rms);
         }else continue;
         xmlFile<<"       <DATA>\n";
         xmlFile<<"          <NUMBER_OF_EVENTS_USED>"<<Statistic<<"</NUMBER_OF_EVENTS_USED>\n";
         xmlFile<<"          <SIGNAL_MEAN>"<<e<<"</SIGNAL_MEAN>\n";
         xmlFile<<"          <SIGNAL_RMS>"<<e_rms<<"</SIGNAL_RMS>\n";
         xmlFile<<"          <TIME_MEAN>"<<t<<"</TIME_MEAN>\n";
         xmlFile<<"          <TIME_RMS>"<<t_rms<<"</TIME_RMS>\n";
         xmlFile<<"          <CHANNEL_STATUS_WORD>"<<Status<<"</CHANNEL_STATUS_WORD>\n";
         xmlFile<<"          <CHANNEL_OBJECTNAME>HcalDetId</CHANNEL_OBJECTNAME>\n";
         xmlFile<<"             <SUBDET>"<<subdet<<"</SUBDET>\n";
         xmlFile<<"             <IETA>"<<eta<<"</IETA>\n";
         xmlFile<<"             <IPHI>"<<phi<<"</IPHI>\n";
         xmlFile<<"             <DEPTH>"<<depth<<"</DEPTH>\n";
         xmlFile<<"             <TYPE>0</TYPE>\n";
         xmlFile<<"       </DATA>\n";
      }
      /////////////////////////////////
      xmlFile<<"  </DATA_SET>\n";
      xmlFile<<"</ROOT>\n";
      xmlFile.close();
      sprintf(str,"zip %s.zip %s",xmlName.c_str(),xmlName.c_str());
      system(str);
      sprintf(str,"rm -f %s",xmlName.c_str());
      system(str);
      sprintf(str,"mv -f %s.zip %s",xmlName.c_str(),XmlFilePath.c_str());
      system(str);
   }
   for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++) hb_data[i][j][k].reset1();
   for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++) he_data[i][j][k].reset1();
   for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++) ho_data[i][j][k].reset1();
   for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++) hf_data[i][j][k].reset1();
   ievt_=0;
   dataset_seq_number++;
}
void HcalDetDiagLaserMonitor::LoadReference(){
double amp,rms,time,time_rms;
int Eta,Phi,Depth;
char subdet[10];
TFile *f;
 if(gSystem->AccessPathName(ReferenceData.c_str())) return;
 f = new TFile(ReferenceData.c_str(),"READ");
 
 if(!f->IsOpen()) return ;
 TObjString *STR=(TObjString *)f->Get("run number");
 
 if(STR){ std::string Ref(STR->String()); ReferenceRun=Ref;}
 
      TTree*  t=(TTree*)f->Get("HCAL Laser data");
      if(!t) return;
      t->SetBranchAddress("Subdet",   subdet);
      t->SetBranchAddress("eta",      &Eta);
      t->SetBranchAddress("phi",      &Phi);
      t->SetBranchAddress("depth",    &Depth);
      t->SetBranchAddress("amp",      &amp);
      t->SetBranchAddress("rms",      &rms);
      t->SetBranchAddress("time",     &time);
      t->SetBranchAddress("time_rms", &time_rms);
     
      for(int ievt=0;ievt<t->GetEntries();ievt++){
         t->GetEntry(ievt);
	 if(strcmp(subdet,"HB")==0) hb_data[Eta+42][Phi-1][Depth-1].set_reference(amp,rms,time,time_rms);
	 if(strcmp(subdet,"HE")==0) he_data[Eta+42][Phi-1][Depth-1].set_reference(amp,rms,time,time_rms);
	 if(strcmp(subdet,"HO")==0) ho_data[Eta+42][Phi-1][Depth-1].set_reference(amp,rms,time,time_rms);
	 if(strcmp(subdet,"HF")==0) hf_data[Eta+42][Phi-1][Depth-1].set_reference(amp,rms,time,time_rms);
      }
      f->Close();
      IsReference=true;
} 

void HcalDetDiagLaserMonitor::endRun(const edm::Run& run, const edm::EventSetup& c) {
    if((LocalRun || !Online_) && ievt_>10){
       fillHistos(HcalBarrel);
       fillHistos(HcalOuter);
       fillHistos(HcalForward);
       fillProblems(HcalBarrel);
       fillProblems(HcalEndcap);
       fillProblems(HcalOuter); 
       fillProblems(HcalForward); 
       SaveReference();
    }
}

void HcalDetDiagLaserMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c){}
void HcalDetDiagLaserMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c){}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDetDiagLaserMonitor);
