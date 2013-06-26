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
// $Id: HcalDetDiagLaserMonitor.cc,v 1.21 2012/08/30 21:48:48 wdd Exp $
//
//

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
#include "TF1.h"

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
int depth;
}Raddam_ch;

Raddam_ch RADDAM_CH[56]={{-30,35,1},{-30,71,1},{-32,15,1},{-32,51,1},{-34,35,1},{-34,71,1},{-36,15,1},
                         {-36,51,1},{-38,35,1},{-38,71,1},{-40,15,1},{-40,51,1},{-41,35,1},{-41,71,1},
                         {30,21,1}, {30,57,1}, {32,1,1},  {32,37,1}, {34,21,1}, {34,57,1}, {36,1,1  },
                         {36,37,1}, {38,21,1}, {38,57,1}, {40,35,1}, {40,71,1}, {41,19,1}, {41,55,1 },
                         {-30,15,2},{-30,51,2},{-32,35,2},{-32,71,2},{-34,15,2},{-34,51,2},{-36,35,2},
                         {-36,71,2},{-38,15,2},{-38,51,2},{-40,35,2},{-40,71,2},{-41,15,2},{-41,51,2},
                         {30,1,2},  {30,37,2}, {32,21,2}, {32,57,2}, {34,1,2},  {34,37,2}, {36,21,2 },
                         {36,57,2}, {38,1,2},  {38,37,2}, {40,19,2}, {40,55,2}, {41,35,2}, {41,71,2}};
class HcalRaddamData{
  public:
   HcalRaddamData(){
      for(int i=0;i<128;i++) s1_adc[i]=s2_adc[i]=0;
      TOTEVNT=CUT1EVNT=CUT2EVNT=S1MEAN=S2MEAN=S1RMS=S2RMS=0;
      S1FITMEAN=S2FITMEAN=S1FITMEANER=S2FITMEANER=S1FITSIGMA=S2FITSIGMA=0;
      S1CHI2=S2CHI2=S1NDF=S2NDF=S1BINWIDTH=S2BINWIDTH=0;
   }
   ~HcalRaddamData(){};
   int TOTEVNT;
   int CUT1EVNT;
   int CUT2EVNT;
   float S1MEAN,S2MEAN;
   float S1RMS,S2RMS;
   float S1FITMEAN,S2FITMEAN;
   float S1FITMEANER,S2FITMEANER;
   float S1FITSIGMA,S2FITSIGMA;
   float S1CHI2,S2CHI2;
   float S1NDF,S2NDF;
   float S1BINWIDTH,S2BINWIDTH;
   int s1_adc[128];
   int s2_adc[128];
};

HcalRaddamData Raddam_data[56];

class HcalDetDiagLaserData{
public: 
   HcalDetDiagLaserData(){ 
	     IsRefetence=false;
             ds_amp=ds_rms=ds_time=ds_time_rms=-100;
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
   void   set_data(float val,float rms,float time,float time_rms){
             ds_amp=val; ds_rms=rms;
	     ds_time=time; ds_time_rms=time_rms;
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
             if(ds_time>-10){ *ave=ds_amp; *rms=ds_rms; return true; }
	     if(n>0){ *ave=Xe/n; *rms=sqrt(XXe/n-(Xe*Xe)/(n*n));} else return false;
             return true; 
          }
   bool   get_average_time(double *ave,double *rms){
             if(ds_time>-10){ *ave=ds_time; *rms=ds_time_rms; return true; }
             if(n>0){ *ave=Xt/n; *rms=sqrt(XXt/n-(Xt*Xt)/(n*n));} else return false;
             return true; 
          }
   int    get_statistics(){
	     return (int)n;
	  } 
   void   set_statistics(int stat){
	     n=stat;
	  } 
   void   set_statistics1(int stat){
	     n1=stat;
	  } 
   int    get_overflow(){
             return overflow;
          }   
   int    get_undeflow(){
             return undeflow;
          }   
   bool   get_average_amp1(double *ave,double *rms){
             if(ds_time>-10){ *ave=ds_amp; *rms=ds_rms; return true; }
	     if(n1>0){ *ave=Xe1/n1; *rms=sqrt(XXe1/n1-(Xe1*Xe1)/(n1*n1));} else return false;
             return true; 
          }
   bool   get_average_time1(double *ave,double *rms){
             if(ds_time>-10){ *ave=ds_time; *rms=ds_time_rms; return true; }
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
   float  ds_amp;
   float  ds_rms;
   float  ds_time;
   float  ds_time_rms;
   int    status;
   float    nChecks,nBadTime,nBadEnergy;
};

class HcalDetDiagLaserMonitor : public HcalBaseDQMonitor {
   public:
      explicit HcalDetDiagLaserMonitor(const edm::ParameterSet&);
      ~HcalDetDiagLaserMonitor();

   private:
      HcalDetDiagLaserData* GetCalib(std::string sd,int eta,int phi){
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
      return &calib_data[SD][ETA+2][PHI-1];
      };   
      void beginRun(const edm::Run& run, const edm::EventSetup& c);  
      void endRun(const edm::Run& run, const edm::EventSetup& c);
      void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c) ;
      void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c);
      void analyze(const edm::Event&, const edm::EventSetup&);

      const HcalElectronicsMap  *emap;
      edm::InputTag inputLabelDigi_;
      edm::InputTag calibDigiLabel_;
      edm::InputTag rawDataLabel_;
      edm::InputTag hcalTBTriggerDataTag_;

      void SaveReference();
      void SaveRaddamData();
      void LoadReference();
      void LoadDataset();

      bool get_ave_rbx(int sd,int side,int rbx,float *ave,float *rms);
      bool get_ave_subdet(int sd,float *ave_t,float *ave_e,float *ave_t_r,float *ave_e_r);
      void fillHistos(int sd);
      void fillProblems(int sd);
      int  nHBHEchecks,nHOchecks,nHFchecks;
      double LaserTimingThreshold,LaserEnergyThreshold,RaddamThreshold1,RaddamThreshold2;

      int         ievt_;
      int         run_number;
      int         dataset_seq_number;
      bool        IsReference;
      bool        LocalRun,RaddamRun;
      int         nHB,nHE,nHO,nHF;
      std::string ReferenceData;
      std::string ReferenceRun;
      std::string OutputFilePath;
      std::string XmlFilePath;
      std::string baseFolder_;
      std::string prefixME_;
      bool        Online_;
      bool        Overwrite;

      // to create html from processed dataset
      std::string DatasetName;
      std::string htmlOutputPath;
      bool createHTMLonly;

      MonitorElement *meEVT_,*meRUN_,*htmlFolder;
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
      TH1F *S1[56],*S2[56];

      EtaPhiHists* ProblemCellsByDepth_timing;
      EtaPhiHists* ProblemCellsByDepth_energy;
      std::vector<std::string> problemnames_;

      EtaPhiHists* ProblemCellsByDepth_timing_val;
      EtaPhiHists* ProblemCellsByDepth_energy_val;

      HcalDetDiagLaserData hb_data[85][72][4];
      HcalDetDiagLaserData he_data[85][72][4];
      HcalDetDiagLaserData ho_data[85][72][4];
      HcalDetDiagLaserData hf_data[85][72][4];
      HcalDetDiagLaserData calib_data[5][5][72];

      std::map<unsigned int, int> KnownBadCells_;
};

HcalDetDiagLaserMonitor::HcalDetDiagLaserMonitor(const edm::ParameterSet& iConfig) :
  hcalTBTriggerDataTag_(iConfig.getParameter<edm::InputTag>("hcalTBTriggerDataTag"))
{
  ievt_=-1;
  emap=0;
  dataset_seq_number=1;
  run_number=-1;
  IsReference=false;
  LocalRun=RaddamRun=false;
  createHTMLonly=false;
  nHB=nHE=nHO=nHF=0;
  nHBHEchecks=nHOchecks=nHFchecks=0;

  inputLabelDigi_  = iConfig.getUntrackedParameter<edm::InputTag>("digiLabel",edm::InputTag("hcalDigis"));
  calibDigiLabel_  = iConfig.getUntrackedParameter<edm::InputTag>("calibDigiLabel",edm::InputTag("hcalDigis"));
  rawDataLabel_   =  iConfig.getUntrackedParameter<edm::InputTag>("RawDataLabel",edm::InputTag("source"));

  ReferenceData    = iConfig.getUntrackedParameter<std::string>("LaserReferenceData" ,"");
  OutputFilePath   = iConfig.getUntrackedParameter<std::string>("OutputFilePath", "");
  DatasetName      = iConfig.getUntrackedParameter<std::string>("LaserDatasetName", "");
  htmlOutputPath   = iConfig.getUntrackedParameter<std::string>("htmlOutputPath", "");
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
  RaddamThreshold1     = iConfig.getUntrackedParameter<double>("RaddamThreshold1",10.0);
  RaddamThreshold2     = iConfig.getUntrackedParameter<double>("RaddamThreshold2",0.95);
}
void HcalDetDiagLaserMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c){
  edm::ESHandle<HcalChannelQuality> p;
  c.get<HcalChannelQualityRcd>().get(p);
  HcalChannelQuality* chanquality= new HcalChannelQuality(*p.product());
  std::vector<DetId> mydetids = chanquality->getAllChannels();
  KnownBadCells_.clear();

  for (std::vector<DetId>::const_iterator i = mydetids.begin();i!=mydetids.end();++i){
     if (i->det()!=DetId::Hcal) continue; // not an hcal cell
     HcalDetId id=HcalDetId(*i);
     int status=(chanquality->getValues(id))->getValue();
     if((status & HcalChannelStatus::HcalCellOff) || (status & HcalChannelStatus::HcalCellMask)){
	 KnownBadCells_[id.rawId()]=status;
     }
  } 

  edm::ESHandle<HcalDbService> conditions_;
  c.get<HcalDbRecord>().get(conditions_);
  emap=conditions_->getHcalMapping();
  
  HcalBaseDQMonitor::setup();
  if (!dbe_) return;
  std::string name;
 
  dbe_->setCurrentFolder(subdir_);   
  meEVT_ = dbe_->bookInt("HcalDetDiagLaserMonitor Event Number");
  meRUN_ = dbe_->bookInt("HcalDetDiagLaserMonitor Run Number");

  ReferenceRun="UNKNOWN";
  LoadReference();
  LoadDataset();
  if(DatasetName.size()>0 && createHTMLonly){
     char str[200]; sprintf(str,"%sHcalDetDiagLaserData_run%i_%i/",htmlOutputPath.c_str(),run_number,dataset_seq_number);
     htmlFolder=dbe_->bookString("HcalDetDiagLaserMonitor HTML folder",str);
     MonitorElement *me;
     dbe_->setCurrentFolder(prefixME_+"HcalInfo");
     me=dbe_->bookInt("HBpresent");
     if(nHB>0) me->Fill(1);
     me=dbe_->bookInt("HEpresent");
     if(nHE>0) me->Fill(1);
     me=dbe_->bookInt("HOpresent");
     if(nHO>0) me->Fill(1);
     me=dbe_->bookInt("HFpresent");
     if(nHF>0) me->Fill(1);
  }
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
 if(createHTMLonly) return;
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
   iEvent.getByLabel(hcalTBTriggerDataTag_, trigger_data);
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
     iEvent.getByLabel(rawDataLabel_ ,rawdata);
       // edm::Handle<FEDRawDataCollection> rawdata;
       // iEvent.getByType(rawdata);
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
   edm::Handle<HBHEDigiCollection> hbhe; 
   iEvent.getByLabel(inputLabelDigi_,hbhe);
   edm::Handle<HODigiCollection> ho; 
   iEvent.getByLabel(inputLabelDigi_,ho);
   edm::Handle<HFDigiCollection> hf;
   iEvent.getByLabel(inputLabelDigi_,hf);
   edm::Handle<HcalCalibDigiCollection> calib;
   iEvent.getByLabel(calibDigiLabel_, calib); 

   if(LocalRun && LaserEvent){
      int N=0; 
      if(hf.isValid()){
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth();
             float e=0;
	     for(int i=0;i<digi->size();i++) e+=adc2fC[digi->sample(i).adc()&0xff]-2.5;
             if(e>40){ N++;}
         }
      }
      if(N>50 && N<57){ RaddamRun=true; /*LaserRaddam=true;*/}
   }
   if(RaddamRun){
      if(hf.isValid()){
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
             int N;
	     for(N=0;N<56;N++)if(eta==RADDAM_CH[N].eta && phi==RADDAM_CH[N].phi && depth==RADDAM_CH[N].depth) break;
	     if(N==56) continue; 
             float max1=0,max2=0;
             int   nmax1=0,nmax2=0;
	     for(int i=0;i<nTS;i++){
                  if(max1<adc2fC[digi->sample(i).adc()&0xff]){ nmax1=i; max1=adc2fC[digi->sample(i).adc()&0xff]; }
             }
             Raddam_data[N].TOTEVNT++;
	     for(int i=0;i<nTS;i++){
                  if(i==nmax1) continue;
                  if(max2<adc2fC[digi->sample(i).adc()&0xff]){ nmax2=i; max2=adc2fC[digi->sample(i).adc()&0xff]; }
             }
             if(nmax1>nmax2){
                int tmp1=nmax2;
                nmax2=nmax1;nmax1=tmp1;
             }
	     if(nmax1==0 || nmax2==(nTS-1)) continue;
             if(nmax2!=(nmax1+1)) continue;
     
             if(max1<RaddamThreshold1 || max2<RaddamThreshold1) continue;
             Raddam_data[N].CUT1EVNT++;
             max1-=2.5; max2-=2.5;
             float S2=max1+max2;
             float S4=S2+adc2fC[digi->sample(nmax1-1).adc()&0xff]+adc2fC[digi->sample(nmax2+1).adc()&0xff]-5.0;
             if((S2/S4)<RaddamThreshold2) continue;
             Raddam_data[N].CUT2EVNT++;
             Raddam_data[N].s1_adc[digi->sample(nmax1).adc()&0xff]++;
             Raddam_data[N].s2_adc[digi->sample(nmax2).adc()&0xff]++;
         }
      }
   }

   meEVT_->Fill(++ievt_);
   run_number=iEvent.id().run();
   double data[20];
   if(!LaserRaddam){
      if(hbhe.isValid()){
         for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
             double ped=(adc2fC[digi->sample(0).adc()&0xff]+adc2fC[digi->sample(1).adc()&0xff])/2.0;
	     if(digi->id().subdet()==HcalBarrel){
		for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-ped;
		hb_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
	     }	 
             if(digi->id().subdet()==HcalEndcap){
		for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-ped;
		he_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
	     }
         }
      }
      if(ho.isValid()){
         for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
             double ped=(adc2fC[digi->sample(0).adc()&0xff]+adc2fC[digi->sample(1).adc()&0xff])/2.0;
	     if((eta>=11 && eta<=15 && phi>=59 && phi<=70) || (eta>=5 && eta<=10 && phi>=47 && phi<=58)){
	        for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-ped;
	     }else{
	        for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-ped;
	     }
             ho_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
         }
      }
      if(hf.isValid()){
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
             double ped=adc2fC[digi->sample(0).adc()&0xff];
	     for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-ped;
	     hf_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
         }   
      }
      if(calib.isValid())for(HcalCalibDigiCollection::const_iterator digi=calib->begin();digi!=calib->end();digi++){
         if(digi->id().cboxChannel()!=0 || digi->id().hcalSubdet()==0) continue; 
         nTS=digi->size();
         float e=0;
         for(int i=0;i<nTS;i++){ data[i]=adc2fC[digi->sample(i).adc()&0xff]; e+=data[i];} 
         if(e<15000) calib_data[digi->id().hcalSubdet()][digi->id().ieta()+2][digi->id().iphi()-1].add_statistics(data,nTS);
      }
   }else{ //Raddam
      if(hf.isValid()){
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     int N;
	     for(N=0;N<56;N++)if(eta==RADDAM_CH[N].eta && phi==RADDAM_CH[N].phi && depth==RADDAM_CH[N].depth) break;
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
     if(!(!(gid.null()) && 
            (gid.genericSubdet()==HcalGenericDetId::HcalGenBarrel ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenEndcap  ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenForward ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenOuter))) continue;
     int eta=0,phi=0,depth=0;
     HcalDetId hid(detid);
     if(KnownBadCells_.find(hid.rawId())==KnownBadCells_.end()) continue;
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
     if(!(!(gid.null()) && 
            (gid.genericSubdet()==HcalGenericDetId::HcalGenBarrel ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenEndcap  ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenForward ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenOuter))) continue;
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
     for(int eta=-10;eta<=15;eta++) for(int phi=1;phi<=72;phi++){
        if(eta>10 && !isSiPM(eta,phi,4)) continue;
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
        if(eta>10 && !isSiPM(eta,phi,4)) continue;
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
       sprintf(Subdet,"CALIB_HB");
       for(int eta=-1;eta<=1;eta++) for(int phi=1;phi<=72;phi++){
          if((Statistic=calib_data[1][eta+2][phi-1].get_statistics1())>10){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[1][eta+2][phi-1].get_status();
 	     calib_data[1][eta+2][phi-1].get_average_amp1(&amp,&rms);
	     calib_data[1][eta+2][phi-1].get_average_time1(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"CALIB_HE");
       for(int eta=-1;eta<=1;eta++) for(int phi=1;phi<=72;phi++){
          if((Statistic=calib_data[2][eta+2][phi-1].get_statistics1())>10){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[2][eta+2][phi-1].get_status();
 	     calib_data[2][eta+2][phi-1].get_average_amp1(&amp,&rms);
	     calib_data[2][eta+2][phi-1].get_average_time1(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"CALIB_HO");
       for(int eta=-2;eta<=2;eta++) for(int phi=1;phi<=72;phi++){
          if((Statistic=calib_data[3][eta+2][phi-1].get_statistics1())>10){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[3][eta+2][phi-1].get_status();
 	     calib_data[3][eta+2][phi-1].get_average_amp1(&amp,&rms);
	     calib_data[3][eta+2][phi-1].get_average_time1(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"CALIB_HF");
       for(int eta=-2;eta<=2;eta++) for(int phi=1;phi<=72;phi++){
          if((Statistic=calib_data[4][eta+2][phi-1].get_statistics1())>10){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[4][eta+2][phi-1].get_status();
 	     calib_data[4][eta+2][phi-1].get_average_amp1(&amp,&rms);
	     calib_data[4][eta+2][phi-1].get_average_time1(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       theFile->Write();
       theFile->Close();
   }
   if(XmlFilePath.size()>0){
      char TIME[40];
      Long_t t; t=time(0); strftime(TIME,30,"%F %T",localtime(&t));
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
      xmlFile<<"     <CREATE_TIMESTAMP>"<<TIME<<"</CREATE_TIMESTAMP>\n";
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
         if(!(!(gid.null()) && 
            (gid.genericSubdet()==HcalGenericDetId::HcalGenBarrel ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenEndcap  ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenForward ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenOuter))) continue;
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


      //create CALIB XML file 
      sprintf(str,"HcalDetDiagLaserCalib_%i_%i.xml",run_number,dataset_seq_number);
      std::string xmlNameCalib=str;
      ofstream xmlFileCalib;
      xmlFileCalib.open(xmlNameCalib.c_str());

      xmlFileCalib<<"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
      xmlFileCalib<<"<ROOT>\n";
      xmlFileCalib<<"  <HEADER>\n";
      xmlFileCalib<<"    <HINTS mode='only-det-root'/>\n";
      xmlFileCalib<<"    <TYPE>\n";
      xmlFileCalib<<"      <EXTENSION_TABLE_NAME>HCAL_DETMON_LED_LASER_V1</EXTENSION_TABLE_NAME>\n";
      xmlFileCalib<<"      <NAME>HCAL Laser CALIB [abort gap global]</NAME>\n";
      xmlFileCalib<<"    </TYPE>\n";
      xmlFileCalib<<"    <!-- run details -->\n";
      xmlFileCalib<<"    <RUN>\n";
      xmlFileCalib<<"      <RUN_TYPE>Global-RUN</RUN_TYPE>\n";
      xmlFileCalib<<"      <RUN_NUMBER>"<<run_number<<"</RUN_NUMBER>\n";
      xmlFileCalib<<"      <RUN_BEGIN_TIMESTAMP>2009-01-01 00:00:00</RUN_BEGIN_TIMESTAMP>\n";
      xmlFileCalib<<"      <COMMENT_DESCRIPTION>hcal Laser CALIB data</COMMENT_DESCRIPTION>\n";
      xmlFileCalib<<"      <LOCATION>P5</LOCATION>\n";
      xmlFileCalib<<"      <INITIATED_BY_USER>dma</INITIATED_BY_USER>\n";
      xmlFileCalib<<"    </RUN>\n";
      xmlFileCalib<<"  </HEADER>\n";
      xmlFileCalib<<"  <DATA_SET>\n";
      xmlFileCalib<<"     <!-- optional dataset metadata -->\n\n";
      xmlFileCalib<<"     <SET_NUMBER>"<<dataset_seq_number<<"</SET_NUMBER>\n";
      xmlFileCalib<<"     <SET_BEGIN_TIMESTAMP>2009-01-01 00:00:00</SET_BEGIN_TIMESTAMP>\n";
      xmlFileCalib<<"     <SET_END_TIMESTAMP>2009-01-01 00:00:00</SET_END_TIMESTAMP>\n";
      xmlFileCalib<<"     <NUMBER_OF_EVENTS_IN_SET>"<<ievt_<<"</NUMBER_OF_EVENTS_IN_SET>\n";
      xmlFileCalib<<"     <COMMENT_DESCRIPTION>Automatic DQM output</COMMENT_DESCRIPTION>\n";
      xmlFileCalib<<"     <DATA_FILE_NAME>"<< xmlNameCalib <<"</DATA_FILE_NAME>\n";
      xmlFileCalib<<"     <IMAGE_FILE_NAME>data plot url or file path</IMAGE_FILE_NAME>\n";
      xmlFileCalib<<"     <!-- who and when created this dataset-->\n\n";
      xmlFileCalib<<"     <CREATE_TIMESTAMP>"<<TIME<<"</CREATE_TIMESTAMP>\n";
      xmlFileCalib<<"     <CREATED_BY_USER>dma</CREATED_BY_USER>\n";
      xmlFileCalib<<"     <!-- version (string) and subversion (number) -->\n";
      xmlFileCalib<<"     <!-- fields are used to read data back from the database -->\n\n";
      xmlFileCalib<<"     <VERSION>"<<run_number<<dataset_seq_number<<"</VERSION>\n";
      xmlFileCalib<<"     <SUBVERSION>1</SUBVERSION>\n";
      xmlFileCalib<<"     <!--  Assign predefined dataset attributes -->\n\n";
      xmlFileCalib<<"     <PREDEFINED_ATTRIBUTES>\n";
      xmlFileCalib<<"        <ATTRIBUTE>\n";
      xmlFileCalib<<"           <NAME>HCAL Dataset Status</NAME>\n";
      xmlFileCalib<<"           <VALUE>VALID</VALUE>\n";
      xmlFileCalib<<"        </ATTRIBUTE>\n";
      xmlFileCalib<<"     </PREDEFINED_ATTRIBUTES>\n";
      xmlFileCalib<<"     <!-- multiple data block records -->\n\n";

      for(int sd=1;sd<=4;sd++) for(int eta=-2;eta<=2;eta++) for(int phi=1;phi<=72;phi++){
         std::string subdet="";
         if(sd==1) subdet="HB";
         if(sd==2) subdet="HE";
         if(sd==3) subdet="HO";
         if(sd==4) subdet="HF";
         if((calib_data[sd][eta+2][phi-1].get_statistics())>100){
             double e=0,e_rms=0,t=0,t_rms=0;
	     Status=calib_data[sd][eta+2][phi-1].get_status();
             Statistic=calib_data[sd][eta+2][phi-1].get_statistics1(); 
 	     calib_data[sd][eta+2][phi-1].get_average_amp1(&e,&e_rms);
	     calib_data[sd][eta+2][phi-1].get_average_time1(&t,&t_rms);
             xmlFileCalib<<"       <DATA>\n";
             xmlFileCalib<<"          <NUMBER_OF_EVENTS_USED>"<<Statistic<<"</NUMBER_OF_EVENTS_USED>\n";
             xmlFileCalib<<"          <SIGNAL_MEAN>"<<e<<"</SIGNAL_MEAN>\n";
             xmlFileCalib<<"          <SIGNAL_RMS>"<<e_rms<<"</SIGNAL_RMS>\n";
             xmlFileCalib<<"          <TIME_MEAN>"<<t<<"</TIME_MEAN>\n";
             xmlFileCalib<<"          <TIME_RMS>"<<t_rms<<"</TIME_RMS>\n";
             xmlFileCalib<<"          <CHANNEL_STATUS_WORD>"<<Status<<"</CHANNEL_STATUS_WORD>\n";
             xmlFileCalib<<"          <CHANNEL_OBJECTNAME>HcalDetId</CHANNEL_OBJECTNAME>\n";
             xmlFileCalib<<"             <SUBDET>"<<subdet<<"</SUBDET>\n";
             xmlFileCalib<<"             <IETA>"<<eta<<"</IETA>\n";
             xmlFileCalib<<"             <IPHI>"<<phi<<"</IPHI>\n";
             xmlFileCalib<<"             <DEPTH>"<<0<<"</DEPTH>\n";
             xmlFileCalib<<"             <TYPE>0</TYPE>\n";
             xmlFileCalib<<"       </DATA>\n";	
         }
      }
      /////////////////////////////////
      xmlFileCalib<<"  </DATA_SET>\n";
      xmlFileCalib<<"</ROOT>\n";
      xmlFileCalib.close();

      sprintf(str,"zip %s.zip %s %s",xmlName.c_str(),xmlName.c_str(),xmlNameCalib.c_str());
      system(str);
      sprintf(str,"rm -f %s %s",xmlName.c_str(),xmlNameCalib.c_str());
      system(str);
      sprintf(str,"mv -f %s.zip %s",xmlName.c_str(),XmlFilePath.c_str());
      system(str);

   }
   for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++)   hb_data[i][j][k].reset1();
   for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++)   he_data[i][j][k].reset1();
   for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++)   ho_data[i][j][k].reset1();
   for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++)   hf_data[i][j][k].reset1();
   for(int i=1;i<=4;i++)for(int j=-2;j<=2;j++)for(int k=1;k<=72;k++)calib_data[i][j][k].reset1();
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
         if(strcmp(subdet,"CALIB_HB")==0) calib_data[1][Eta+2][Phi-1].set_reference(amp,rms,time,time_rms);
         if(strcmp(subdet,"CALIB_HE")==0) calib_data[2][Eta+2][Phi-1].set_reference(amp,rms,time,time_rms);
         if(strcmp(subdet,"CALIB_HO")==0) calib_data[3][Eta+2][Phi-1].set_reference(amp,rms,time,time_rms);
         if(strcmp(subdet,"CALIB_HF")==0) calib_data[4][Eta+2][Phi-1].set_reference(amp,rms,time,time_rms);
     }
      f->Close();
      IsReference=true;
} 
void HcalDetDiagLaserMonitor::LoadDataset(){
double amp,rms,time,time_rms;
int Eta,Phi,Depth,Statistic;
char subdet[10];
TFile *f;
      if(DatasetName.size()==0) return;
      createHTMLonly=true;
      if(gSystem->AccessPathName(DatasetName.c_str())) return;
      f = new TFile(DatasetName.c_str(),"READ");
 
      if(!f->IsOpen()) return ;

      TTree*  t=0;
      t=(TTree*)f->Get("HCAL Laser data");
      if(!t) return;
      t->SetBranchAddress("Subdet",   subdet);
      t->SetBranchAddress("eta",      &Eta);
      t->SetBranchAddress("phi",      &Phi);
      t->SetBranchAddress("depth",    &Depth);
      t->SetBranchAddress("amp",      &amp);
      t->SetBranchAddress("rms",      &rms);
      t->SetBranchAddress("time",     &time);
      t->SetBranchAddress("time_rms", &time_rms);
      t->SetBranchAddress("statistic",&Statistic);
      for(int ievt=0;ievt<t->GetEntries();ievt++){
         t->GetEntry(ievt);
	 if(strcmp(subdet,"HB")==0){ nHB++;
            hb_data[Eta+42][Phi-1][Depth-1].set_data(amp,rms,time,time_rms);
            hb_data[Eta+42][Phi-1][Depth-1].set_statistics(Statistic);
            hb_data[Eta+42][Phi-1][Depth-1].set_statistics1(Statistic);
	 }
         if(strcmp(subdet,"HE")==0){ nHE++; 
            he_data[Eta+42][Phi-1][Depth-1].set_data(amp,rms,time,time_rms);
            he_data[Eta+42][Phi-1][Depth-1].set_statistics(Statistic);
            he_data[Eta+42][Phi-1][Depth-1].set_statistics1(Statistic);
	 }
         if(strcmp(subdet,"HO")==0){ nHO++;
            ho_data[Eta+42][Phi-1][Depth-1].set_data(amp,rms,time,time_rms);
            ho_data[Eta+42][Phi-1][Depth-1].set_statistics(Statistic);
            ho_data[Eta+42][Phi-1][Depth-1].set_statistics1(Statistic);
	 }
         if(strcmp(subdet,"HF")==0){ nHF++;
            hf_data[Eta+42][Phi-1][Depth-1].set_data(amp,rms,time,time_rms);
            hf_data[Eta+42][Phi-1][Depth-1].set_statistics(Statistic);
            hf_data[Eta+42][Phi-1][Depth-1].set_statistics1(Statistic);
        }
        if(strcmp(subdet,"CALIB_HB")==0){
            calib_data[1][Eta+2][Phi-1].set_data(amp,rms,time,time_rms);
            calib_data[1][Eta+2][Phi-1].set_statistics(Statistic);
            calib_data[1][Eta+2][Phi-1].set_statistics1(Statistic);
        }
        if(strcmp(subdet,"CALIB_HE")==0){
            calib_data[2][Eta+2][Phi-1].set_data(amp,rms,time,time_rms);
            calib_data[2][Eta+2][Phi-1].set_statistics(Statistic);
            calib_data[2][Eta+2][Phi-1].set_statistics1(Statistic);
        }
        if(strcmp(subdet,"CALIB_HO")==0){
            calib_data[3][Eta+2][Phi-1].set_data(amp,rms,time,time_rms);
            calib_data[3][Eta+2][Phi-1].set_statistics(Statistic);
            calib_data[3][Eta+2][Phi-1].set_statistics1(Statistic);
        }
        if(strcmp(subdet,"CALIB_HF")==0){
            calib_data[4][Eta+2][Phi-1].set_data(amp,rms,time,time_rms);
            calib_data[4][Eta+2][Phi-1].set_statistics(Statistic);
            calib_data[4][Eta+2][Phi-1].set_statistics1(Statistic); 
        }
      }
      TObjString *STR1=(TObjString *)f->Get("run number");
      if(STR1){ int run; sscanf(STR1->String(),"%i",&run); meRUN_->Fill(run); run_number=run;}

      TObjString *STR2=(TObjString *)f->Get("Total events processed");
      if(STR2){ int events; sscanf(STR2->String(),"%i",&events); meEVT_->Fill(events); ievt_=events;}

      TObjString *STR3=(TObjString *)f->Get("Dataset number");
      if(STR3){ int ds; sscanf(STR3->String(),"%i",&ds); dataset_seq_number=ds;}
      f->Close(); 
} 

void HcalDetDiagLaserMonitor::SaveRaddamData(){
float adc_range[20]={14,28,40,52,67,132,202,262,322,397,722,1072,1372,1672,2047,3672,5422,6922,8422,10297};
int   adc_bins[20]={1,2,3,4,5,5,10,15,20,25,25,50,75,100,125,125,250,375,500,625};
char str[100];
      TF1 *fitFunc = new TF1("fitFunc","gaus");
      if(fitFunc==0) return;
      for(int i=0;i<56;i++){
          float sum1=0,sum2=0,n=0;
          S1[i]=S2[i]=0;
          for(int j=0;j<128;j++){
            sum1+=(adc2fC[j]-2.5)*Raddam_data[i].s1_adc[j];
            sum2+=(adc2fC[j]-2.5)*Raddam_data[i].s2_adc[j];
            n+=Raddam_data[i].s1_adc[j];
          }
          if(n<100) continue;
          sum1=sum1/n;
          sum2=sum2/n;
          int N=0;
          int Ws1=1,Ws2=1;
          for(N=1;N<19;N++) if(sum1>adc_range[N-1] && sum1<adc_range[N]) break;
          Ws1=adc_bins[N+1];
          for(N=1;N<19;N++) if(sum2>adc_range[N-1] && sum2<adc_range[N]) break;
          Ws2=adc_bins[N+1];
          sprintf(str,"Raddam(%i,%i,%i) S1",RADDAM_CH[i].eta,RADDAM_CH[i].phi,RADDAM_CH[i].depth);
          S1[i]=new TH1F(str,str,10000/Ws1,0,10000);
          sprintf(str,"Raddam(%i,%i,%i) S2",RADDAM_CH[i].eta,RADDAM_CH[i].phi,RADDAM_CH[i].depth);
          S2[i]=new TH1F(str,str,10000/Ws1,0,10000);
          for(int j=0;j<128;j++){
            S1[i]->Fill(adc2fC[j]-2.5,Raddam_data[i].s1_adc[j]);
            S2[i]->Fill(adc2fC[j]-2.5,Raddam_data[i].s2_adc[j]); 
          }
          double parm[3];
          S1[i]->Fit("fitFunc");
          S1[i]->GetFunction("fitFunc")->GetParameters(parm);
          Raddam_data[i].S1MEAN=S1[i]->GetMean();
          Raddam_data[i].S1RMS=S1[i]->GetRMS();
          Raddam_data[i].S1FITMEAN=parm[1];
          Raddam_data[i].S1FITMEANER=S1[i]->GetFunction("fitFunc")->GetParError(1);
          Raddam_data[i].S1FITSIGMA=parm[2];
          Raddam_data[i].S1CHI2=S1[i]->GetFunction("fitFunc")->GetChisquare();
          Raddam_data[i].S1NDF=S1[i]->GetFunction("fitFunc")->GetNDF();
          Raddam_data[i].S1BINWIDTH=Ws1;
          S2[i]->Fit("fitFunc");
          S2[i]->GetFunction("fitFunc")->GetParameters(parm);
          Raddam_data[i].S2MEAN=S2[i]->GetMean();
          Raddam_data[i].S2RMS=S2[i]->GetRMS();
          Raddam_data[i].S2FITMEAN=parm[1];
          Raddam_data[i].S2FITMEANER=S2[i]->GetFunction("fitFunc")->GetParError(1);
          Raddam_data[i].S2FITSIGMA=parm[2];
          Raddam_data[i].S2CHI2=S2[i]->GetFunction("fitFunc")->GetChisquare();
          Raddam_data[i].S2NDF=S2[i]->GetFunction("fitFunc")->GetNDF();
          Raddam_data[i].S2BINWIDTH=Ws2;
      }
      if(XmlFilePath.size()>0){
          char TIME[40];
          Long_t t; t=time(0); strftime(TIME,30,"%F %T",localtime(&t));      
          //create XML file
          if(!Overwrite){
             sprintf(str,"HcalDetDiagRaddam_%i_%i.xml",run_number,dataset_seq_number);
          }else{
             sprintf(str,"HcalDetDiagRaddam.xml");
          }
          std::string xmlName=str;
          ofstream xmlFile;
          xmlFile.open(xmlName.c_str());
          xmlFile<<"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>\n";
          xmlFile<<"<ROOT>\n";
          xmlFile<<"  <HEADER>\n";
          xmlFile<<"    <TYPE>\n";
          xmlFile<<"      <EXTENSION_TABLE_NAME>HCAL_RADDAM</EXTENSION_TABLE_NAME>\n";
          xmlFile<<"      <NAME>HCAL Raddam</NAME>\n";
          xmlFile<<"    </TYPE>\n";
          xmlFile<<"    <!-- run details -->\n";
          xmlFile<<"    <RUN>\n";
          xmlFile<<"      <RUN_TYPE>TEST LOCAL-RUN</RUN_TYPE>\n";
          xmlFile<<"      <RUN_NUMBER>"<<run_number<<"</RUN_NUMBER>\n";
          xmlFile<<"      <RUN_BEGIN_TIMESTAMP>"<<TIME<<"</RUN_BEGIN_TIMESTAMP>\n";
          xmlFile<<"      <COMMENT_DESCRIPTION>hcal raddam data</COMMENT_DESCRIPTION>\n";
          xmlFile<<"      <LOCATION>P5</LOCATION>\n";
          xmlFile<<"      <INITIATED_BY_USER>dma</INITIATED_BY_USER>\n";
          xmlFile<<"    </RUN>\n";
          xmlFile<<"  </HEADER>\n";
          xmlFile<<"  <DATA_SET>\n";
          xmlFile<<"     <COMMENT_DESCRIPTION>Test Raddam data</COMMENT_DESCRIPTION>\n";
          xmlFile<<"     <CREATE_TIMESTAMP>"<<TIME<<"</CREATE_TIMESTAMP>\n";
          xmlFile<<"     <CREATED_BY_USER>dma</CREATED_BY_USER>\n";
          xmlFile<<"     <VERSION>Test_Version_1</VERSION>\n";
 
          for(int i=0;i<56;i++){
             xmlFile<<"     <DATA>\n";
             xmlFile<<"        <SUBDET>HF</SUBDET>\n";
             xmlFile<<"        <IETA>"<<RADDAM_CH[i].eta<<"</IETA>\n";
             xmlFile<<"        <IPHI>"<<RADDAM_CH[i].phi<<"</IPHI>\n";
             xmlFile<<"        <DEPTH>"<<RADDAM_CH[i].depth<<"</DEPTH>\n";

             xmlFile<<"        <TOTEVNT>"<<Raddam_data[i].TOTEVNT<<"</TOTEVNT>\n";
             xmlFile<<"        <CUT1EVNT>"<<Raddam_data[i].CUT1EVNT<<"</CUT1EVNT>\n";
             xmlFile<<"        <CUT2EVNT>"<<Raddam_data[i].CUT2EVNT<<"</CUT2EVNT>\n";

             xmlFile<<"        <S1MEAN>"<<Raddam_data[i].S1MEAN <<"</S1MEAN>\n";
             xmlFile<<"        <S1RMS>"<<Raddam_data[i].S1RMS <<"</S1RMS>\n";
             xmlFile<<"        <S1FITMEAN>"<<Raddam_data[i].S1FITMEAN <<"</S1FITMEAN>\n";
             xmlFile<<"        <S1FITMEANER>"<<Raddam_data[i].S1FITMEANER <<"</S1FITMEANER>\n";
             xmlFile<<"        <S1FITSIGMA>"<<Raddam_data[i].S1FITSIGMA <<"</S1FITSIGMA>\n";
             xmlFile<<"        <S1CHI2>"<<Raddam_data[i].S1CHI2 <<"</S1CHI2>\n";
             xmlFile<<"        <S1NDF>"<<Raddam_data[i].S1NDF <<"</S1NDF>\n";
             xmlFile<<"        <S1BINWIDTH>"<<Raddam_data[i].S1BINWIDTH <<"</S1BINWIDTH>\n";

             xmlFile<<"        <S2MEAN>"<<Raddam_data[i].S2MEAN <<"</S2MEAN>\n";
             xmlFile<<"        <S2RMS>"<<Raddam_data[i].S2RMS <<"</S2RMS>\n";
             xmlFile<<"        <S2FITMEAN>"<<Raddam_data[i].S2FITMEAN <<"</S2FITMEAN>\n";
             xmlFile<<"        <S2FITMEANER>"<<Raddam_data[i].S2FITMEANER <<"</S2FITMEANER>\n";
             xmlFile<<"        <S2FITSIGMA>"<<Raddam_data[i].S2FITSIGMA <<"</S2FITSIGMA>\n";
             xmlFile<<"        <S2CHI2>"<<Raddam_data[i].S2CHI2 <<"</S2CHI2>\n";
             xmlFile<<"        <S2NDF>"<<Raddam_data[i].S2NDF <<"</S2NDF>\n";
             xmlFile<<"        <S2BINWIDTH>"<<Raddam_data[i].S2BINWIDTH <<"</S2BINWIDTH>\n";
             xmlFile<<"    </DATA>\n";
          }
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
      if(OutputFilePath.size()>0){
         if(!Overwrite){
            sprintf(str,"%sHcalDetDiagRaddamData_run%06i_%i.root",OutputFilePath.c_str(),run_number,dataset_seq_number);
         }else{
            sprintf(str,"%sHcalDetDiagRaddamData.root",OutputFilePath.c_str());
         }
         TFile *theFile = new TFile(str, "RECREATE");
         if(!theFile->IsOpen()) return;
         theFile->cd();
         for(int i=0;i<56;i++){
            if(S1[i]!=0)S1[i]->Write();
            if(S2[i]!=0)S2[i]->Write();
         }
         theFile->Write();
         theFile->Close();
      } 
}

void HcalDetDiagLaserMonitor::endRun(const edm::Run& run, const edm::EventSetup& c) {
    if(RaddamRun){
       SaveRaddamData();
    }
    if((LocalRun || !Online_ || createHTMLonly) && ievt_>10){
       fillHistos(HcalBarrel);
       fillHistos(HcalOuter);
       fillHistos(HcalForward);
       fillProblems(HcalBarrel);
       fillProblems(HcalEndcap);
       fillProblems(HcalOuter); 
       fillProblems(HcalForward); 
       if(!RaddamRun)SaveReference();
    }
}

void HcalDetDiagLaserMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c){}
void HcalDetDiagLaserMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c){}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDetDiagLaserMonitor);
