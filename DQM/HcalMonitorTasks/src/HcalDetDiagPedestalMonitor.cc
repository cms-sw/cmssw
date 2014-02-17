// -*- C++ -*-
//
// Package:    HcalDetDiagPedestalMonitor
// Class:      HcalDetDiagPedestalMonitor
// 
/**\class HcalDetDiagPedestalMonitor HcalDetDiagPedestalMonitor.cc DQM/HcalDetDiagPedestalMonitor/src/HcalDetDiagPedestalMonitor.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Dmitry Vishnevskiy,591 R-013,+41227674265,
//         Created:  Tue Mar  9 12:59:18 CET 2010
// $Id: HcalDetDiagPedestalMonitor.cc,v 1.21 2012/08/30 21:48:48 wdd Exp $
//
//
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQM/HcalMonitorTasks/interface/HcalBaseMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include <math.h>
// this is to retrieve HCAL digi's
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
// to retrive trigger information (local runs only)
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalLogicalMapGenerator.h"
#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"

#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include <iostream>
#include <fstream>

class HcalDetDiagPedestalData{
public: 
   HcalDetDiagPedestalData(){ 
             reset();
	     IsReference=false;
	     status=n=0;
             ds_ped=ds_rms=-100; 
             nChecks=nMissing=nBadPed=nBadRms=nUnstable=0;  
	  }
   void   reset(){
             for(int i=0;i<128;i++) adc[i]=0; 
	     overflow=0;
          }	  
   void   add_statistics(unsigned int val){
             if(val<128){ adc[val&0x7F]++; n++;}else overflow++;    
	  }
   void   set_reference(float val,float rms){
             ref_ped=val; ref_rms=rms;
	     IsReference=true;
          }	  
   void   set_data(float val,float rms){
             ds_ped=val; ds_rms=rms;
          }	  
   void   change_status(int val){
             status|=val;
          }	  
   int    get_status(){
             return status;
          }	  
   bool   get_reference(double *val,double *rms){
             *val=ref_ped; *rms=ref_rms;
	     return IsReference;
          }	  
   bool   get_average(double *ave,double *rms){
             if(ds_ped>-100){ *ave=ds_ped;  *rms=ds_rms;  return true;}
             double Sum=0,nSum=0; 
	     int from,to,max=adc[0],maxi=0;
	     for(int i=1;i<128;i++) if(adc[i]>max){ max=adc[i]; maxi=i;} 
	     from=0; to=maxi+6; if(to>127) to=127;
             for(int i=from;i<=to;i++){
                Sum+=i*adc[i];
	        nSum+=adc[i];
             } 
             if(nSum>0) *ave=Sum/nSum; else return false;
             Sum=0;
             for(int i=from;i<=to;i++) Sum+=adc[i]*(i-*ave)*(i-*ave);
             *rms=sqrt(Sum/nSum);
             return true; 
          }
   int    get_statistics(){
	     return n;
	  } 
   void   set_statistics(int stat){
	     n=stat;
	  } 
   int    get_overflow(){
             return overflow;
          }   
   float   nChecks;
   float   nMissing;
   float   nUnstable;
   float   nBadPed;
   float   nBadRms;
private:   
   int   adc[128];
   int   overflow;
   bool  IsReference;
   float ref_ped;
   float ref_rms;
   float ds_ped;
   float ds_rms;
   int   n;
   int   status;
};


class HcalDetDiagPedestalMonitor : public HcalBaseDQMonitor {
   public:
      explicit HcalDetDiagPedestalMonitor(const edm::ParameterSet&);
      ~HcalDetDiagPedestalMonitor();


   private:
      void SaveReference();
      void LoadReference();
      void LoadDataset();

      void CheckStatus();
      void fillHistos();

      const HcalElectronicsMap  *emap;
      edm::InputTag inputLabelDigi_;
      edm::InputTag  inputLabelRawData_;

      void beginRun(const edm::Run& run, const edm::EventSetup& c);  
      void endRun(const edm::Run& run, const edm::EventSetup& c);
      void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c) ;
      void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c);
      void analyze(const edm::Event&, const edm::EventSetup&);

      int         ievt_;
      int         run_number;
      int         dataset_seq_number;
      bool        IsReference;
      bool        LocalRun;
      int         nHB,nHE,nHO,nHF;

      double      HBMeanTreshold;
      double      HBRmsTreshold;
      double      HEMeanTreshold;
      double      HERmsTreshold;
      double      HOMeanTreshold;
      double      HORmsTreshold;
      double      HFMeanTreshold;
      double      HFRmsTreshold;

      std::string ReferenceData;
      std::string ReferenceRun;
      std::string OutputFilePath;
      std::string XmlFilePath;

      // to create html from processed dataset
      std::string DatasetName;
      std::string htmlOutputPath;
      bool createHTMLonly;

      std::string prefixME_;
      bool        Online_;
      bool        Overwrite;

      int nTS_HBHE,nTS_HO,nTS_HF;
      MonitorElement *meEVT_,*meRUN_,*htmlFolder;
      MonitorElement *RefRun_;
      MonitorElement *PedestalsAve4HB;
      MonitorElement *PedestalsAve4HE;
      MonitorElement *PedestalsAve4HO;
      MonitorElement *PedestalsAve4HF;
      MonitorElement *PedestalsAve4Simp;
 
      MonitorElement *PedestalsAve4HBref;
      MonitorElement *PedestalsAve4HEref;
      MonitorElement *PedestalsAve4HOref;
      MonitorElement *PedestalsAve4HFref;
      MonitorElement *PedestalsRmsHB;
      MonitorElement *PedestalsRmsHE;
      MonitorElement *PedestalsRmsHO;
      MonitorElement *PedestalsRmsHF;
      MonitorElement *PedestalsRmsSimp;
  
      MonitorElement *PedestalsRmsHBref;
      MonitorElement *PedestalsRmsHEref;
      MonitorElement *PedestalsRmsHOref;
      MonitorElement *PedestalsRmsHFref;
  
      MonitorElement *Pedestals2DRmsHBHEHF;
      MonitorElement *Pedestals2DRmsHO;
      MonitorElement *Pedestals2DHBHEHF;
      MonitorElement *Pedestals2DHO;
      MonitorElement *Pedestals2DErrorHBHEHF;
      MonitorElement *Pedestals2DErrorHO;
 
      EtaPhiHists* ProblemCellsByDepth_missing;
      EtaPhiHists* ProblemCellsByDepth_unstable;
      EtaPhiHists* ProblemCellsByDepth_badped;
      EtaPhiHists* ProblemCellsByDepth_badrms;

      EtaPhiHists* ProblemCellsByDepth_missing_val;
      EtaPhiHists* ProblemCellsByDepth_unstable_val;
      EtaPhiHists* ProblemCellsByDepth_badped_val;
      EtaPhiHists* ProblemCellsByDepth_badrms_val;
      std::vector<std::string> problemnames_;
 
      HcalDetDiagPedestalData hb_data[85][72][4][4];
      HcalDetDiagPedestalData he_data[85][72][4][4];
      HcalDetDiagPedestalData ho_data[85][72][4][4];
      HcalDetDiagPedestalData hf_data[85][72][4][4];

      std::map<unsigned int, int> KnownBadCells_;

      edm::InputTag hcalTBTriggerDataTag_;
};

HcalDetDiagPedestalMonitor::HcalDetDiagPedestalMonitor(const edm::ParameterSet& iConfig) :
  hcalTBTriggerDataTag_(iConfig.getParameter<edm::InputTag>("hcalTBTriggerDataTag"))
{
  ievt_=-1;
  emap=0;
  dataset_seq_number=1;
  run_number=-1;
  IsReference=false;
  LocalRun=false;
  nHB=nHE=nHO=nHF=0;
  createHTMLonly=false;
  nTS_HBHE=nTS_HO=nTS_HF=0;
  inputLabelDigi_    = iConfig.getUntrackedParameter<edm::InputTag>("digiLabel");
  inputLabelRawData_ = iConfig.getUntrackedParameter<edm::InputTag>("rawDataLabel");
  ReferenceData    = iConfig.getUntrackedParameter<std::string>("PedestalReferenceData" ,"");
  OutputFilePath   = iConfig.getUntrackedParameter<std::string>("OutputFilePath", "");
  DatasetName      = iConfig.getUntrackedParameter<std::string>("PedestalDatasetName", "");
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

  HBMeanTreshold   = iConfig.getUntrackedParameter<double>("HBMeanPedestalTreshold" , 0.2);
  HBRmsTreshold    = iConfig.getUntrackedParameter<double>("HBRmsPedestalTreshold"  , 0.3);
  HEMeanTreshold   = iConfig.getUntrackedParameter<double>("HEMeanPedestalTreshold" , 0.2);
  HERmsTreshold    = iConfig.getUntrackedParameter<double>("HERmsPedestalTreshold"  , 0.3);
  HOMeanTreshold   = iConfig.getUntrackedParameter<double>("HOMeanPedestalTreshold" , 0.2);
  HORmsTreshold    = iConfig.getUntrackedParameter<double>("HORmsPedestalTreshold"  , 0.3);
  HFMeanTreshold   = iConfig.getUntrackedParameter<double>("HFMeanPedestalTreshold" , 0.2);
  HFRmsTreshold    = iConfig.getUntrackedParameter<double>("HFRmsPedestalTreshold"  , 0.3);
}
HcalDetDiagPedestalMonitor::~HcalDetDiagPedestalMonitor(){}

void HcalDetDiagPedestalMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c){
  edm::ESHandle<HcalDbService> conditions_;
  c.get<HcalDbRecord>().get(conditions_);
  emap=conditions_->getHcalMapping();
  
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
 

  HcalBaseDQMonitor::setup();
  if (!dbe_) return;
  std::string name;

  dbe_->setCurrentFolder(subdir_);   
  meEVT_ = dbe_->bookInt("HcalDetDiagPedestalMonitor Event Number");
  meRUN_ = dbe_->bookInt("HcalDetDiagPedestalMonitor Run Number");

  ReferenceRun="UNKNOWN";
  LoadReference();
  LoadDataset();
  dbe_->setCurrentFolder(subdir_);
  RefRun_= dbe_->bookString("HcalDetDiagPedestalMonitor Reference Run",ReferenceRun);
  if(DatasetName.size()>0 && createHTMLonly){
     char str[200]; sprintf(str,"%sHcalDetDiagPedestalData_run%i_%i/",htmlOutputPath.c_str(),run_number,dataset_seq_number);
     htmlFolder=dbe_->bookString("HcalDetDiagPedestalMonitor HTML folder",str);
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

  ProblemCellsByDepth_missing = new EtaPhiHists();
  ProblemCellsByDepth_missing->setup(dbe_," Problem Missing Channels");
  for(unsigned int i=0;i<ProblemCellsByDepth_missing->depth.size();i++)
          problemnames_.push_back(ProblemCellsByDepth_missing->depth[i]->getName());
  ProblemCellsByDepth_unstable = new EtaPhiHists();
  ProblemCellsByDepth_unstable->setup(dbe_," Problem Unstable Channels");
  for(unsigned int i=0;i<ProblemCellsByDepth_unstable->depth.size();i++)
          problemnames_.push_back(ProblemCellsByDepth_unstable->depth[i]->getName());
  ProblemCellsByDepth_badped = new EtaPhiHists();
  ProblemCellsByDepth_badped->setup(dbe_," Problem Bad Pedestal Value");
  for(unsigned int i=0;i<ProblemCellsByDepth_badped->depth.size();i++)
          problemnames_.push_back(ProblemCellsByDepth_badped->depth[i]->getName());
  ProblemCellsByDepth_badrms = new EtaPhiHists();
  ProblemCellsByDepth_badrms->setup(dbe_," Problem Bad Rms Value");
  for(unsigned int i=0;i<ProblemCellsByDepth_badrms->depth.size();i++)
          problemnames_.push_back(ProblemCellsByDepth_badrms->depth[i]->getName());

  dbe_->setCurrentFolder(subdir_+"Summary Plots");
  name="HB Pedestal Distribution (average over 4 caps)";           PedestalsAve4HB = dbe_->book1D(name,name,200,0,6);
  name="HE Pedestal Distribution (average over 4 caps)";           PedestalsAve4HE = dbe_->book1D(name,name,200,0,6);
  name="HO Pedestal Distribution (average over 4 caps)";           PedestalsAve4HO = dbe_->book1D(name,name,200,0,6);
  name="HF Pedestal Distribution (average over 4 caps)";           PedestalsAve4HF = dbe_->book1D(name,name,200,0,6);
  name="SIPM Pedestal Distribution (average over 4 caps)";         PedestalsAve4Simp = dbe_->book1D(name,name,200,5,15);
     
  name="HB Pedestal-Reference Distribution (average over 4 caps)"; PedestalsAve4HBref= dbe_->book1D(name,name,1500,-3,3);
  name="HE Pedestal-Reference Distribution (average over 4 caps)"; PedestalsAve4HEref= dbe_->book1D(name,name,1500,-3,3);
  name="HO Pedestal-Reference Distribution (average over 4 caps)"; PedestalsAve4HOref= dbe_->book1D(name,name,1500,-3,3);
  name="HF Pedestal-Reference Distribution (average over 4 caps)"; PedestalsAve4HFref= dbe_->book1D(name,name,1500,-3,3);
    
  name="HB Pedestal RMS Distribution (individual cap)";            PedestalsRmsHB = dbe_->book1D(name,name,200,0,2);
  name="HE Pedestal RMS Distribution (individual cap)";            PedestalsRmsHE = dbe_->book1D(name,name,200,0,2);
  name="HO Pedestal RMS Distribution (individual cap)";            PedestalsRmsHO = dbe_->book1D(name,name,200,0,2);
  name="HF Pedestal RMS Distribution (individual cap)";            PedestalsRmsHF = dbe_->book1D(name,name,200,0,2);
  name="SIPM Pedestal RMS Distribution (individual cap)";          PedestalsRmsSimp = dbe_->book1D(name,name,200,0,4);
     
  name="HB Pedestal_rms-Reference_rms Distribution";               PedestalsRmsHBref = dbe_->book1D(name,name,1500,-3,3);
  name="HE Pedestal_rms-Reference_rms Distribution";               PedestalsRmsHEref = dbe_->book1D(name,name,1500,-3,3);
  name="HO Pedestal_rms-Reference_rms Distribution";               PedestalsRmsHOref = dbe_->book1D(name,name,1500,-3,3);
  name="HF Pedestal_rms-Reference_rms Distribution";               PedestalsRmsHFref = dbe_->book1D(name,name,1500,-3,3);
     
  name="HBHEHF pedestal mean map";       Pedestals2DHBHEHF      = dbe_->book2D(name,name,87,-43,43,74,0,73);
  name="HO pedestal mean map";           Pedestals2DHO          = dbe_->book2D(name,name,33,-16,16,74,0,73);
  name="HBHEHF pedestal rms map";        Pedestals2DRmsHBHEHF   = dbe_->book2D(name,name,87,-43,43,74,0,73);
  name="HO pedestal rms map";            Pedestals2DRmsHO       = dbe_->book2D(name,name,33,-16,16,74,0,73);
  name="HBHEHF pedestal problems map";   Pedestals2DErrorHBHEHF = dbe_->book2D(name,name,87,-43,43,74,0,73);
  name="HO pedestal problems map";       Pedestals2DErrorHO     = dbe_->book2D(name,name,33,-16,16,74,0,73);

  Pedestals2DHBHEHF->setAxisRange(1,5,3);
  Pedestals2DHO->setAxisRange(1,5,3);
  Pedestals2DRmsHBHEHF->setAxisRange(0,2,3);
  Pedestals2DRmsHO->setAxisRange(0,2,3);

  Pedestals2DHBHEHF->setAxisTitle("i#eta",1);
  Pedestals2DHBHEHF->setAxisTitle("i#phi",2);
  Pedestals2DHO->setAxisTitle("i#eta",1);
  Pedestals2DHO->setAxisTitle("i#phi",2);
  Pedestals2DRmsHBHEHF->setAxisTitle("i#eta",1);
  Pedestals2DRmsHBHEHF->setAxisTitle("i#phi",2);
  Pedestals2DRmsHO->setAxisTitle("i#eta",1);
  Pedestals2DRmsHO->setAxisTitle("i#phi",2);
  Pedestals2DErrorHBHEHF->setAxisTitle("i#eta",1);
  Pedestals2DErrorHBHEHF->setAxisTitle("i#phi",2);
  Pedestals2DErrorHO->setAxisTitle("i#eta",1);
  Pedestals2DErrorHO->setAxisTitle("i#phi",2);
  PedestalsAve4HB->setAxisTitle("ADC counts",1);
  PedestalsAve4HE->setAxisTitle("ADC counts",1);
  PedestalsAve4HO->setAxisTitle("ADC counts",1);
  PedestalsAve4HF->setAxisTitle("ADC counts",1);
  PedestalsAve4Simp->setAxisTitle("ADC counts",1);
  PedestalsAve4HBref->setAxisTitle("ADC counts",1);
  PedestalsAve4HEref->setAxisTitle("ADC counts",1);
  PedestalsAve4HOref->setAxisTitle("ADC counts",1);
  PedestalsAve4HFref->setAxisTitle("ADC counts",1);
  PedestalsRmsHB->setAxisTitle("ADC counts",1);
  PedestalsRmsHE->setAxisTitle("ADC counts",1);
  PedestalsRmsHO->setAxisTitle("ADC counts",1);
  PedestalsRmsHF->setAxisTitle("ADC counts",1);
  PedestalsRmsSimp->setAxisTitle("ADC counts",1);
  PedestalsRmsHBref->setAxisTitle("ADC counts",1);
  PedestalsRmsHEref->setAxisTitle("ADC counts",1);
  PedestalsRmsHOref->setAxisTitle("ADC counts",1);
  PedestalsRmsHFref->setAxisTitle("ADC counts",1);

  dbe_->setCurrentFolder(subdir_+"Plots for client");
  ProblemCellsByDepth_missing_val = new EtaPhiHists();
  ProblemCellsByDepth_missing_val->setup(dbe_," Missing channels");
  ProblemCellsByDepth_unstable_val = new EtaPhiHists();
  ProblemCellsByDepth_unstable_val->setup(dbe_," Channel instability value");
  ProblemCellsByDepth_badped_val = new EtaPhiHists();
  ProblemCellsByDepth_badped_val->setup(dbe_," Bad Pedestal-Ref Value");
  ProblemCellsByDepth_badrms_val = new EtaPhiHists();
  ProblemCellsByDepth_badrms_val->setup(dbe_," Bad Rms-ref Value");
}



void HcalDetDiagPedestalMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
 if(createHTMLonly) return;
  HcalBaseDQMonitor::analyze(iEvent, iSetup); // increments counters
int  eta,phi,depth,nTS;
static bool PEDseq;
static int  lastPEDorbit,nChecksPED;
   if(ievt_==-1){ ievt_=0; PEDseq=false; lastPEDorbit=-1;nChecksPED=0; }
   int orbit=iEvent.orbitNumber();
   meRUN_->Fill(iEvent.id().run());

   bool PedestalEvent=false;
   
   // for local runs 
   edm::Handle<HcalTBTriggerData> trigger_data;
   iEvent.getByLabel(hcalTBTriggerDataTag_, trigger_data);
   if(trigger_data.isValid()){
     if((trigger_data->triggerWord())==5) PedestalEvent=true;
       LocalRun=true;
   }
  
  if(LocalRun && !PedestalEvent) return; 

  if(!LocalRun && Online_){
      if(PEDseq && (orbit-lastPEDorbit)>(11223*10) && ievt_>500){
         PEDseq=false;
         fillHistos();
         CheckStatus();
         nChecksPED++;
         if(nChecksPED==1 || (nChecksPED>1 && ((nChecksPED-1)%12)==0)){
             SaveReference();
         }
         for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++)for(int l=0;l<4;l++) hb_data[i][j][k][l].reset();
         for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++)for(int l=0;l<4;l++) he_data[i][j][k][l].reset();
         for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++)for(int l=0;l<4;l++) ho_data[i][j][k][l].reset();
         for(int i=0;i<85;i++)for(int j=0;j<72;j++)for(int k=0;k<4;k++)for(int l=0;l<4;l++) hf_data[i][j][k][l].reset();
         ievt_=0;
      }
   }

  
   // Abort Gap pedestals 
   int calibType = -1 ;
   if(LocalRun==false){
       edm::Handle<FEDRawDataCollection> rawdata;
       iEvent.getByLabel(inputLabelRawData_,rawdata);
       //checking FEDs for calibration information
       for (int i=FEDNumbering::MINHCALFEDID;i<=FEDNumbering::MAXHCALFEDID; i++){
         const FEDRawData& fedData = rawdata->FEDData(i) ;
	 if ( fedData.size() < 24 ) continue ;
	 int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ;
	 if ( calibType < 0 )  calibType = value ;
         if(value==hc_Pedestal){   PEDseq=true;  lastPEDorbit=orbit; break;} 
       }
   }
   if(!LocalRun && calibType!=hc_Pedestal) return; 

   ievt_++;
   meEVT_->Fill(ievt_);
   run_number=iEvent.id().run();

   edm::Handle<HBHEDigiCollection> hbhe; 
   iEvent.getByLabel(inputLabelDigi_,hbhe);
   if(hbhe.isValid()){
	 if(hbhe->size()<30 && calibType==hc_Pedestal){
             ievt_--;
             meEVT_->Fill(ievt_);
             return;	 
	 }
         for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
             if(nTS>8) nTS=8;
	     if(nTS<8 && nTS>=4) nTS=4;
             nTS_HBHE=nTS;
	     if(digi->id().subdet()==HcalBarrel){
		for(int i=0;i<nTS;i++) hb_data[eta+42][phi-1][depth-1][digi->sample(i).capid()].add_statistics(digi->sample(i).adc());
	     }	 
             if(digi->id().subdet()==HcalEndcap){
		for(int i=0;i<nTS;i++) he_data[eta+42][phi-1][depth-1][digi->sample(i).capid()].add_statistics(digi->sample(i).adc());
	     }
         }   
   }
   edm::Handle<HODigiCollection> ho; 
   iEvent.getByLabel(inputLabelDigi_,ho);
   if(ho.isValid()){
         for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     if(nTS>8) nTS=8;
	     if(nTS<8 && nTS>=4) nTS=4;
             nTS_HO=nTS;
             for(int i=0;i<nTS;i++) ho_data[eta+42][phi-1][depth-1][digi->sample(i).capid()].add_statistics(digi->sample(i).adc());
         }   
   }
   edm::Handle<HFDigiCollection> hf;
   iEvent.getByLabel(inputLabelDigi_,hf);
   if(hf.isValid()){
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     if(nTS>8) nTS=8;
	     if(nTS<8 && nTS>=4) nTS=4;
             nTS_HF=nTS;
	     for(int i=0;i<nTS;i++) hf_data[eta+42][phi-1][depth-1][digi->sample(i).capid()].add_statistics(digi->sample(i).adc());
         }   
   }
}

void HcalDetDiagPedestalMonitor::fillHistos(){
   PedestalsRmsHB->Reset();
   PedestalsAve4HB->Reset();
   PedestalsRmsHE->Reset();
   PedestalsAve4HE->Reset();
   PedestalsRmsHO->Reset();
   PedestalsAve4HO->Reset();
   PedestalsRmsHF->Reset();
   PedestalsAve4HF->Reset();
   PedestalsRmsSimp->Reset();
   PedestalsAve4Simp->Reset();
   Pedestals2DRmsHBHEHF->Reset();
   Pedestals2DRmsHO->Reset();
   Pedestals2DHBHEHF->Reset();
   Pedestals2DHO->Reset();
   // HBHEHF summary map
   for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++){ 
      double PED=0,RMS=0,nped=0,nrms=0,ave=0,rms=0;
      for(int depth=1;depth<=3;depth++){
         if(hb_data[eta+42][phi-1][depth-1][0].get_statistics()>100){
	    hb_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++;
	    hb_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
	    hb_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
	    hb_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
         }
         if(he_data[eta+42][phi-1][depth-1][0].get_statistics()>100){
	    he_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++;
	    he_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
	    he_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
	    he_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
         }
         if(hf_data[eta+42][phi-1][depth-1][0].get_statistics()>100){
	    hf_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++;
	    hf_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
	    hf_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
	    hf_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
         }
      }
      if(nped>0) Pedestals2DHBHEHF->Fill(eta,phi,PED/nped);
      if(nrms>0) Pedestals2DRmsHBHEHF->Fill(eta,phi,RMS/nrms); 
      if(nped>0 && abs(eta)>20) Pedestals2DHBHEHF->Fill(eta,phi+1,PED/nped);
      if(nrms>0 && abs(eta)>20) Pedestals2DRmsHBHEHF->Fill(eta,phi+1,RMS/nrms); 
   }
   // HO summary map
   for(int eta=-10;eta<=15;eta++) for(int phi=1;phi<=72;phi++){
      if(eta>10 && !isSiPM(eta,phi,4)) continue;
      double PED=0,RMS=0,nped=0,nrms=0,ave=0,rms=0;
      if(ho_data[eta+42][phi-1][4-1][0].get_statistics()>100){
	 ho_data[eta+42][phi-1][4-1][0].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++;
	 ho_data[eta+42][phi-1][4-1][1].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
	 ho_data[eta+42][phi-1][4-1][2].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
	 ho_data[eta+42][phi-1][4-1][3].get_average(&ave,&rms); PED+=ave; nped++; RMS+=rms; nrms++; 
      }
      if(nped>0) Pedestals2DHO->Fill(eta,phi,PED/nped);
      if(nrms>0) Pedestals2DRmsHO->Fill(eta,phi,RMS/nrms); 
   }
   // HB histograms
   for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
      if(hb_data[eta+42][phi-1][depth-1][0].get_statistics()>100){
          double ave=0,rms=0,sum=0;
	  hb_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); sum+=ave; PedestalsRmsHB->Fill(rms);
	  hb_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); sum+=ave; PedestalsRmsHB->Fill(rms);
	  hb_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); sum+=ave; PedestalsRmsHB->Fill(rms);
	  hb_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); sum+=ave; PedestalsRmsHB->Fill(rms);
	  PedestalsAve4HB->Fill(sum/4.0);
      }
   } 
   // HE histograms
   for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=3;depth++){
      if(he_data[eta+42][phi-1][depth-1][0].get_statistics()>100){
          double ave=0,rms=0,sum=0;
	  he_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); sum+=ave; PedestalsRmsHE->Fill(rms);
	  he_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); sum+=ave; PedestalsRmsHE->Fill(rms);
	  he_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); sum+=ave; PedestalsRmsHE->Fill(rms);
	  he_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); sum+=ave; PedestalsRmsHE->Fill(rms);
	  PedestalsAve4HE->Fill(sum/4.0);
      }
   } 
   // HO histograms
   for(int eta=-10;eta<=15;eta++) for(int phi=1;phi<=72;phi++) for(int depth=4;depth<=4;depth++){
      if(eta>10 && !isSiPM(eta,phi,4)) continue;
      if(ho_data[eta+42][phi-1][depth-1][0].get_statistics()>100){
          double ave=0,rms=0,sum=0;
	  if((eta>=11 && eta<=15 && phi>=59 && phi<=70) || (eta>=5 && eta<=10 && phi>=47 && phi<=58)){
	     ho_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); sum+=ave; PedestalsRmsSimp->Fill(rms);
	     ho_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); sum+=ave; PedestalsRmsSimp->Fill(rms);
	     ho_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); sum+=ave; PedestalsRmsSimp->Fill(rms);
	     ho_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); sum+=ave; PedestalsRmsSimp->Fill(rms);
	     PedestalsAve4Simp->Fill(sum/4.0);	  
	  }else{
	     ho_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); sum+=ave; PedestalsRmsHO->Fill(rms);
	     ho_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); sum+=ave; PedestalsRmsHO->Fill(rms);
	     ho_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); sum+=ave; PedestalsRmsHO->Fill(rms);
	     ho_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); sum+=ave; PedestalsRmsHO->Fill(rms);
	     PedestalsAve4HO->Fill(sum/4.0);
	  }
      }
   } 
   // HF histograms
   for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
      if(hf_data[eta+42][phi-1][depth-1][0].get_statistics()>100){
          double ave=0,rms=0,sum=0;
	  hf_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); sum+=ave; PedestalsRmsHF->Fill(rms);
	  hf_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); sum+=ave; PedestalsRmsHF->Fill(rms);
	  hf_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); sum+=ave; PedestalsRmsHF->Fill(rms);
	  hf_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); sum+=ave; PedestalsRmsHF->Fill(rms);
	  PedestalsAve4HF->Fill(sum/4.0);
      }
   } 
} 
void HcalDetDiagPedestalMonitor::CheckStatus(){
   for(int i=0;i<4;i++){
      ProblemCellsByDepth_missing->depth[i]->Reset();
      ProblemCellsByDepth_unstable->depth[i]->Reset();
      ProblemCellsByDepth_badped->depth[i]->Reset();
      ProblemCellsByDepth_badrms->depth[i]->Reset();
   }
   PedestalsRmsHBref->Reset();
   PedestalsAve4HBref->Reset();
   PedestalsRmsHEref->Reset();
   PedestalsAve4HEref->Reset();
   PedestalsRmsHOref->Reset();
   PedestalsAve4HOref->Reset();
   PedestalsRmsHFref->Reset();
   PedestalsAve4HFref->Reset();
     
   Pedestals2DErrorHBHEHF->Reset();
   Pedestals2DErrorHO->Reset();

   if(emap==0) return;
   
   std::vector <HcalElectronicsId> AllElIds = emap->allElectronicsIdPrecision();
   for (std::vector <HcalElectronicsId>::iterator eid = AllElIds.begin(); eid != AllElIds.end(); eid++) {
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
     int sd=detid.subdetId();

     if(sd==HcalBarrel){
          int ovf=hb_data[eta+42][phi-1][depth-1][0].get_overflow();
	  int stat=hb_data[eta+42][phi-1][depth-1][0].get_statistics()+ovf;
	  double status=0;
	  double ped[4],rms[4],ped_ref[4],rms_ref[4]; 
          hb_data[eta+42][phi-1][depth-1][0].nChecks++;
	  if(stat==0){ 
              status=1;
              int e=CalcEtaBin(sd,eta,depth)+1; 
              hb_data[eta+42][phi-1][depth-1][0].nMissing++;
              double val=hb_data[eta+42][phi-1][depth-1][0].nMissing/hb_data[eta+42][phi-1][depth-1][0].nChecks;
              ProblemCellsByDepth_missing->depth[depth-1]->setBinContent(e,phi,val);
              ProblemCellsByDepth_missing_val->depth[depth-1]->setBinContent(e,phi,1);
          }
          if(status) hb_data[eta+42][phi-1][depth-1][0].change_status(1); 
	  if(stat>0 && stat!=(ievt_*2)){ if(nTS_HBHE==8) status=(double)stat/(double)(ievt_*2); else status=(double)stat/(double)(ievt_);
	      if(status<0.995){ 
                int e=CalcEtaBin(sd,eta,depth)+1; 
                hb_data[eta+42][phi-1][depth-1][0].nUnstable++;
                double val=hb_data[eta+42][phi-1][depth-1][0].nUnstable/hb_data[eta+42][phi-1][depth-1][0].nChecks;
                ProblemCellsByDepth_unstable->depth[depth-1]->setBinContent(e,phi,val);
                ProblemCellsByDepth_unstable_val->depth[depth-1]->setBinContent(e,phi,status);
	        hb_data[eta+42][phi-1][depth-1][0].change_status(2);
	      }
	  }
	  if(hb_data[eta+42][phi-1][depth-1][0].get_reference(&ped_ref[0],&rms_ref[0]) 
	                                                 && hb_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0])){
	     hb_data[eta+42][phi-1][depth-1][1].get_reference(&ped_ref[1],&rms_ref[1]);
	     hb_data[eta+42][phi-1][depth-1][2].get_reference(&ped_ref[2],&rms_ref[2]);
	     hb_data[eta+42][phi-1][depth-1][3].get_reference(&ped_ref[3],&rms_ref[3]);
	     hb_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
	     hb_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
	     hb_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
	     double ave=(ped[0]+ped[1]+ped[2]+ped[3])/4.0; 
	     double ave_ref=(ped_ref[0]+ped_ref[1]+ped_ref[2]+ped_ref[3])/4.0; 
	     double deltaPed=ave-ave_ref; PedestalsAve4HBref->Fill(deltaPed); if(deltaPed<0) deltaPed=-deltaPed;
	     double deltaRms=rms[0]-rms_ref[0]; PedestalsRmsHBref->Fill(deltaRms); if(deltaRms<0) deltaRms=-deltaRms;
	     for(int i=1;i<4;i++){
	        double tmp=rms[i]-rms_ref[i]; PedestalsRmsHBref->Fill(tmp); 
		if(fabs(tmp)>fabs(deltaRms)) deltaRms=tmp;
	     }
	     if(deltaPed>HBMeanTreshold){ 
                 int e=CalcEtaBin(sd,eta,depth)+1; 
                 hb_data[eta+42][phi-1][depth-1][0].nBadPed++;
                 double val=hb_data[eta+42][phi-1][depth-1][0].nBadPed/hb_data[eta+42][phi-1][depth-1][0].nChecks;
                 ProblemCellsByDepth_badped_val->depth[depth-1]->setBinContent(e,phi,ave-ave_ref);
                 ProblemCellsByDepth_badped->depth[depth-1]->setBinContent(e,phi,val);
                 Pedestals2DErrorHBHEHF->Fill(eta,phi,1);
             }
	     if(deltaRms>HBRmsTreshold){  
                 int e=CalcEtaBin(sd,eta,depth)+1; 
                 hb_data[eta+42][phi-1][depth-1][0].nBadRms++;
                 double val=hb_data[eta+42][phi-1][depth-1][0].nBadRms/hb_data[eta+42][phi-1][depth-1][0].nChecks;
                 ProblemCellsByDepth_badrms->depth[depth-1]->setBinContent(e,phi,val);
                 ProblemCellsByDepth_badrms_val->depth[depth-1]->setBinContent(e,phi,deltaRms);
                 Pedestals2DErrorHBHEHF->Fill(eta,phi,1);
             }
	  } 
      }
      if(sd==HcalEndcap){
          int ovf=he_data[eta+42][phi-1][depth-1][0].get_overflow();
	  int stat=he_data[eta+42][phi-1][depth-1][0].get_statistics()+ovf;
	  double status=0; 
	  double ped[4],rms[4],ped_ref[4],rms_ref[4]; 
          he_data[eta+42][phi-1][depth-1][0].nChecks++;
	  if(stat==0){ 
              status=1;
              int e=CalcEtaBin(sd,eta,depth)+1; 
              he_data[eta+42][phi-1][depth-1][0].nMissing++;
              double val=he_data[eta+42][phi-1][depth-1][0].nMissing/he_data[eta+42][phi-1][depth-1][0].nChecks;
              ProblemCellsByDepth_missing->depth[depth-1]->setBinContent(e,phi,val);
              ProblemCellsByDepth_missing_val->depth[depth-1]->setBinContent(e,phi,1);
          }
	  if(status) he_data[eta+42][phi-1][depth-1][0].change_status(1); 
	  if(stat>0 && stat!=(ievt_*2)){ if(nTS_HBHE==8) status=(double)stat/(double)(ievt_*2); else status=(double)stat/(double)(ievt_);
	     if(status<0.995){ 
                int e=CalcEtaBin(sd,eta,depth)+1; 
                he_data[eta+42][phi-1][depth-1][0].nUnstable++;
                double val=he_data[eta+42][phi-1][depth-1][0].nUnstable/he_data[eta+42][phi-1][depth-1][0].nChecks;
                ProblemCellsByDepth_unstable->depth[depth-1]->setBinContent(e,phi,val);
                ProblemCellsByDepth_unstable_val->depth[depth-1]->setBinContent(e,phi,status);
	        he_data[eta+42][phi-1][depth-1][0].change_status(2); 
	     }
	  }
	  if(he_data[eta+42][phi-1][depth-1][0].get_reference(&ped_ref[0],&rms_ref[0]) 
	                                                 && he_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0])){
	     he_data[eta+42][phi-1][depth-1][1].get_reference(&ped_ref[1],&rms_ref[1]);
	     he_data[eta+42][phi-1][depth-1][2].get_reference(&ped_ref[2],&rms_ref[2]);
	     he_data[eta+42][phi-1][depth-1][3].get_reference(&ped_ref[3],&rms_ref[3]);
	     he_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
	     he_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
	     he_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
	     double ave=(ped[0]+ped[1]+ped[2]+ped[3])/4.0; 
	     double ave_ref=(ped_ref[0]+ped_ref[1]+ped_ref[2]+ped_ref[3])/4.0; 
	     double deltaPed=ave-ave_ref; PedestalsAve4HEref->Fill(deltaPed); if(deltaPed<0) deltaPed=-deltaPed;
	     double deltaRms=rms[0]-rms_ref[0]; PedestalsRmsHEref->Fill(deltaRms); if(deltaRms<0) deltaRms=-deltaRms;
	     for(int i=1;i<4;i++){
	        double tmp=rms[i]-rms_ref[i]; PedestalsRmsHEref->Fill(tmp); 
		if(fabs(tmp)>fabs(deltaRms)) deltaRms=tmp;
	     }
	     if(deltaPed>HEMeanTreshold){
                 int e=CalcEtaBin(sd,eta,depth)+1; 
                 he_data[eta+42][phi-1][depth-1][0].nBadPed++;
                 double val=he_data[eta+42][phi-1][depth-1][0].nBadPed/he_data[eta+42][phi-1][depth-1][0].nChecks;
                 ProblemCellsByDepth_badped->depth[depth-1]->setBinContent(e,phi,val);
                 ProblemCellsByDepth_badped_val->depth[depth-1]->setBinContent(e,phi,ave-ave_ref);
                 Pedestals2DErrorHBHEHF->Fill(eta,phi,1);
             }
	     if(deltaRms>HERmsTreshold){ 
                 int e=CalcEtaBin(sd,eta,depth)+1; 
                 he_data[eta+42][phi-1][depth-1][0].nBadRms++;
                 double val=he_data[eta+42][phi-1][depth-1][0].nBadRms/he_data[eta+42][phi-1][depth-1][0].nChecks;
                 ProblemCellsByDepth_badrms->depth[depth-1]->setBinContent(e,phi,val);
                 ProblemCellsByDepth_badrms_val->depth[depth-1]->setBinContent(e,phi,deltaRms);
                 Pedestals2DErrorHBHEHF->Fill(eta,phi,1);
             }
	  } 
      }
      if(sd==HcalOuter){
          int ovf=ho_data[eta+42][phi-1][depth-1][0].get_overflow(); 
	  int stat=ho_data[eta+42][phi-1][depth-1][0].get_statistics()+ovf;
	  double status=0; 
	  double ped[4],rms[4],ped_ref[4],rms_ref[4]; 
          ho_data[eta+42][phi-1][depth-1][0].nChecks++;
	  if(stat==0){ 
              status=1; 
              int e=CalcEtaBin(sd,eta,depth)+1; 
              ho_data[eta+42][phi-1][depth-1][0].nMissing++;
              double val=ho_data[eta+42][phi-1][depth-1][0].nMissing/ho_data[eta+42][phi-1][depth-1][0].nChecks;
              ProblemCellsByDepth_missing->depth[depth-1]->setBinContent(e,phi,val);
              ProblemCellsByDepth_missing_val->depth[depth-1]->setBinContent(e,phi,1);
          }
	  if(status) ho_data[eta+42][phi-1][depth-1][0].change_status(1); 
	  if(stat>0 && stat!=(ievt_*2)){ if(nTS_HO==8) status=(double)stat/(double)(ievt_*2); else status=(double)stat/(double)(ievt_);
	     if(status<0.995){ 
                int e=CalcEtaBin(sd,eta,depth)+1; 
                ho_data[eta+42][phi-1][depth-1][0].nUnstable++;
                double val=ho_data[eta+42][phi-1][depth-1][0].nUnstable/ho_data[eta+42][phi-1][depth-1][0].nChecks;
                ProblemCellsByDepth_unstable->depth[depth-1]->setBinContent(e,phi,val);
                ProblemCellsByDepth_unstable_val->depth[depth-1]->setBinContent(e,phi,status);
	        ho_data[eta+42][phi-1][depth-1][0].change_status(2); 
	     }
	  }
	  if(ho_data[eta+42][phi-1][depth-1][0].get_reference(&ped_ref[0],&rms_ref[0]) 
	                                                 && ho_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0])){
	     ho_data[eta+42][phi-1][depth-1][1].get_reference(&ped_ref[1],&rms_ref[1]);
	     ho_data[eta+42][phi-1][depth-1][2].get_reference(&ped_ref[2],&rms_ref[2]);
	     ho_data[eta+42][phi-1][depth-1][3].get_reference(&ped_ref[3],&rms_ref[3]);
	     ho_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
	     ho_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
	     ho_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
	     
	     double THRESTHOLD=HORmsTreshold;
	     if((eta>=11 && eta<=15 && phi>=59 && phi<=70) || (eta>=5 && eta<=10 && phi>=47 && phi<=58))THRESTHOLD*=2; 
	     double ave=(ped[0]+ped[1]+ped[2]+ped[3])/4.0; 
	     double ave_ref=(ped_ref[0]+ped_ref[1]+ped_ref[2]+ped_ref[3])/4.0; 
	     double deltaPed=ave-ave_ref; PedestalsAve4HOref->Fill(deltaPed);if(deltaPed<0) deltaPed=-deltaPed;
	     double deltaRms=rms[0]-rms_ref[0]; PedestalsRmsHOref->Fill(deltaRms); if(deltaRms<0) deltaRms=-deltaRms;
	     for(int i=1;i<4;i++){
	        double tmp=rms[i]-rms_ref[i]; PedestalsRmsHOref->Fill(tmp); 
		if(fabs(tmp)>fabs(deltaRms)) deltaRms=tmp;
	     }
	     if(deltaPed>HOMeanTreshold){ 
                 int e=CalcEtaBin(sd,eta,depth)+1; 
                 ho_data[eta+42][phi-1][depth-1][0].nBadPed++;
                 double val=ho_data[eta+42][phi-1][depth-1][0].nBadPed/ho_data[eta+42][phi-1][depth-1][0].nChecks;
                 ProblemCellsByDepth_badped->depth[depth-1]->setBinContent(e,phi,val);
                 ProblemCellsByDepth_badped_val->depth[depth-1]->setBinContent(e,phi,ave-ave_ref);
                 Pedestals2DErrorHO->Fill(eta,phi,1);
             }
	     if(deltaRms>THRESTHOLD){ 
                 int e=CalcEtaBin(sd,eta,depth)+1; 
                 ho_data[eta+42][phi-1][depth-1][0].nBadRms++;
                 double val=ho_data[eta+42][phi-1][depth-1][0].nBadRms/ho_data[eta+42][phi-1][depth-1][0].nChecks;
                 ProblemCellsByDepth_badrms->depth[depth-1]->setBinContent(e,phi,val);
                 ProblemCellsByDepth_badrms_val->depth[depth-1]->setBinContent(e,phi,deltaRms);
                 Pedestals2DErrorHO->Fill(eta,phi,1);
             }
	  } 
      }
      if(sd==HcalForward){
          int ovf=hf_data[eta+42][phi-1][depth-1][0].get_overflow();
	  int stat=hf_data[eta+42][phi-1][depth-1][0].get_statistics()+ovf;
	  double status=0; 
	  double ped[4],rms[4],ped_ref[4],rms_ref[4]; 
          hf_data[eta+42][phi-1][depth-1][0].nChecks++;
	  if(stat==0){ 
             status=1;                    
             int e=CalcEtaBin(sd,eta,depth)+1; 
             hf_data[eta+42][phi-1][depth-1][0].nMissing++;
             double val=hf_data[eta+42][phi-1][depth-1][0].nMissing/hf_data[eta+42][phi-1][depth-1][0].nChecks;
             ProblemCellsByDepth_missing->depth[depth-1]->setBinContent(e,phi,val);
             ProblemCellsByDepth_missing_val->depth[depth-1]->setBinContent(e,phi,1);
          }
	  if(status) hf_data[eta+42][phi-1][depth-1][0].change_status(1); 
	  if(stat>0 && stat!=(ievt_*2)){ if(nTS_HF==8) status=(double)stat/(double)(ievt_*2); else status=(double)stat/(double)(ievt_); 
	     if(status<0.995){ 
                int e=CalcEtaBin(sd,eta,depth)+1; 
                hf_data[eta+42][phi-1][depth-1][0].nUnstable++;
                double val=hf_data[eta+42][phi-1][depth-1][0].nUnstable/hf_data[eta+42][phi-1][depth-1][0].nChecks;
                ProblemCellsByDepth_unstable->depth[depth-1]->setBinContent(e,phi,val);
                ProblemCellsByDepth_unstable_val->depth[depth-1]->setBinContent(e,phi,status);
	        hf_data[eta+42][phi-1][depth-1][0].change_status(2); 
	     }
	  }
	  if(hf_data[eta+42][phi-1][depth-1][0].get_reference(&ped_ref[0],&rms_ref[0]) 
	                                                 && hf_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0])){
	     hf_data[eta+42][phi-1][depth-1][1].get_reference(&ped_ref[1],&rms_ref[1]);
	     hf_data[eta+42][phi-1][depth-1][2].get_reference(&ped_ref[2],&rms_ref[2]);
	     hf_data[eta+42][phi-1][depth-1][3].get_reference(&ped_ref[3],&rms_ref[3]);
	     hf_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
	     hf_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
	     hf_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
	     double ave=(ped[0]+ped[1]+ped[2]+ped[3])/4.0; 
	     double ave_ref=(ped_ref[0]+ped_ref[1]+ped_ref[2]+ped_ref[3])/4.0; 
	     double deltaPed=ave-ave_ref; PedestalsAve4HFref->Fill(deltaPed); if(deltaPed<0) deltaPed=-deltaPed;
	     double deltaRms=rms[0]-rms_ref[0]; PedestalsRmsHFref->Fill(deltaRms); if(deltaRms<0) deltaRms=-deltaRms;
	     for(int i=1;i<4;i++){
	        double tmp=rms[i]-rms_ref[i]; PedestalsRmsHFref->Fill(tmp); 
		if(fabs(tmp)>fabs(deltaRms)) deltaRms=tmp;
	     }
	     if(deltaPed>HFMeanTreshold){ 
                 int e=CalcEtaBin(sd,eta,depth)+1; 
                 hf_data[eta+42][phi-1][depth-1][0].nBadPed++;
                 double val=hf_data[eta+42][phi-1][depth-1][0].nBadPed/hf_data[eta+42][phi-1][depth-1][0].nChecks;
                 ProblemCellsByDepth_badped->depth[depth-1]->setBinContent(e,phi,val);
                 ProblemCellsByDepth_badped_val->depth[depth-1]->setBinContent(e,phi,ave-ave_ref);
                 Pedestals2DErrorHBHEHF->Fill(eta,phi,1);
             }
	     if(deltaRms>HFRmsTreshold){ 
                 int e=CalcEtaBin(sd,eta,depth)+1; 
                 hf_data[eta+42][phi-1][depth-1][0].nBadRms++;
                 double val=hf_data[eta+42][phi-1][depth-1][0].nBadRms/hf_data[eta+42][phi-1][depth-1][0].nChecks;
                 ProblemCellsByDepth_badrms->depth[depth-1]->setBinContent(e,phi,val);
                 ProblemCellsByDepth_badrms_val->depth[depth-1]->setBinContent(e,phi,deltaRms);
                 Pedestals2DErrorHBHEHF->Fill(eta,phi,1);
             }
	  } 
      }
   }
}

void HcalDetDiagPedestalMonitor::endRun(const edm::Run& run, const edm::EventSetup& c){
 
    if((LocalRun || !Online_ || createHTMLonly) && ievt_>=100){
       fillHistos();
       CheckStatus(); 
       SaveReference();
    }
}

void HcalDetDiagPedestalMonitor::SaveReference(){
double ped[4],rms[4];
int    Eta,Phi,Depth,Statistic,Status=0;
char   Subdet[10],str[500];
   if(OutputFilePath.size()>0){
       if(!Overwrite){
          sprintf(str,"%sHcalDetDiagPedestalData_run%06i_%i.root",OutputFilePath.c_str(),run_number,dataset_seq_number);
       }else{
          sprintf(str,"%sHcalDetDiagPedestalData.root",OutputFilePath.c_str());
       }
       TFile *theFile = new TFile(str, "RECREATE");
       if(!theFile->IsOpen()) return;
       theFile->cd();
       sprintf(str,"%d",run_number);              TObjString run(str);    run.Write("run number");
       sprintf(str,"%d",ievt_);                   TObjString events(str); events.Write("Total events processed");
       sprintf(str,"%d",dataset_seq_number);      TObjString dsnum(str);  dsnum.Write("Dataset number");
       Long_t t; t=time(0); strftime(str,30,"%F %T",localtime(&t)); TObjString tm(str);  tm.Write("Dataset creation time");

       TTree *tree   =new TTree("HCAL Pedestal data","HCAL Pedestal data");
       if(tree==0)   return;
       tree->Branch("Subdet",   &Subdet,         "Subdet/C");
       tree->Branch("eta",      &Eta,            "Eta/I");
       tree->Branch("phi",      &Phi,            "Phi/I");
       tree->Branch("depth",    &Depth,          "Depth/I");
       tree->Branch("statistic",&Statistic,      "Statistic/I");
       tree->Branch("status",   &Status,         "Status/I");
       tree->Branch("cap0_ped", &ped[0],         "cap0_ped/D");
       tree->Branch("cap0_rms", &rms[0],         "cap0_rms/D");
       tree->Branch("cap1_ped", &ped[1],         "cap1_ped/D");
       tree->Branch("cap1_rms", &rms[1],         "cap1_rms/D");
       tree->Branch("cap2_ped", &ped[2],         "cap2_ped/D");
       tree->Branch("cap2_rms", &rms[2],         "cap2_rms/D");
       tree->Branch("cap3_ped", &ped[3],         "cap3_ped/D");
       tree->Branch("cap3_rms", &rms[3],         "cap3_rms/D");
       sprintf(Subdet,"HB");
       for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
          if((Statistic=hb_data[eta+42][phi-1][depth-1][0].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hb_data[eta+42][phi-1][depth-1][0].get_status();
	     hb_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0]);
	     hb_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
	     hb_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
	     hb_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"HE");
       for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=3;depth++){
         if((Statistic=he_data[eta+42][phi-1][depth-1][0].get_statistics())>100){
            Eta=eta; Phi=phi; Depth=depth;
	    Status=he_data[eta+42][phi-1][depth-1][0].get_status();
	    he_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0]);
	    he_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
	    he_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
	    he_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
	    tree->Fill();
         }
      } 
      sprintf(Subdet,"HO");
      for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++) for(int depth=4;depth<=4;depth++){
         if((Statistic=ho_data[eta+42][phi-1][depth-1][0].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=ho_data[eta+42][phi-1][depth-1][0].get_status();
	     ho_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0]);
	     ho_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
	     ho_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
	     ho_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
	     tree->Fill();
         }
      } 
      sprintf(Subdet,"HF");
      for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
         if((Statistic=hf_data[eta+42][phi-1][depth-1][0].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hf_data[eta+42][phi-1][depth-1][0].get_status();
	     hf_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0]);
	     hf_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
	     hf_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
	     hf_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
	     tree->Fill();
         }
      }
      theFile->Write();
      theFile->Close();
   }

   if(XmlFilePath.size()>0){
      //create XML file
      if(!Overwrite){
         sprintf(str,"HcalDetDiagPedestals_%i_%i.xml",run_number,dataset_seq_number);
      }else{
         sprintf(str,"HcalDetDiagPedestals.xml");
      }
      printf("%s\n",str);
      std::string xmlName=str;
      ofstream xmlFile;
      xmlFile.open(xmlName.c_str());

      xmlFile<<"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
      xmlFile<<"<ROOT>\n";
      xmlFile<<"  <HEADER>\n";
      xmlFile<<"    <HINTS mode='only-det-root'/>\n";
      xmlFile<<"    <TYPE>\n";
      xmlFile<<"      <EXTENSION_TABLE_NAME>HCAL_DETMON_PEDESTALS_V1</EXTENSION_TABLE_NAME>\n";
      xmlFile<<"      <NAME>HCAL Pedestals [abort gap global]</NAME>\n";
      xmlFile<<"    </TYPE>\n";
      xmlFile<<"    <!-- run details -->\n";
      xmlFile<<"    <RUN>\n";
      xmlFile<<"      <RUN_TYPE>GLOBAL-RUN</RUN_TYPE>\n";
      xmlFile<<"      <RUN_NUMBER>"<<run_number<<"</RUN_NUMBER>\n";
      xmlFile<<"      <RUN_BEGIN_TIMESTAMP>2009-01-01 00:00:00</RUN_BEGIN_TIMESTAMP>\n";
      xmlFile<<"      <COMMENT_DESCRIPTION>hcal ped data</COMMENT_DESCRIPTION>\n";
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
      
         double ped[4]={0,0,0,0},rms[4]={0,0,0,0};
         if(detid.subdetId()==HcalBarrel){
             subdet="HB";
             Statistic=hb_data[eta+42][phi-1][depth-1][0].get_statistics();
             Status   =hb_data[eta+42][phi-1][depth-1][0].get_status();
             hb_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0]);
             hb_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
             hb_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
             hb_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
         }else if(detid.subdetId()==HcalEndcap){
             subdet="HE";
             he_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0]);
             he_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
             he_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
             he_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
             Statistic=he_data[eta+42][phi-1][depth-1][0].get_statistics();
             Status   =he_data[eta+42][phi-1][depth-1][0].get_status();
	 }else if(detid.subdetId()==HcalForward){
             subdet="HF";
             hf_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0]);
             hf_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
             hf_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
             hf_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
             Statistic=hf_data[eta+42][phi-1][depth-1][0].get_statistics();
             Status   =hf_data[eta+42][phi-1][depth-1][0].get_status();
	 }else if(detid.subdetId()==HcalOuter){
             subdet="HO";
             ho_data[eta+42][phi-1][depth-1][0].get_average(&ped[0],&rms[0]);
             ho_data[eta+42][phi-1][depth-1][1].get_average(&ped[1],&rms[1]);
             ho_data[eta+42][phi-1][depth-1][2].get_average(&ped[2],&rms[2]);
             ho_data[eta+42][phi-1][depth-1][3].get_average(&ped[3],&rms[3]);
             Statistic=ho_data[eta+42][phi-1][depth-1][0].get_statistics();
             Status   =ho_data[eta+42][phi-1][depth-1][0].get_status();
         }else continue;
         xmlFile<<"       <DATA>\n";
         xmlFile<<"          <NUMBER_OF_EVENTS_USED>"<<Statistic<<"</NUMBER_OF_EVENTS_USED>\n";
         xmlFile<<"          <MEAN0>"<<ped[0]<<"</MEAN0>\n";
         xmlFile<<"          <MEAN1>"<<ped[1]<<"</MEAN1>\n";
         xmlFile<<"          <MEAN2>"<<ped[2]<<"</MEAN2>\n";
         xmlFile<<"          <MEAN3>"<<ped[3]<<"</MEAN3>\n";
         xmlFile<<"          <RMS0>"<<rms[0]<<"</RMS0>\n";
         xmlFile<<"          <RMS1>"<<rms[1]<<"</RMS1>\n";
         xmlFile<<"          <RMS2>"<<rms[2]<<"</RMS2>\n";
         xmlFile<<"          <RMS3>"<<rms[3]<<"</RMS3>\n";
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
   dataset_seq_number++;
}

void HcalDetDiagPedestalMonitor::LoadReference(){
  double ped[4],rms[4];
  int Eta,Phi,Depth;
  char subdet[10];
  TFile *f;
  if(gSystem->AccessPathName(ReferenceData.c_str())) return;
  f = new TFile(ReferenceData.c_str(),"READ");
  if(!f->IsOpen()) return ;
  TObjString *STR=(TObjString *)f->Get("run number");
  
  if(STR){ std::string Ref(STR->String()); ReferenceRun=Ref;}
  
  TTree*  t=(TTree*)f->Get("HCAL Pedestal data");
  if(!t) return;
  t->SetBranchAddress("Subdet",   subdet);
  t->SetBranchAddress("eta",      &Eta);
  t->SetBranchAddress("phi",      &Phi);
  t->SetBranchAddress("depth",    &Depth);
  t->SetBranchAddress("cap0_ped", &ped[0]);
  t->SetBranchAddress("cap0_rms", &rms[0]);
  t->SetBranchAddress("cap1_ped", &ped[1]);
  t->SetBranchAddress("cap1_rms", &rms[1]);
  t->SetBranchAddress("cap2_ped", &ped[2]);
  t->SetBranchAddress("cap2_rms", &rms[2]);
  t->SetBranchAddress("cap3_ped", &ped[3]);
  t->SetBranchAddress("cap3_rms", &rms[3]);
  for(int ievt=0;ievt<t->GetEntries();ievt++){
    t->GetEntry(ievt);
    if(strcmp(subdet,"HB")==0){
      hb_data[Eta+42][Phi-1][Depth-1][0].set_reference(ped[0],rms[0]);
      hb_data[Eta+42][Phi-1][Depth-1][1].set_reference(ped[1],rms[1]);
      hb_data[Eta+42][Phi-1][Depth-1][2].set_reference(ped[2],rms[2]);
      hb_data[Eta+42][Phi-1][Depth-1][3].set_reference(ped[3],rms[3]);
    }
    if(strcmp(subdet,"HE")==0){
      he_data[Eta+42][Phi-1][Depth-1][0].set_reference(ped[0],rms[0]);
      he_data[Eta+42][Phi-1][Depth-1][1].set_reference(ped[1],rms[1]);
      he_data[Eta+42][Phi-1][Depth-1][2].set_reference(ped[2],rms[2]);
      he_data[Eta+42][Phi-1][Depth-1][3].set_reference(ped[3],rms[3]);
    }
    if(strcmp(subdet,"HO")==0){
      ho_data[Eta+42][Phi-1][Depth-1][0].set_reference(ped[0],rms[0]);
      ho_data[Eta+42][Phi-1][Depth-1][1].set_reference(ped[1],rms[1]);
      ho_data[Eta+42][Phi-1][Depth-1][2].set_reference(ped[2],rms[2]);
      ho_data[Eta+42][Phi-1][Depth-1][3].set_reference(ped[3],rms[3]);
    }
    if(strcmp(subdet,"HF")==0){
      hf_data[Eta+42][Phi-1][Depth-1][0].set_reference(ped[0],rms[0]);
      hf_data[Eta+42][Phi-1][Depth-1][1].set_reference(ped[1],rms[1]);
      hf_data[Eta+42][Phi-1][Depth-1][2].set_reference(ped[2],rms[2]);
      hf_data[Eta+42][Phi-1][Depth-1][3].set_reference(ped[3],rms[3]);
    }
  }
  f->Close();
  IsReference=true;
} 

void HcalDetDiagPedestalMonitor::LoadDataset(){
  double ped[4],rms[4];
  int Eta,Phi,Depth,Statistic;
  char subdet[10];
  TFile *f;
  if(DatasetName.size()==0) return;
  createHTMLonly=true;
  if(gSystem->AccessPathName(DatasetName.c_str())) return;
  f = new TFile(DatasetName.c_str(),"READ");
  if(!f->IsOpen()) return ;
 
  TTree*  t=0;
  t=(TTree*)f->Get("HCAL Pedestal data");
  if(!t) return;
  t->SetBranchAddress("Subdet",   subdet);
  t->SetBranchAddress("eta",      &Eta);
  t->SetBranchAddress("phi",      &Phi);
  t->SetBranchAddress("depth",    &Depth);
  t->SetBranchAddress("cap0_ped", &ped[0]);
  t->SetBranchAddress("cap0_rms", &rms[0]);
  t->SetBranchAddress("cap1_ped", &ped[1]);
  t->SetBranchAddress("cap1_rms", &rms[1]);
  t->SetBranchAddress("cap2_ped", &ped[2]);
  t->SetBranchAddress("cap2_rms", &rms[2]);
  t->SetBranchAddress("cap3_ped", &ped[3]);
  t->SetBranchAddress("cap3_rms", &rms[3]);
  t->SetBranchAddress("statistic",&Statistic);

  for(int ievt=0;ievt<t->GetEntries();ievt++){
    t->GetEntry(ievt);
    if(strcmp(subdet,"HB")==0){ nHB++;
      hb_data[Eta+42][Phi-1][Depth-1][0].set_data(ped[0],rms[0]);
      hb_data[Eta+42][Phi-1][Depth-1][1].set_data(ped[1],rms[1]);
      hb_data[Eta+42][Phi-1][Depth-1][2].set_data(ped[2],rms[2]);
      hb_data[Eta+42][Phi-1][Depth-1][3].set_data(ped[3],rms[3]);
      hb_data[Eta+42][Phi-1][Depth-1][0].set_statistics(Statistic);
      hb_data[Eta+42][Phi-1][Depth-1][1].set_statistics(Statistic);
      hb_data[Eta+42][Phi-1][Depth-1][2].set_statistics(Statistic);
      hb_data[Eta+42][Phi-1][Depth-1][3].set_statistics(Statistic);
    }
    if(strcmp(subdet,"HE")==0){ nHE++;
      he_data[Eta+42][Phi-1][Depth-1][0].set_data(ped[0],rms[0]);
      he_data[Eta+42][Phi-1][Depth-1][1].set_data(ped[1],rms[1]);
      he_data[Eta+42][Phi-1][Depth-1][2].set_data(ped[2],rms[2]);
      he_data[Eta+42][Phi-1][Depth-1][3].set_data(ped[3],rms[3]);
      he_data[Eta+42][Phi-1][Depth-1][0].set_statistics(Statistic);
      he_data[Eta+42][Phi-1][Depth-1][1].set_statistics(Statistic);
      he_data[Eta+42][Phi-1][Depth-1][2].set_statistics(Statistic);
      he_data[Eta+42][Phi-1][Depth-1][3].set_statistics(Statistic);
    }
    if(strcmp(subdet,"HO")==0){ nHO++;
      ho_data[Eta+42][Phi-1][Depth-1][0].set_data(ped[0],rms[0]);
      ho_data[Eta+42][Phi-1][Depth-1][1].set_data(ped[1],rms[1]);
      ho_data[Eta+42][Phi-1][Depth-1][2].set_data(ped[2],rms[2]);
      ho_data[Eta+42][Phi-1][Depth-1][3].set_data(ped[3],rms[3]);
      ho_data[Eta+42][Phi-1][Depth-1][0].set_statistics(Statistic);
      ho_data[Eta+42][Phi-1][Depth-1][1].set_statistics(Statistic);
      ho_data[Eta+42][Phi-1][Depth-1][2].set_statistics(Statistic);
      ho_data[Eta+42][Phi-1][Depth-1][3].set_statistics(Statistic);
    }
    if(strcmp(subdet,"HF")==0){ nHF++;
      hf_data[Eta+42][Phi-1][Depth-1][0].set_data(ped[0],rms[0]);
      hf_data[Eta+42][Phi-1][Depth-1][1].set_data(ped[1],rms[1]);
      hf_data[Eta+42][Phi-1][Depth-1][2].set_data(ped[2],rms[2]);
      hf_data[Eta+42][Phi-1][Depth-1][3].set_data(ped[3],rms[3]);
      hf_data[Eta+42][Phi-1][Depth-1][0].set_statistics(Statistic);
      hf_data[Eta+42][Phi-1][Depth-1][1].set_statistics(Statistic);
      hf_data[Eta+42][Phi-1][Depth-1][2].set_statistics(Statistic);
      hf_data[Eta+42][Phi-1][Depth-1][3].set_statistics(Statistic);
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
void HcalDetDiagPedestalMonitor::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c){}
void HcalDetDiagPedestalMonitor::endLuminosityBlock(const edm::LuminosityBlock& lumiSeg,const edm::EventSetup& c){}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDetDiagPedestalMonitor);
