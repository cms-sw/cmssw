#include "DQMServices/Core/interface/MonitorElement.h"
// this is to retrieve HCAL digi's
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
// to retrive trigger information (local runs only)
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
// to retrive GMT information, for cosmic runs muon triggers can be used as pedestal (global runs only)
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
// to retrive trigger desition words, to select pedestal (from hcal point of view) triggers (global runs only)
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "CondFormats/HcalObjects/interface/HcalLogicalMap.h"
#include "DQM/HcalMonitorTasks/interface/HcalBaseDQMonitor.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include <math.h>
#include <iostream>
#include <fstream>


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
             int MaxI=0; double Time=-9999,SumT=0,MaxT=-10;
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

class HcalDetDiagLEDMonitor:public HcalBaseDQMonitor {
public:
  HcalDetDiagLEDMonitor(const edm::ParameterSet& ps); 
  ~HcalDetDiagLEDMonitor(); 

  void beginRun(const edm::Run& run, const edm::EventSetup& c);
  void setup();
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);// const HcalDbService& cond)
  void endRun(const edm::Run& run, const edm::EventSetup& c);
  void reset();
  void cleanup(); 
  void fillHistos();
  int  GetStatistics(){ return ievt_; }
private:
  HcalElectronicsMap*      emap;
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
    return &calib_data[SD][ETA+2][PHI-1];
  };
  int         ievt_;
  int         run_number;
  int         dataset_seq_number;
  bool        IsReference;
  
  double      LEDMeanTreshold;
  double      LEDRmsTreshold;
   
  std::string ReferenceData;
  std::string ReferenceRun;
  std::string OutputFilePath;
  std::string XmlFilePath;

  MonitorElement *meEVT_,*meRUN_;
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

  HcalDetDiagLEDData hb_data[85][72][4];
  HcalDetDiagLEDData he_data[85][72][4];
  HcalDetDiagLEDData ho_data[85][72][4];
  HcalDetDiagLEDData hf_data[85][72][4];
  HcalDetDiagLEDData calib_data[5][5][72];
  
  EtaPhiHists *ChannelsLEDEnergy;
  EtaPhiHists *ChannelsLEDEnergyRef;
  EtaPhiHists *ChannelStatusMissingChannels;
  EtaPhiHists *ChannelStatusUnstableChannels;
  EtaPhiHists *ChannelStatusUnstableLEDsignal;
  EtaPhiHists *ChannelStatusLEDMean;
  EtaPhiHists *ChannelStatusLEDRMS;
  EtaPhiHists *ChannelStatusTimeMean;
  EtaPhiHists *ChannelStatusTimeRMS;

  edm::InputTag digiLabel_;
  edm::InputTag calibDigiLabel_;
  edm::InputTag hcalTBTriggerDataTag_;

  std::map<unsigned int, int> KnownBadCells_;

  void fill_channel_status(std::string subdet,int eta,int phi,int depth,int type,double status);
  void   fill_energy(std::string subdet,int eta,int phi,int depth,double e,int type);
  double get_energy(std::string subdet,int eta,int phi,int depth,int type);
};

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




HcalDetDiagLEDMonitor::HcalDetDiagLEDMonitor(const edm::ParameterSet& ps) :
  hcalTBTriggerDataTag_(ps.getParameter<edm::InputTag>("hcalTBTriggerDataTag"))
{
  ievt_=0;
  dataset_seq_number=1;
  run_number=-1;
  IsReference=false;

  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","DetDiagLEDMonitor_Hcal");
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",false);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

  LEDMeanTreshold  = ps.getUntrackedParameter<double>("LEDMeanTreshold" , 0.1);
  LEDRmsTreshold   = ps.getUntrackedParameter<double>("LEDRmsTreshold"  , 0.1);
  
  ReferenceData    = ps.getUntrackedParameter<std::string>("LEDReferenceData" ,"");
  OutputFilePath   = ps.getUntrackedParameter<std::string>("OutputFilePath", "");
  XmlFilePath      = ps.getUntrackedParameter<std::string>("XmlFilePath", "");

  digiLabel_       = ps.getUntrackedParameter<edm::InputTag>("digiLabel", edm::InputTag("hcalDigis"));
  calibDigiLabel_  = ps.getUntrackedParameter<edm::InputTag>("calibDigiLabel",edm::InputTag("hcalDigis"));

  emap=0;
  needLogicalMap_ = true;
}

HcalDetDiagLEDMonitor::~HcalDetDiagLEDMonitor(){}

void HcalDetDiagLEDMonitor::cleanup(){
  if(dbe_){
    dbe_->setCurrentFolder(subdir_);
    dbe_->removeContents();
    dbe_ = 0;
  }
} 
void HcalDetDiagLEDMonitor::reset(){}

void HcalDetDiagLEDMonitor::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>1) std::cout <<"HcalDetDiagLEDMonitor::beginRun"<<std::endl;
  HcalBaseDQMonitor::beginRun(run,c);

  if (tevt_==0) this->setup(); // set up histograms if they have not been created before
  if (mergeRuns_==false) this->reset();

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

  return;
} // void HcalNDetDiagLEDMonitor::beginRun(...)

void HcalDetDiagLEDMonitor::setup(){
     // Call base class setup
     HcalBaseDQMonitor::setup();
     if (!dbe_) return;

     std::string name;
     dbe_->setCurrentFolder(subdir_);   
     meEVT_ = dbe_->bookInt("HcalDetDiagLEDMonitor Event Number");
     meRUN_ = dbe_->bookInt("HcalDetDiagLEDMonitor Run Number");
     ReferenceRun="UNKNOWN";
     LoadReference();
     dbe_->setCurrentFolder(subdir_);
     RefRun_= dbe_->bookString("HcalDetDiagLEDMonitor Reference Run",ReferenceRun);
     dbe_->setCurrentFolder(subdir_+"Summary Plots");
     
     name="HBHEHO LED Energy Distribution";               Energy         = dbe_->book1D(name,name,200,0,3000);
     name="HBHEHO LED Timing Distribution";               Time           = dbe_->book1D(name,name,200,0,10);
     name="HBHEHO LED Energy RMS_div_Energy Distribution";EnergyRMS      = dbe_->book1D(name,name,200,0,0.2);
     name="HBHEHO LED Timing RMS Distribution";           TimeRMS        = dbe_->book1D(name,name,200,0,0.4);
     name="HF LED Energy Distribution";                   EnergyHF       = dbe_->book1D(name,name,200,0,3000);
     name="HF LED Timing Distribution";                   TimeHF         = dbe_->book1D(name,name,200,0,10);
     name="HF LED Energy RMS_div_Energy Distribution";    EnergyRMSHF    = dbe_->book1D(name,name,200,0,0.5);
     name="HF LED Timing RMS Distribution";               TimeRMSHF      = dbe_->book1D(name,name,200,0,0.4);
     name="LED Energy Corr(PinDiod) Distribution";        EnergyCorr     = dbe_->book1D(name,name,200,0,10);
     name="LED Timing HBHEHF";                            Time2Dhbhehf   = dbe_->book2D(name,name,87,-43,43,74,0,73);
     name="LED Timing HO";                                Time2Dho       = dbe_->book2D(name,name,33,-16,16,74,0,73);
     name="LED Energy HBHEHF";                            Energy2Dhbhehf = dbe_->book2D(name,name,87,-43,43,74,0,73);
     name="LED Energy HO";                                Energy2Dho     = dbe_->book2D(name,name,33,-16,16,74,0,73);

     name="HBP Average over HPD LED Ref";          HBPphi = dbe_->book2D(name,name,180,1,73,400,0,2);
     name="HBM Average over HPD LED Ref";          HBMphi = dbe_->book2D(name,name,180,1,73,400,0,2);
     name="HEP Average over HPD LED Ref";          HEPphi = dbe_->book2D(name,name,180,1,73,400,0,2);
     name="HEM Average over HPD LED Ref";          HEMphi = dbe_->book2D(name,name,180,1,73,400,0,2);
     name="HFP Average over RM LED Ref";           HFPphi = dbe_->book2D(name,name,180,1,37,400,0,2);
     name="HFM Average over RM LED Ref";           HFMphi = dbe_->book2D(name,name,180,1,37,400,0,2);
     name="HO0 Average over HPD LED Ref";          HO0phi = dbe_->book2D(name,name,180,1,49,400,0,2);
     name="HO1P Average over HPD LED Ref";         HO1Pphi= dbe_->book2D(name,name,180,1,49,400,0,2);
     name="HO2P Average over HPD LED Ref";         HO2Pphi= dbe_->book2D(name,name,180,1,49,400,0,2);
     name="HO1M Average over HPD LED Ref";         HO1Mphi= dbe_->book2D(name,name,180,1,49,400,0,2);
     name="HO2M Average over HPD LED Ref";         HO2Mphi= dbe_->book2D(name,name,180,1,49,400,0,2);

     ChannelsLEDEnergy = new EtaPhiHists();
     ChannelsLEDEnergy->setup(dbe_," Channel LED Energy");
     ChannelsLEDEnergyRef = new EtaPhiHists();
     ChannelsLEDEnergyRef->setup(dbe_," Channel LED Energy Reference");
     
     dbe_->setCurrentFolder(subdir_+"channel status");
     ChannelStatusMissingChannels = new EtaPhiHists();
     ChannelStatusMissingChannels->setup(dbe_," Missing Channels");
     ChannelStatusUnstableChannels = new EtaPhiHists();
     ChannelStatusUnstableChannels->setup(dbe_," Unstable Channels");
     ChannelStatusUnstableLEDsignal = new EtaPhiHists();
     ChannelStatusUnstableLEDsignal->setup(dbe_," Unstable LED");
     ChannelStatusLEDMean = new EtaPhiHists();
     ChannelStatusLEDMean->setup(dbe_," LED Mean");
     ChannelStatusLEDRMS = new EtaPhiHists();
     ChannelStatusLEDRMS->setup(dbe_," LED RMS");
     ChannelStatusTimeMean = new EtaPhiHists();
     ChannelStatusTimeMean->setup(dbe_," Time Mean");
     ChannelStatusTimeRMS = new EtaPhiHists();
     ChannelStatusTimeRMS->setup(dbe_," Time RMS");

     
} 

void HcalDetDiagLEDMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
int  eta,phi,depth,nTS;
   HcalBaseDQMonitor::getLogicalMap(iSetup);
   if (emap==0) {
     emap=new HcalElectronicsMap(logicalMap_->generateHcalElectronicsMap());
   }

   if(!dbe_) return; 
   bool LEDEvent=false;
   bool LocalRun=false;
   // for local runs 

   edm::Handle<HcalTBTriggerData> trigger_data;
   iEvent.getByLabel(hcalTBTriggerDataTag_, trigger_data);
   if(trigger_data.isValid()){
      if(trigger_data->triggerWord()==6){ LEDEvent=true;LocalRun=true;}
   } 
   if(!LocalRun) return;  
   if(!LEDEvent) return; 
   
   HcalBaseDQMonitor::analyze(iEvent, iSetup);
   meEVT_->Fill(++ievt_);
   run_number=iEvent.id().run();
   meRUN_->Fill(iEvent.id().run());

   double data[20];

   edm::Handle<HBHEDigiCollection> hbhe; 
   iEvent.getByLabel(digiLabel_, hbhe);
   if(hbhe.isValid()) for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
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

   edm::Handle<HODigiCollection> ho; 
   iEvent.getByLabel(digiLabel_,ho);
   if(ho.isValid()) for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
     eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
     for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-2.5;
     ho_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
   }   

   edm::Handle<HFDigiCollection> hf;
   iEvent.getByLabel(digiLabel_,hf);
   if(hf.isValid()) for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
     eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
     for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-2.5;
     hf_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
   }   
 
   edm::Handle<HcalCalibDigiCollection> calib;
   iEvent.getByLabel(calibDigiLabel_, calib);
   if(calib.isValid())for(HcalCalibDigiCollection::const_iterator digi=calib->begin();digi!=calib->end();digi++){
     if(digi->id().cboxChannel()!=0 || digi->id().hcalSubdet()==0) continue; 
     nTS=digi->size();
     double e=0; 
     for(int i=0;i<nTS;i++){ data[i]=adc2fC[digi->sample(i).adc()&0xff]; e+=data[i];}
     if(e<15000) calib_data[digi->id().hcalSubdet()][digi->id().ieta()+2][digi->id().iphi()-1].add_statistics(data,nTS);
   }   
  
   if(((ievt_)%500)==0){
       fillHistos();
       CheckStatus(); 
   }
   return;
}


void HcalDetDiagLEDMonitor::fillHistos(){
  std::string subdet[4]={"HB","HE","HO","HF"};
    Energy->Reset();
   Time->Reset();
   EnergyRMS->Reset();
   TimeRMS->Reset();
   EnergyHF->Reset();
   TimeHF->Reset();
   EnergyRMSHF->Reset();
   TimeRMSHF->Reset();
   EnergyCorr->Reset();
   Time2Dhbhehf->Reset();
   Time2Dho->Reset();
   Energy2Dhbhehf->Reset();
   Energy2Dho->Reset();
   HBPphi->Reset();
   HBMphi->Reset();
   HEPphi->Reset();
   HEMphi->Reset();
   HFPphi->Reset();
   HFMphi->Reset();
   HO0phi->Reset();
   HO1Pphi->Reset();
   HO2Pphi->Reset();
   HO1Mphi->Reset();
   HO2Mphi->Reset();
   
   // HB histograms
   for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++){ 
      double T=0,nT=0,E=0,nE=0;
      for(int depth=1;depth<=2;depth++){
         if(hb_data[eta+42][phi-1][depth-1].get_statistics()>100){
            double ave=0;
	    double rms=0;
	    double time=0;
	    double time_rms=0;
	    hb_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    hb_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    TimeRMS->Fill(time_rms);
	    T+=time; nT++; E+=ave; nE++;
	    if(GetCalib("HB",eta,phi)->get_statistics()>100){
	      double ave_calib=0;
	      double rms_calib=0;
	      GetCalib("HB",eta,phi)->get_average_led(&ave_calib,&rms_calib);
	      fill_energy("HB",eta,phi,depth,ave/ave_calib,1);
	      EnergyCorr->Fill(ave_calib/ave);
	    }
         }
      } 
      if(nT>0){Time2Dhbhehf->Fill(eta,phi,T/nT);Energy2Dhbhehf->Fill(eta,phi,E/nE); }
   } 
   // HE histograms
   for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++){
      double T=0,nT=0,E=0,nE=0;
      for(int depth=1;depth<=3;depth++){
         if(he_data[eta+42][phi-1][depth-1].get_statistics()>100){
	    double ave=0;
	    double rms=0;
	    double time=0;
	    double time_rms=0;
	    he_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    he_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    T+=time; nT++; E+=ave; nE++;
	    TimeRMS->Fill(time_rms);
	    if(GetCalib("HE",eta,phi)->get_statistics()>100){
	      double ave_calib=0;
	      double rms_calib=0;
	      GetCalib("HE",eta,phi)->get_average_led(&ave_calib,&rms_calib);
	      fill_energy("HE",eta,phi,depth,ave/ave_calib,1);
	      EnergyCorr->Fill(ave_calib/ave);
	    }
         }
      }
      if(nT>0 && abs(eta)>16 ){Time2Dhbhehf->Fill(eta,phi,T/nT);   Energy2Dhbhehf->Fill(eta,phi,E/nE); }	 
      if(nT>0 && abs(eta)>20 ){Time2Dhbhehf->Fill(eta,phi+1,T/nT); Energy2Dhbhehf->Fill(eta,phi+1,E/nE);}	 
   } 
   // HF histograms
   for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++){
      double T=0,nT=0,E=0,nE=0;
      for(int depth=1;depth<=2;depth++){
         if(hf_data[eta+42][phi-1][depth-1].get_statistics()>100){
	   double ave=0;
	   double rms=0;
	   double time=0;
	   double time_rms=0;
	   hf_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	   hf_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	   EnergyHF->Fill(ave);
	   if(ave>0)EnergyRMSHF->Fill(rms/ave);
	   TimeHF->Fill(time);
	   T+=time; nT++; E+=ave; nE++;
	   TimeRMSHF->Fill(time_rms);
	   if(GetCalib("HF",eta,phi)->get_statistics()>100){
	     double ave_calib=0;
	     double rms_calib=0;
	     GetCalib("HF",eta,phi)->get_average_led(&ave_calib,&rms_calib);
	     fill_energy("HF",eta,phi,depth,ave/ave_calib,1);
	     EnergyCorr->Fill(ave_calib/ave);
	   }
         }
      }	
      if(nT>0 && abs(eta)>29 ){ Time2Dhbhehf->Fill(eta,phi,T/nT); Time2Dhbhehf->Fill(eta,phi+1,T/nT);}	 
      if(nT>0 && abs(eta)>29 ){ Energy2Dhbhehf->Fill(eta,phi,E/nE); Energy2Dhbhehf->Fill(eta,phi+1,E/nE);}	 
   } 
   // HO histograms
   for(int eta=-10;eta<=15;eta++) for(int phi=1;phi<=72;phi++){
      double T=0,nT=0,E=0,nE=0;
      if(eta>10 && !isSiPM(eta,phi,4)) continue;
      for(int depth=4;depth<=4;depth++){
         if(ho_data[eta+42][phi-1][depth-1].get_statistics()>100){
	    double ave=0;
	    double rms=0;
	    double time=0;
	    double time_rms=0;
	    ho_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    ho_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    T+=time; nT++; E+=ave; nE++;
	    TimeRMS->Fill(time_rms);
	    if(GetCalib("HO",eta,phi)->get_statistics()>100){
	      double ave_calib=0;
	      double rms_calib=0;
	      GetCalib("HO",eta,phi)->get_average_led(&ave_calib,&rms_calib);
	      fill_energy("HO",eta,phi,depth,ave/ave_calib,1);
	      EnergyCorr->Fill(ave_calib/ave);
	    }
         }
      }
      if(nT>0){ Time2Dho->Fill(eta,phi,T/nT); Energy2Dho->Fill(eta,phi+1,E/nE) ;}
   } 

   double ave=0.,rms=0.,ave_calib=0.,rms_calib=0.;
   // HB Ref histograms
   for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
      if(hb_data[eta+42][phi-1][depth-1].get_reference(&ave,&rms) && GetCalib("HB",eta,phi)->get_reference(&ave_calib,&rms_calib)){
	    fill_energy("HB",eta,phi,depth,ave/ave_calib,2);
      }
   } 
   // HE Ref histograms
   for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=3;depth++){
      if(he_data[eta+42][phi-1][depth-1].get_reference(&ave,&rms) && GetCalib("HE",eta,phi)->get_reference(&ave_calib,&rms_calib)){
	    fill_energy("HE",eta,phi,depth,ave/ave_calib,2);
      }
   } 
   // HO Ref histograms
   for(int eta=-10;eta<=15;eta++) for(int phi=1;phi<=72;phi++) for(int depth=4;depth<=4;depth++){
      if(eta>10 && !isSiPM(eta,phi,4)) continue;
      if(ho_data[eta+42][phi-1][depth-1].get_reference(&ave,&rms) && GetCalib("HO",eta,phi)->get_reference(&ave_calib,&rms_calib)){
	    fill_energy("HO",eta,phi,depth,ave/ave_calib,2);
      }
   } 
   // HF Ref histograms
   for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
      if(hf_data[eta+42][phi-1][depth-1].get_reference(&ave,&rms) && GetCalib("HF",eta,phi)->get_reference(&ave_calib,&rms_calib)){
	    fill_energy("HF",eta,phi,depth,ave/ave_calib,2);
      }
   } 

  //fill RM histograms: this part is incomplete, will be modefied later 
  double hbp[18][4],nhbp[18][4],hbm[18][4],nhbm[18][4];
  double hep[18][4],nhep[18][4],hem[18][4],nhem[18][4];
  double hfp[18][4],nhfp[18][4],hfm[18][4],nhfm[18][4];
  double ho0[18][4],nho0[18][4];
  double ho1p[18][4],nho1p[18][4];
  double ho2p[18][4],nho2p[18][4];
  double ho1m[18][4],nho1m[18][4];
  double ho2m[18][4],nho2m[18][4];
  for(int i=0;i<18;i++) for(int j=0;j<4;j++)
   hbp[i][j]=nhbp[i][j]=hbm[i][j]=nhbm[i][j]=hep[i][j]=nhep[i][j]=hem[i][j]=nhem[i][j]=hfp[i][j]=nhfp[i][j]=hfm[i][j]=nhfm[i][j]=0;
  for(int i=0;i<18;i++) for(int j=0;j<4;j++)
   ho0[i][j]=nho0[i][j]=ho1p[i][j]=nho1p[i][j]=ho2p[i][j]=nho2p[i][j]=ho1m[i][j]=nho1m[i][j]=ho2m[i][j]=nho2m[i][j]=0;

   std::vector <HcalElectronicsId> AllElIds = emap->allElectronicsIdPrecision();
   for(std::vector <HcalElectronicsId>::iterator eid = AllElIds.begin(); eid != AllElIds.end(); eid++){
      DetId detid=emap->lookup(*eid);
      if(detid.det()!=DetId::Hcal) continue;
      HcalGenericDetId gid(emap->lookup(*eid));
      if(!(!(gid.null()) && 
            (gid.genericSubdet()==HcalGenericDetId::HcalGenBarrel ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenEndcap  ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenForward ||
             gid.genericSubdet()==HcalGenericDetId::HcalGenOuter))) continue;
      int sd=0,eta=0,phi=0,depth=0; 
      if(gid.genericSubdet()==HcalGenericDetId::HcalGenBarrel)      sd=0;
      else if(gid.genericSubdet()==HcalGenericDetId::HcalGenEndcap) sd=1;
      else if(gid.genericSubdet()==HcalGenericDetId::HcalGenOuter)  sd=2;
      else if(gid.genericSubdet()==HcalGenericDetId::HcalGenForward)sd=3;
      HcalDetId hid(detid);
      if(KnownBadCells_.find(hid.rawId())==KnownBadCells_.end()) continue;
    
      eta=hid.ieta();
      phi=hid.iphi();
      depth=hid.depth(); 
      
      double ave =get_energy(subdet[sd],eta,phi,depth,1);
      double ref =get_energy(subdet[sd],eta,phi,depth,2);

      HcalFrontEndId  lmap_entry=logicalMap_->getHcalFrontEndId(hid);
      int rbx; 
      if(sd==0 || sd==1 || sd==3){
	   sscanf(&(lmap_entry.rbx().c_str())[3],"%d",&rbx);
      }else{
	   if(abs(eta)<5) sscanf(&(lmap_entry.rbx().c_str())[3],"%d",&rbx);
	   if(abs(eta)>=5) sscanf(&(lmap_entry.rbx().c_str())[4],"%d",&rbx);	       
      }
      if(ave>0 && ref>0){
	   if(sd==0 && eta>0){ hbp[rbx-1][lmap_entry.rm()-1]+=ave/ref; nhbp[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==0 && eta<0){ hbm[rbx-1][lmap_entry.rm()-1]+=ave/ref; nhbm[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==1 && eta>0){ hep[rbx-1][lmap_entry.rm()-1]+=ave/ref; nhep[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==1 && eta<0){ hem[rbx-1][lmap_entry.rm()-1]+=ave/ref; nhem[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==3 && eta>0){ hfp[rbx-1][lmap_entry.rm()-1]+=ave/ref; nhfp[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==3 && eta<0){ hfm[rbx-1][lmap_entry.rm()-1]+=ave/ref; nhfm[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==2 && abs(eta)<5){ ho0[rbx-1][lmap_entry.rm()-1]+=ave/ref; nho0[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==2 && eta>=5 && eta<=10){ ho1p[rbx-1][lmap_entry.rm()-1]+=ave/ref; nho1p[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==2 && eta>=11 && eta<=15){ ho2p[rbx-1][lmap_entry.rm()-1]+=ave/ref; nho2p[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==2 && eta>=-10 && eta<=-5){ ho1m[rbx-1][lmap_entry.rm()-1]+=ave/ref; nho1m[rbx-1][lmap_entry.rm()-1]++; }
	   if(sd==2 && eta>=-15 && eta<=-11){ ho2m[rbx-1][lmap_entry.rm()-1]+=ave/ref; nho2m[rbx-1][lmap_entry.rm()-1]++; }
      }
  }  
  for(int i=0;i<18;i++)for(int j=0;j<4;j++){
     int phi=i*4+j+1; 
     if(nhbp[i][j]>1) HBPphi->Fill(phi+0.5,hbp[i][j]/nhbp[i][j]);
     if(nhbm[i][j]>1) HBMphi->Fill(phi+0.5,hbm[i][j]/nhbm[i][j]);
     if(nhep[i][j]>1) HEPphi->Fill(phi+0.5,hep[i][j]/nhep[i][j]);
     if(nhem[i][j]>1) HEMphi->Fill(phi+0.5,hem[i][j]/nhem[i][j]);
  }   
  for(int i=0;i<12;i++)for(int j=0;j<3;j++){
     int phi=i*3+j+1; 
     if(nhfp[i][j]>1) HFPphi->Fill(phi+0.5,hfp[i][j]/nhfp[i][j]);
     if(nhfm[i][j]>1) HFMphi->Fill(phi+0.5,hfm[i][j]/nhfm[i][j]);
  } 
  for(int i=0;i<12;i++)for(int j=0;j<4;j++){
     int phi=i*4+j+1; 
     if(nho0[i][j]>1) HO0phi->Fill(phi+0.5,ho0[i][j]/nho0[i][j]);
     if(nho1p[i][j]>1) HO1Pphi->Fill(phi+0.5,ho1p[i][j]/nho1p[i][j]);
     if(nho2p[i][j]>1) HO2Pphi->Fill(phi+0.5,ho2p[i][j]/nho2p[i][j]);
     if(nho1m[i][j]>1) HO1Mphi->Fill(phi+0.5,ho1m[i][j]/nho1m[i][j]);
     if(nho2m[i][j]>1) HO2Mphi->Fill(phi+0.5,ho2m[i][j]/nho2m[i][j]);
  } 
} 

void HcalDetDiagLEDMonitor::SaveReference(){
double led,rms,Time,time_rms;
int    Eta,Phi,Depth,Statistic,Status=0;
char   Subdet[10],str[500];
       sprintf(str,"%sHcalDetDiagLEDData_run%06i_%i.root",OutputFilePath.c_str(),run_number,dataset_seq_number);
       TFile *theFile = new TFile(str, "RECREATE");
       if(!theFile->IsOpen()) return;
       theFile->cd();
       sprintf(str,"%d",run_number); TObjString run(str);    run.Write("run number");
       sprintf(str,"%d",ievt_);      TObjString events(str); events.Write("Total events processed");
       
       TTree *tree   =new TTree("HCAL LED data","HCAL LED data");
       if(tree==0)   return;
       tree->Branch("Subdet",   &Subdet,         "Subdet/C");
       tree->Branch("eta",      &Eta,            "Eta/I");
       tree->Branch("phi",      &Phi,            "Phi/I");
       tree->Branch("depth",    &Depth,          "Depth/I");
       tree->Branch("statistic",&Statistic,      "Statistic/I");
       tree->Branch("status",   &Status,         "Status/I");
       tree->Branch("led",      &led,            "led/D");
       tree->Branch("rms",      &rms,            "rms/D");
       tree->Branch("time",     &Time,           "time/D");
       tree->Branch("time_rms", &time_rms,       "time_rms/D");
       sprintf(Subdet,"HB");
       for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
          if((Statistic=hb_data[eta+42][phi-1][depth-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hb_data[eta+42][phi-1][depth-1].get_status();
	     hb_data[eta+42][phi-1][depth-1].get_average_led(&led,&rms);
	     hb_data[eta+42][phi-1][depth-1].get_average_time(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"HE");
       for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=3;depth++){
         if((Statistic=he_data[eta+42][phi-1][depth-1].get_statistics())>100){
            Eta=eta; Phi=phi; Depth=depth;
	    Status=he_data[eta+42][phi-1][depth-1].get_status();
	    he_data[eta+42][phi-1][depth-1].get_average_led(&led,&rms);
	    he_data[eta+42][phi-1][depth-1].get_average_time(&Time,&time_rms);
	    tree->Fill();
         }
       } 
       sprintf(Subdet,"HO");
       for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++) for(int depth=4;depth<=4;depth++){
         if((Statistic=ho_data[eta+42][phi-1][depth-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=ho_data[eta+42][phi-1][depth-1].get_status();
	     ho_data[eta+42][phi-1][depth-1].get_average_led(&led,&rms);
	     ho_data[eta+42][phi-1][depth-1].get_average_time(&Time,&time_rms);
	     tree->Fill();
         }
       } 
       sprintf(Subdet,"HF");
       for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
         if((Statistic=hf_data[eta+42][phi-1][depth-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hf_data[eta+42][phi-1][depth-1].get_status();
	     hf_data[eta+42][phi-1][depth-1].get_average_led(&led,&rms);
	     hf_data[eta+42][phi-1][depth-1].get_average_time(&Time,&time_rms);
	     tree->Fill();
         }
       }
       sprintf(Subdet,"CALIB_HB");
       for(int eta=-1;eta<=1;eta++) for(int phi=1;phi<=72;phi++){
          if((calib_data[1][eta+2][phi-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[1][eta+2][phi-1].get_status();
 	     calib_data[1][eta+2][phi-1].get_average_led(&led,&rms);
	     calib_data[1][eta+2][phi-1].get_average_time(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"CALIB_HE");
       for(int eta=-1;eta<=1;eta++) for(int phi=1;phi<=72;phi++){
          if((calib_data[2][eta+2][phi-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[2][eta+2][phi-1].get_status();
 	     calib_data[2][eta+2][phi-1].get_average_led(&led,&rms);
	     calib_data[2][eta+2][phi-1].get_average_time(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"CALIB_HO");
       for(int eta=-2;eta<=2;eta++) for(int phi=1;phi<=72;phi++){
          if((calib_data[3][eta+2][phi-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[3][eta+2][phi-1].get_status();
 	     calib_data[3][eta+2][phi-1].get_average_led(&led,&rms);
	     calib_data[3][eta+2][phi-1].get_average_time(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"CALIB_HF");
       for(int eta=-2;eta<=2;eta++) for(int phi=1;phi<=72;phi++){
          if((calib_data[4][eta+2][phi-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[4][eta+2][phi-1].get_status();
 	     calib_data[4][eta+2][phi-1].get_average_led(&led,&rms);
	     calib_data[4][eta+2][phi-1].get_average_time(&Time,&time_rms);
	     tree->Fill();
          }
       } 
       theFile->Write();
       theFile->Close();


   if(XmlFilePath.size()>0){
      //create XML file
      char TIME[40];
      Long_t t; t=time(0); strftime(TIME,30,"%F %T",localtime(&t));

      sprintf(str,"HcalDetDiagLED_%i_%i.xml",run_number,dataset_seq_number);
      std::string xmlName=str;
      ofstream xmlFile;
      xmlFile.open(xmlName.c_str());

      xmlFile<<"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
      xmlFile<<"<ROOT>\n";
      xmlFile<<"  <HEADER>\n";
      xmlFile<<"    <HINTS mode='only-det-root'/>\n";
      xmlFile<<"    <TYPE>\n";
      xmlFile<<"      <EXTENSION_TABLE_NAME>HCAL_DETMON_LED_LASER_V1</EXTENSION_TABLE_NAME>\n";
      xmlFile<<"      <NAME>HCAL LED [local]</NAME>\n";
      xmlFile<<"    </TYPE>\n";
      xmlFile<<"    <!-- run details -->\n";
      xmlFile<<"    <RUN>\n";
      xmlFile<<"      <RUN_TYPE>LOCAL-RUN</RUN_TYPE>\n";
      xmlFile<<"      <RUN_NUMBER>"<<run_number<<"</RUN_NUMBER>\n";
      xmlFile<<"      <RUN_BEGIN_TIMESTAMP>2009-01-01 00:00:00</RUN_BEGIN_TIMESTAMP>\n";
      xmlFile<<"      <COMMENT_DESCRIPTION>hcal LED data</COMMENT_DESCRIPTION>\n";
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
             Statistic=hb_data[eta+42][phi-1][depth-1].get_statistics();
             Status   =hb_data[eta+42][phi-1][depth-1].get_status();
             hb_data[eta+42][phi-1][depth-1].get_average_led(&e,&e_rms);
             hb_data[eta+42][phi-1][depth-1].get_average_time(&t,&t_rms);
         }else if(detid.subdetId()==HcalEndcap){
             subdet="HE";
             Statistic=he_data[eta+42][phi-1][depth-1].get_statistics();
             Status   =he_data[eta+42][phi-1][depth-1].get_status();
             he_data[eta+42][phi-1][depth-1].get_average_led(&e,&e_rms);
             he_data[eta+42][phi-1][depth-1].get_average_time(&t,&t_rms);
         }else if(detid.subdetId()==HcalForward){
             subdet="HF";
             Statistic=hf_data[eta+42][phi-1][depth-1].get_statistics();
             Status   =hf_data[eta+42][phi-1][depth-1].get_status();
             hf_data[eta+42][phi-1][depth-1].get_average_led(&e,&e_rms);
             hf_data[eta+42][phi-1][depth-1].get_average_time(&t,&t_rms);
         }else if(detid.subdetId()==HcalOuter){
             subdet="HO";
             Statistic=ho_data[eta+42][phi-1][depth-1].get_statistics();
             Status   =ho_data[eta+42][phi-1][depth-1].get_status();
             ho_data[eta+42][phi-1][depth-1].get_average_led(&e,&e_rms);
             ho_data[eta+42][phi-1][depth-1].get_average_time(&t,&t_rms);
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
      sprintf(str,"HcalDetDiagLEDCalib_%i_%i.xml",run_number,dataset_seq_number);
      std::string xmlNameCalib=str;
      ofstream xmlFileCalib;
      xmlFileCalib.open(xmlNameCalib.c_str());

      xmlFileCalib<<"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n";
      xmlFileCalib<<"<ROOT>\n";
      xmlFileCalib<<"  <HEADER>\n";
      xmlFileCalib<<"    <HINTS mode='only-det-root'/>\n";
      xmlFileCalib<<"    <TYPE>\n";
      xmlFileCalib<<"      <EXTENSION_TABLE_NAME>HCAL_DETMON_LED_LASER_V1</EXTENSION_TABLE_NAME>\n";
      xmlFileCalib<<"      <NAME>HCAL LED CALIB [local]</NAME>\n";
      xmlFileCalib<<"    </TYPE>\n";
      xmlFileCalib<<"    <!-- run details -->\n";
      xmlFileCalib<<"    <RUN>\n";
      xmlFileCalib<<"      <RUN_TYPE>LOCAL-RUN</RUN_TYPE>\n";
      xmlFileCalib<<"      <RUN_NUMBER>"<<run_number<<"</RUN_NUMBER>\n";
      xmlFileCalib<<"      <RUN_BEGIN_TIMESTAMP>2009-01-01 00:00:00</RUN_BEGIN_TIMESTAMP>\n";
      xmlFileCalib<<"      <COMMENT_DESCRIPTION>hcal LED CALIB data</COMMENT_DESCRIPTION>\n";
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
             Statistic=calib_data[sd][eta+2][phi-1].get_statistics(); 
 	     calib_data[sd][eta+2][phi-1].get_average_led(&e,&e_rms);
	     calib_data[sd][eta+2][phi-1].get_average_time(&t,&t_rms);
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


   dataset_seq_number++;
}

void HcalDetDiagLEDMonitor::LoadReference(){
double led,rms;
int Eta,Phi,Depth;
char subdet[10];
TFile *f;
   if(gSystem->AccessPathName(ReferenceData.c_str())) return;
   f = new TFile(ReferenceData.c_str(),"READ");
   if(!f->IsOpen()) return ;
   TObjString *STR=(TObjString *)f->Get("run number");
   if(STR){ std::string Ref(STR->String()); ReferenceRun=Ref;}
   TTree*  t=(TTree*)f->Get("HCAL LED data");
   if(!t) return;
   t->SetBranchAddress("Subdet",   subdet);
   t->SetBranchAddress("eta",      &Eta);
   t->SetBranchAddress("phi",      &Phi);
   t->SetBranchAddress("depth",    &Depth);
   t->SetBranchAddress("led",      &led);
   t->SetBranchAddress("rms",      &rms);
   for(int ievt=0;ievt<t->GetEntries();ievt++){

     t->GetEntry(ievt);
     if(strcmp(subdet,"HB")==0) hb_data[Eta+42][Phi-1][Depth-1].set_reference(led,rms);
     if(strcmp(subdet,"HE")==0) he_data[Eta+42][Phi-1][Depth-1].set_reference(led,rms);
     if(strcmp(subdet,"HO")==0) ho_data[Eta+42][Phi-1][Depth-1].set_reference(led,rms);
     if(strcmp(subdet,"HF")==0) hf_data[Eta+42][Phi-1][Depth-1].set_reference(led,rms);
     if(strcmp(subdet,"CALIB_HB")==0) calib_data[1][Eta+2][Phi-1].set_reference(led,rms);
     if(strcmp(subdet,"CALIB_HE")==0) calib_data[2][Eta+2][Phi-1].set_reference(led,rms);
     if(strcmp(subdet,"CALIB_HO")==0) calib_data[3][Eta+2][Phi-1].set_reference(led,rms);
     if(strcmp(subdet,"CALIB_HF")==0) calib_data[4][Eta+2][Phi-1].set_reference(led,rms);
   }
   f->Close();
   IsReference=true;
} 
void HcalDetDiagLEDMonitor::CheckStatus(){
   for(int i=0;i<4;i++){
      ChannelStatusMissingChannels->depth[i]->Reset();
      ChannelStatusUnstableChannels->depth[i]->Reset();
      ChannelStatusUnstableLEDsignal->depth[i]->Reset();
      ChannelStatusLEDMean->depth[i]->Reset();
      ChannelStatusLEDRMS->depth[i]->Reset();
      ChannelStatusTimeMean->depth[i]->Reset();
      ChannelStatusTimeRMS->depth[i]->Reset();
   }
  
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
      if(KnownBadCells_.find(gid.rawId())==KnownBadCells_.end()) continue;

      HcalDetId hid(detid);
      eta=hid.ieta();
      phi=hid.iphi();
      depth=hid.depth(); 
      
      double AVE_TIME=Time->getMean();
      if(detid.subdetId()==HcalBarrel){
	 int stat=hb_data[eta+42][phi-1][depth-1].get_statistics()+
	           hb_data[eta+42][phi-1][depth-1].get_overflow()+hb_data[eta+42][phi-1][depth-1].get_undeflow();
	 if(stat==0){ 
	     fill_channel_status("HB",eta,phi,depth,1,1); 
	     hb_data[eta+42][phi-1][depth-1].change_status(1); 
	 }
         if(stat>0 && stat!=(ievt_)){ 
             fill_channel_status("HB",eta,phi,depth,2,(double)stat/(double)(ievt_)); 
	     hb_data[eta+42][phi-1][depth-1].change_status(2); 
         }
         if(hb_data[eta+42][phi-1][depth-1].get_statistics()>100){ 
	     double ave=0;
	     double rms=0;
	     hb_data[eta+42][phi-1][depth-1].get_average_time(&ave,&rms);
	     if((AVE_TIME-ave)>0.75 || (AVE_TIME-ave)<-0.75){
                fill_channel_status("HB",eta,phi,depth,6,AVE_TIME-ave); 
	        hb_data[eta+42][phi-1][depth-1].change_status(8); 
	     }
	 }  
         stat=hb_data[eta+42][phi-1][depth-1].get_undeflow();	  
         if(stat>0){ 
             fill_channel_status("HB",eta,phi,depth,3,(double)stat/(double)(ievt_)); 
	     hb_data[eta+42][phi-1][depth-1].change_status(4); 
	 }    
      } 
      if(detid.subdetId()==HcalEndcap){
	 int stat=he_data[eta+42][phi-1][depth-1].get_statistics()+
	           he_data[eta+42][phi-1][depth-1].get_overflow()+he_data[eta+42][phi-1][depth-1].get_undeflow();
	 if(stat==0){ 
	     fill_channel_status("HE",eta,phi,depth,1,1); 
	     he_data[eta+42][phi-1][depth-1].change_status(1); 
	 }
         if(stat>0 && stat!=(ievt_)){ 
             fill_channel_status("HE",eta,phi,depth,2,(double)stat/(double)(ievt_)); 
	     he_data[eta+42][phi-1][depth-1].change_status(2); 
         }
         if(he_data[eta+42][phi-1][depth-1].get_statistics()>100){ 
	     double ave=0;
	     double rms=0;
	     he_data[eta+42][phi-1][depth-1].get_average_time(&ave,&rms);
	     if((AVE_TIME-ave)>0.75 || (AVE_TIME-ave)<-0.75){ 
                fill_channel_status("HE",eta,phi,depth,6,AVE_TIME-ave); 
	        he_data[eta+42][phi-1][depth-1].change_status(8); 
	     }	
	 }  
         stat=he_data[eta+42][phi-1][depth-1].get_undeflow();	  
         if(stat>0){ 
             fill_channel_status("HE",eta,phi,depth,3,(double)stat/(double)(ievt_)); 
	     he_data[eta+42][phi-1][depth-1].change_status(4); 
	 }  
      } 
      if(detid.subdetId()==HcalOuter){
	 int stat=ho_data[eta+42][phi-1][depth-1].get_statistics()+
	           ho_data[eta+42][phi-1][depth-1].get_overflow()+ho_data[eta+42][phi-1][depth-1].get_undeflow();
	 if(stat==0){ 
	     fill_channel_status("HO",eta,phi,depth,1,1); 
	     ho_data[eta+42][phi-1][depth-1].change_status(1); 
	 }
         if(stat>0 && stat!=(ievt_)){ 
             fill_channel_status("HO",eta,phi,depth,2,(double)stat/(double)(ievt_)); 
	     ho_data[eta+42][phi-1][depth-1].change_status(2); 
         }
         if(ho_data[eta+42][phi-1][depth-1].get_statistics()>100){ 
	     double ave=0;
	     double rms=0;
	     ho_data[eta+42][phi-1][depth-1].get_average_time(&ave,&rms);
	     if((AVE_TIME-ave)>0.75 || (AVE_TIME-ave)<-0.75){
                fill_channel_status("HO",eta,phi,depth,6,AVE_TIME-ave); 
	        ho_data[eta+42][phi-1][depth-1].change_status(8);
	     } 
	 }  
         stat=ho_data[eta+42][phi-1][depth-1].get_undeflow();	  
         if(stat>0){ 
             fill_channel_status("HO",eta,phi,depth,3,(double)stat/(double)(ievt_)); 
	     ho_data[eta+42][phi-1][depth-1].change_status(4); 
	 }  
      } 
      if(detid.subdetId()==HcalForward){
	 AVE_TIME=TimeHF->getMean();
	 int stat=hf_data[eta+42][phi-1][depth-1].get_statistics()+
	           hf_data[eta+42][phi-1][depth-1].get_overflow()+hf_data[eta+42][phi-1][depth-1].get_undeflow();
	 if(stat==0){ 
	     fill_channel_status("HF",eta,phi,depth,1,1); 
	     hf_data[eta+42][phi-1][depth-1].change_status(1); 
	 }
         if(stat>0 && stat!=(ievt_)){ 
             fill_channel_status("HF",eta,phi,depth,2,(double)stat/(double)(ievt_)); 
	     hf_data[eta+42][phi-1][depth-1].change_status(2); 
         }
         if(hf_data[eta+42][phi-1][depth-1].get_statistics()>100){ 
	     double ave=0;
	     double rms=0;
	     hf_data[eta+42][phi-1][depth-1].get_average_time(&ave,&rms);
	     if((AVE_TIME-ave)>0.75 || (AVE_TIME-ave)<-0.75){
                fill_channel_status("HF",eta,phi,depth,6,AVE_TIME-ave); 
	        hf_data[eta+42][phi-1][depth-1].change_status(8);
	     } 
	 }  
         stat=hf_data[eta+42][phi-1][depth-1].get_undeflow();	  
         if(stat>0){ 
             fill_channel_status("HF",eta,phi,depth,3,(double)stat/(double)(ievt_)); 
	     hf_data[eta+42][phi-1][depth-1].change_status(4); 
	 }  
      } 
   }
}
void HcalDetDiagLEDMonitor::fill_energy(std::string subdet,int eta,int phi,int depth,double e,int type){ 
  int subdetval=-1;
  if (subdet.compare("HB")==0) subdetval=(int)HcalBarrel;
  else if (subdet.compare("HE")==0) subdetval=(int)HcalEndcap;
  else if (subdet.compare("HO")==0) subdetval=(int)HcalOuter;
  else if (subdet.compare("HF")==0) subdetval=(int)HcalForward;
  else return;

  int ietabin=CalcEtaBin(subdetval, eta, depth)+1;
  if(type==1) ChannelsLEDEnergy->depth[depth-1]   ->setBinContent(ietabin,phi,e);
  else if(type==2) ChannelsLEDEnergyRef->depth[depth-1]->setBinContent(ietabin,phi,e);
}

double HcalDetDiagLEDMonitor::get_energy(std::string subdet,int eta,int phi,int depth,int type){
  int subdetval=-1;
  if (subdet.compare("HB")==0) subdetval=(int)HcalBarrel;
  else if (subdet.compare("HE")==0) subdetval=(int)HcalEndcap;
  else if (subdet.compare("HO")==0) subdetval=(int)HcalOuter;
  else if (subdet.compare("HF")==0) subdetval=(int)HcalForward;
  else return -1.0;

  int ietabin=CalcEtaBin(subdetval, eta, depth)+1;
  if(type==1) return ChannelsLEDEnergy->depth[depth-1]  ->getBinContent(ietabin, phi);
  else if(type==2) return ChannelsLEDEnergyRef->depth[depth-1] ->getBinContent(ietabin,phi);
  return -1.0;
}

void HcalDetDiagLEDMonitor::fill_channel_status(std::string subdet,int eta,int phi,int depth,int type,double status){
  int subdetval=-1;
  if (subdet.compare("HB")==0) subdetval=(int)HcalBarrel;
  else if (subdet.compare("HE")==0) subdetval=(int)HcalEndcap;
  else if (subdet.compare("HO")==0) subdetval=(int)HcalOuter;
  else if (subdet.compare("HF")==0) subdetval=(int)HcalForward;
  else return;
  int ietabin=CalcEtaBin(subdetval, eta, depth)+1;

   if(type==1) ChannelStatusMissingChannels->depth[depth-1]  ->setBinContent(ietabin,phi,status);
   if(type==2) ChannelStatusUnstableChannels->depth[depth-1] ->setBinContent(ietabin,phi,status);
   if(type==3) ChannelStatusUnstableLEDsignal->depth[depth-1]->setBinContent(ietabin,phi,status);
   if(type==4) ChannelStatusLEDMean->depth[depth-1]          ->setBinContent(ietabin,phi,status);
   if(type==5) ChannelStatusLEDRMS->depth[depth-1]           ->setBinContent(ietabin,phi,status);
   if(type==6) ChannelStatusTimeMean->depth[depth-1]         ->setBinContent(ietabin,phi,status);
   if(type==7) ChannelStatusTimeRMS->depth[depth-1]          ->setBinContent(ietabin,phi,status);
}
void HcalDetDiagLEDMonitor::endRun(const edm::Run& run, const edm::EventSetup& c){   
   if(ievt_>=100){
      fillHistos();
      CheckStatus();
      SaveReference(); 
   }   
} 
DEFINE_FWK_MODULE (HcalDetDiagLEDMonitor);

