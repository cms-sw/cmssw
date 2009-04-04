#include "DQM/HcalMonitorTasks/interface/HcalDetDiagLaserMonitor.h"

#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include "TFile.h"
#include "TTree.h"

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

HcalDetDiagLaserMonitor::HcalDetDiagLaserMonitor() {
  ievt_=0;
  emap=0;
  dataset_seq_number=1;
  run_number=-1;
  IsReference=false;
}

HcalDetDiagLaserMonitor::~HcalDetDiagLaserMonitor(){}

void HcalDetDiagLaserMonitor::clearME(){
  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();
    m_dbe = 0;
  }
} 
void HcalDetDiagLaserMonitor::reset(){}

void HcalDetDiagLaserMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  m_dbe=NULL;
  ievt_=0;
  if(dbe!=NULL) m_dbe=dbe;
  clearME();
 
  UseDB            = ps.getUntrackedParameter<bool>  ("UseDB"  , false);
  
  ReferenceData    = ps.getUntrackedParameter<string>("LaserReferenceData" ,"");
  OutputFilePath   = ps.getUntrackedParameter<string>("OutputFilePath", "");
 
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"HcalDetDiagLaserMonitor";
  char *name;
  if(m_dbe!=NULL){    
     m_dbe->setCurrentFolder(baseFolder_);   
     meEVT_ = m_dbe->bookInt("HcalDetDiagLaserMonitor Event Number");
     m_dbe->setCurrentFolder(baseFolder_+"/Summary Plots");
     
     name="Laser Energy Distribution";               Energy         = m_dbe->book1D(name,name,200,0,3000);
     name="Laser Timing Distribution";               Time           = m_dbe->book1D(name,name,200,0,10);
     name="Laser Energy RMS/Energy Distribution";    EnergyRMS      = m_dbe->book1D(name,name,200,0,0.2);
     name="Laser Timing RMS Distribution";           TimeRMS        = m_dbe->book1D(name,name,200,0,1);
     name="Laser Timing HBHEHF";                     Time2Dhbhehf   = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="Laser Timing HO";                         Time2Dho       = m_dbe->book2D(name,name,33,-16,16,74,0,73);
     name="Laser Energy HBHEHF";                     Energy2Dhbhehf = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="Laser Energy HO";                         Energy2Dho     = m_dbe->book2D(name,name,33,-16,16,74,0,73);
  }  
  LoadReference();
  return;
} 

void HcalDetDiagLaserMonitor::processEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HcalDbService& cond){
int  eta,phi,depth,nTS;
   if(emap==0) emap=cond.getHcalMapping();
   if(!m_dbe) return; 
   bool LaserEvent=false;
   bool LocalRun=false;
   // for local runs 
   try{
       edm::Handle<HcalTBTriggerData> trigger_data;
       iEvent.getByType(trigger_data);
       if(trigger_data->wasLaserTrigger()) LaserEvent=true;
       LocalRun=true;
   }catch(...){}
   //if(LocalRun && !LaserEvent) return;
   
   // Abort Gap laser 
   if(LocalRun==false || LaserEvent==false){
       edm::Handle<FEDRawDataCollection> rawdata;
       iEvent.getByType(rawdata);
       //checking FEDs for calibration information
       for (int i=FEDNumbering::getHcalFEDIds().first;i<=FEDNumbering::getHcalFEDIds().second; i++) {
          const FEDRawData& fedData = rawdata->FEDData(i) ;
          if ( fedData.size() < 24 ) continue ;
          int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ;
          if(value==hc_HBHEHPD || value==hc_HOHPD || value==hc_HFPMT){ LaserEvent=true; break;} 
       }
   }   
   if(!LaserEvent) return;
   
   ievt_++;
   meEVT_->Fill(ievt_);
   run_number=iEvent.id().run();
   double data[20];
   try{
         edm::Handle<HBHEDigiCollection> hbhe; 
         iEvent.getByType(hbhe);
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
   }catch(...){}      
   try{
         edm::Handle<HODigiCollection> ho; 
         iEvent.getByType(ho);
         for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-2.5;
             ho_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
         }   
   }catch(...){}  
   try{
         edm::Handle<HFDigiCollection> hf;
         iEvent.getByType(hf);
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-2.5;
	     hf_data[eta+42][phi-1][depth-1].add_statistics(data,nTS);
         }   
   }catch(...){}    
 
   if(((ievt_)%100)==0){
       fillHistos();
   }
   return;
}


void HcalDetDiagLaserMonitor::fillHistos(){
   Energy->Reset();
   Time->Reset();
   EnergyRMS->Reset();
   TimeRMS->Reset();
   Time2Dhbhehf->Reset();
   Time2Dho->Reset();
   Energy2Dhbhehf->Reset();
   Energy2Dho->Reset();
   // HB histograms
   for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++){ 
      double T=0,nT=0,E=0,nE=0;
      for(int depth=1;depth<=2;depth++){
         if(hb_data[eta+42][phi-1][depth-1].get_statistics()>10){
            double ave,rms,time,time_rms;
	    hb_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    hb_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    TimeRMS->Fill(time_rms);
	    T+=time; nT++; E+=ave; nE++;
         }
      } 
      if(nT>0){Time2Dhbhehf->Fill(eta,phi,T/nT);Energy2Dhbhehf->Fill(eta,phi,E/nE); }
   } 
   // HE histograms
   for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++){
      double T=0,nT=0,E=0,nE=0;
      for(int depth=1;depth<=3;depth++){
         if(he_data[eta+42][phi-1][depth-1].get_statistics()>10){
            double ave,rms,time,time_rms;
	    he_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    he_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    T+=time; nT++; E+=ave; nE++;
	    TimeRMS->Fill(time_rms);
         }
      }
      if(nT>0 && abs(eta)>16 ){Time2Dhbhehf->Fill(eta,phi,T/nT);   Energy2Dhbhehf->Fill(eta,phi,E/nE); }	 
      if(nT>0 && abs(eta)>20 ){Time2Dhbhehf->Fill(eta,phi+1,T/nT); Energy2Dhbhehf->Fill(eta,phi+1,E/nE);}	 
   } 
   // HF histograms
   for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++){
      double T=0,nT=0,E=0,nE=0;
      for(int depth=1;depth<=2;depth++){
         if(hf_data[eta+42][phi-1][depth-1].get_statistics()>10){
            double ave,rms,time,time_rms;
	    hf_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    hf_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    T+=time; nT++; E+=ave; nE++;
	    TimeRMS->Fill(time_rms);
         }
      }	
      if(nT>0 && abs(eta)>29 ){ Time2Dhbhehf->Fill(eta,phi,T/nT); Time2Dhbhehf->Fill(eta,phi+1,T/nT);}	 
      if(nT>0 && abs(eta)>29 ){ Energy2Dhbhehf->Fill(eta,phi,E/nE); Energy2Dhbhehf->Fill(eta,phi+1,E/nE);}	 
   } 
   // HO histograms
   for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++){
      double T=0,nT=0,E=0,nE=0;
      for(int depth=4;depth<=4;depth++){
         if(ho_data[eta+42][phi-1][depth-1].get_statistics()>10){
            double ave,rms,time,time_rms;
	    ho_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    ho_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    T+=time; nT++; E+=ave; nE++;
	    TimeRMS->Fill(time_rms);
         }
      }
      if(nT>0){ Time2Dho->Fill(eta,phi,T/nT); Energy2Dho->Fill(eta,phi+1,E/nE) ;}
   } 

} 

void HcalDetDiagLaserMonitor::SaveReference(){}
void HcalDetDiagLaserMonitor::LoadReference(){} 

void HcalDetDiagLaserMonitor::done(){   
   if(ievt_>10){
      fillHistos();
      SaveReference();
   }   
} 
