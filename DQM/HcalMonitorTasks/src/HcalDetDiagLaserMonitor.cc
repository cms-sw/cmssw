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
  std::string name;
  if(m_dbe!=NULL){    
     m_dbe->setCurrentFolder(baseFolder_);   
     meEVT_ = m_dbe->bookInt("HcalDetDiagLaserMonitor Event Number");
     m_dbe->setCurrentFolder(baseFolder_+"/Summary Plots");
     
     name="HBHE Laser Energy Distribution";             hbheEnergy       = m_dbe->book1D(name,name,200,0,3000);
     name="HBHE Laser Timing Distribution";             hbheTime         = m_dbe->book1D(name,name,200,0,10);
     name="HBHE Laser Energy RMS/Energy Distribution";  hbheEnergyRMS    = m_dbe->book1D(name,name,200,0,0.5);
     name="HBHE Laser Timing RMS Distribution";         hbheTimeRMS      = m_dbe->book1D(name,name,200,0,1);
     name="HO Laser Energy Distribution";               hoEnergy         = m_dbe->book1D(name,name,200,0,3000);
     name="HO Laser Timing Distribution";               hoTime           = m_dbe->book1D(name,name,200,0,10);
     name="HO Laser Energy RMS/Energy Distribution";    hoEnergyRMS      = m_dbe->book1D(name,name,200,0,0.5);
     name="HO Laser Timing RMS Distribution";           hoTimeRMS        = m_dbe->book1D(name,name,200,0,1);
     name="HF Laser Energy Distribution";               hfEnergy         = m_dbe->book1D(name,name,200,0,3000);
     name="HF Laser Timing Distribution";               hfTime           = m_dbe->book1D(name,name,200,0,10);
     name="HF Laser Energy RMS/Energy Distribution";    hfEnergyRMS      = m_dbe->book1D(name,name,200,0,0.7);
     name="HF Laser Timing RMS Distribution";           hfTimeRMS        = m_dbe->book1D(name,name,200,0,1);
     
     name="Laser Timing HBHEHF";                     Time2Dhbhehf   = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="Laser Timing HO";                         Time2Dho       = m_dbe->book2D(name,name,33,-16,16,74,0,73);
     name="Laser Energy HBHEHF";                     Energy2Dhbhehf = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="Laser Energy HO";                         Energy2Dho     = m_dbe->book2D(name,name,33,-16,16,74,0,73);
     name="HBHEHF Laser (Timing-Ref)+1";             refTime2Dhbhehf   = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="HO Laser (Timing-Ref)+1";                 refTime2Dho       = m_dbe->book2D(name,name,33,-16,16,74,0,73);
     name="HBHEHF Laser Energy/Ref";                 refEnergy2Dhbhehf = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="HO Laser Energy/Ref";                     refEnergy2Dho     = m_dbe->book2D(name,name,33,-16,16,74,0,73);
     char str[100];
     for(int i=0;i<56;i++){   
        sprintf(str,"RADDAM (%i %i)",RADDAM_CH[i].eta,RADDAM_CH[i].phi);                                             
        Raddam[i] = m_dbe->book1D(str,str,10,-0.5,9.5);  
     }
  }  
  ReferenceRun="UNKNOWN";
  LoadReference();
  m_dbe->setCurrentFolder(baseFolder_);
  RefRun_= m_dbe->bookString("HcalDetDiagLaserMonitor Reference Run",ReferenceRun);

  return;
} 

void HcalDetDiagLaserMonitor::processEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HcalDbService& cond){
int  eta,phi,depth,nTS;
   if(emap==0) emap=cond.getHcalMapping();
   if(!m_dbe) return; 
   bool LaserEvent=false;
   bool LocalRun=false;
   bool LaserRaddam=false;
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
       for (int i=FEDNumbering::MINHCALFEDID;i<=FEDNumbering::MAXHCALFEDID; i++) {
          const FEDRawData& fedData = rawdata->FEDData(i) ;
          if ( fedData.size() < 24 ) continue ;
          int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ;
	  //printf("Value: %i\n",value);
          if(value==hc_HBHEHPD || value==hc_HOHPD || value==hc_HFPMT){ LaserEvent=true; break;}
	  if(value==hc_RADDAM){ LaserEvent=true; LaserRaddam=true; break;} 
       }
   }   
   if(!LaserEvent) return;
   
   ievt_++;
   meEVT_->Fill(ievt_);
   run_number=iEvent.id().run();
   double data[20];
   if(!LaserRaddam){
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
	     if((eta>=11 && eta<=15 && phi>=59 && phi<=70) || (eta>=5 && eta<=10 && phi>=47 && phi<=58)){
	        for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-11.0;
	     }else{
	        for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()&0xff]-2.5;
	     }
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
   }else{ //Raddam
      try{
         edm::Handle<HFDigiCollection> hf;
         iEvent.getByType(hf);
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     int N;
	     for(N=0;N<56;N++)if(eta==RADDAM_CH[N].eta && phi==RADDAM_CH[N].phi) break;
	     if(N==56) continue;      
	     for(int i=0;i<nTS;i++) Raddam[N]->Fill(i,adc2fC[digi->sample(i).adc()&0xff]-2.5);
	     
         }   
      }catch(...){}    
   
   
    //printf("RADDAM\n");
   } 
   if(((ievt_)%50)==0){
       fillHistos();
   }
   return;
}


void HcalDetDiagLaserMonitor::fillHistos(){
   hbheEnergy->Reset();
   hbheTime->Reset();
   hbheEnergyRMS->Reset();
   hbheTimeRMS->Reset();
   hoEnergy->Reset();
   hoTime->Reset();
   hoEnergyRMS->Reset();
   hoTimeRMS->Reset();
   hfEnergy->Reset();
   hfTime->Reset();
   hfEnergyRMS->Reset();
   hfTimeRMS->Reset();
   
   Time2Dhbhehf->Reset();
   Time2Dho->Reset();
   Energy2Dhbhehf->Reset();
   Energy2Dho->Reset();
   refTime2Dhbhehf->Reset();
   refTime2Dho->Reset();
   refEnergy2Dhbhehf->Reset();
   refEnergy2Dho->Reset();
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
      if(nT>0){Time2Dhbhehf->Fill(eta,phi,T/nT);Energy2Dhbhehf->Fill(eta,phi,E/nE); }
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
      if(nT>0 && abs(eta)>16 ){Time2Dhbhehf->Fill(eta,phi,T/nT);   Energy2Dhbhehf->Fill(eta,phi,E/nE); }	 
      if(nT>0 && abs(eta)>20 ){Time2Dhbhehf->Fill(eta,phi+1,T/nT); Energy2Dhbhehf->Fill(eta,phi+1,E/nE);}	 
   } 
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
      if(nT>0 && abs(eta)>29 ){ Time2Dhbhehf->Fill(eta,phi,T/nT); Time2Dhbhehf->Fill(eta,phi+1,T/nT);}	 
      if(nT>0 && abs(eta)>29 ){ Energy2Dhbhehf->Fill(eta,phi,E/nE); Energy2Dhbhehf->Fill(eta,phi+1,E/nE);}	 
   } 
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
   
   // compare with reference...
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
      if(nE>0) refEnergy2Dhbhehf->Fill(eta,phi,E/nE);  
      if(nT>0){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01;  refTime2Dhbhehf->Fill(eta,phi,TTT); } 
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
      if(nE>0 && abs(eta)>16) refEnergy2Dhbhehf->Fill(eta,phi,E/nE);  
      if(nT>0 && abs(eta)>16){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01;  refTime2Dhbhehf->Fill(eta,phi,TTT); } 
      if(nE>0 && abs(eta)>20) refEnergy2Dhbhehf->Fill(eta,phi+1,E/nE);  
      if(nT>0 && abs(eta)>20){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01;  refTime2Dhbhehf->Fill(eta,phi+1,TTT); }  
   } 
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
      if(nE>0 && abs(eta)>29) refEnergy2Dhbhehf->Fill(eta,phi,E/nE);  
      if(nT>0 && abs(eta)>29){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01; refTime2Dhbhehf->Fill(eta,phi,TTT); }
      if(nE>0 && abs(eta)>29) refEnergy2Dhbhehf->Fill(eta,phi+1,E/nE);  
      if(nT>0 && abs(eta)>29){ double TTT=T/nT+1; if(TTT<0.01) TTT=0.01; refTime2Dhbhehf->Fill(eta,phi+1,TTT); } 
   } 
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

void HcalDetDiagLaserMonitor::SaveReference(){
double amp,rms,time,time_rms;
int    Eta,Phi,Depth,Statistic,Status=0;
char   Subdet[10],str[500];
    if(UseDB==false){
       sprintf(str,"%sHcalDetDiagLaserData_run%06i_%i.root",OutputFilePath.c_str(),run_number,dataset_seq_number);
       TFile *theFile = new TFile(str, "RECREATE");
       if(!theFile->IsOpen()) return;
       theFile->cd();
       char str[100]; 
       sprintf(str,"%d",run_number); TObjString run(str);    run.Write("run number");
       sprintf(str,"%d",ievt_);      TObjString events(str); events.Write("Total events processed");
       
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
       tree->Branch("time",     &time,           "time/D");
       tree->Branch("time_rms", &time_rms,       "time_rms/D");
       sprintf(Subdet,"HB");
       for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
          if((Statistic=hb_data[eta+42][phi-1][depth-1].get_statistics())>10){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hb_data[eta+42][phi-1][depth-1].get_status();
	     hb_data[eta+42][phi-1][depth-1].get_average_amp(&amp,&rms);
	     hb_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"HE");
       for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=3;depth++){
         if((Statistic=he_data[eta+42][phi-1][depth-1].get_statistics())>10){
            Eta=eta; Phi=phi; Depth=depth;
	    Status=he_data[eta+42][phi-1][depth-1].get_status();
	    he_data[eta+42][phi-1][depth-1].get_average_amp(&amp,&rms);
	    he_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    tree->Fill();
         }
       } 
       sprintf(Subdet,"HO");
       for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++) for(int depth=4;depth<=4;depth++){
         if((Statistic=ho_data[eta+42][phi-1][depth-1].get_statistics())>10){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=ho_data[eta+42][phi-1][depth-1].get_status();
	     ho_data[eta+42][phi-1][depth-1].get_average_amp(&amp,&rms);
	     ho_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	     tree->Fill();
         }
       } 
       sprintf(Subdet,"HF");
       for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
         if((Statistic=hf_data[eta+42][phi-1][depth-1].get_statistics())>10){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hf_data[eta+42][phi-1][depth-1].get_status();
	     hf_data[eta+42][phi-1][depth-1].get_average_amp(&amp,&rms);
	     hf_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	     tree->Fill();
         }
       }

       theFile->Write();
       theFile->Close();
   }
   dataset_seq_number++;
}
void HcalDetDiagLaserMonitor::LoadReference(){
double amp,rms,time,time_rms;
int Eta,Phi,Depth;
char subdet[10];
TFile *f;
   if(UseDB==false){
      try{ 
         f = new TFile(ReferenceData.c_str(),"READ");
      }catch(...){ return ;}
      if(!f->IsOpen()) return ;
      TObjString *STR=(TObjString *)f->Get("run number");
      
      if(STR){ string Ref(STR->String()); ReferenceRun=Ref;}
      
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

} 

void HcalDetDiagLaserMonitor::done(){   
   if(ievt_>10){
      fillHistos();
      SaveReference();
   }   
} 
