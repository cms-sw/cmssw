#include "DQM/HcalMonitorTasks/interface/HcalDetDiagLEDMonitor.h"
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

HcalDetDiagLEDMonitor::HcalDetDiagLEDMonitor() {
  ievt_=0;
  dataset_seq_number=1;
  run_number=-1;
  IsReference=false;
}

HcalDetDiagLEDMonitor::~HcalDetDiagLEDMonitor(){}

void HcalDetDiagLEDMonitor::clearME(){
  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();
    m_dbe = 0;
  }
} 
void HcalDetDiagLEDMonitor::reset(){}

void HcalDetDiagLEDMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  m_dbe=NULL;
  ievt_=0;
  if(dbe!=NULL) m_dbe=dbe;
  clearME();
  LEDMeanTreshold  = ps.getUntrackedParameter<double>("LEDMeanTreshold" , 0.1);
  LEDRmsTreshold   = ps.getUntrackedParameter<double>("LEDRmsTreshold"  , 0.1);
  UseDB            = ps.getUntrackedParameter<bool>  ("UseDB"  , false);
  
  ReferenceData    = ps.getUntrackedParameter<string>("LEDReferenceData" ,"");
  OutputFilePath   = ps.getUntrackedParameter<string>("OutputFilePath", "");
 
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"HcalDetDiagLEDMonitor";
  char *name;
  if(m_dbe!=NULL){    
     m_dbe->setCurrentFolder(baseFolder_);   
     meEVT_ = m_dbe->bookInt("HcalDetDiagLEDMonitor Event Number");
     m_dbe->setCurrentFolder(baseFolder_+"/Summary Plots");
     
     name="HBHEHO LED Energy Distribution";               Energy         = m_dbe->book1D(name,name,200,0,3000);
     name="HBHEHO LED Timing Distribution";               Time           = m_dbe->book1D(name,name,200,0,10);
     name="HBHEHO LED Energy RMS/Energy Distribution";    EnergyRMS      = m_dbe->book1D(name,name,200,0,0.2);
     name="HBHEHO LED Timing RMS Distribution";           TimeRMS        = m_dbe->book1D(name,name,200,0,0.4);
     name="HF LED Energy Distribution";               EnergyHF       = m_dbe->book1D(name,name,200,0,3000);
     name="HF LED Timing Distribution";               TimeHF         = m_dbe->book1D(name,name,200,0,10);
     name="HF LED Energy RMS/Energy Distribution";    EnergyRMSHF    = m_dbe->book1D(name,name,200,0,0.5);
     name="HF LED Timing RMS Distribution";           TimeRMSHF      = m_dbe->book1D(name,name,200,0,0.4);
     name="LED Energy Corr(PinDiod) Distribution"; EnergyCorr     = m_dbe->book1D(name,name,200,0,10);
     name="LED Timing HBHEHF";                     Time2Dhbhehf   = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="LED Timing HO";                         Time2Dho       = m_dbe->book2D(name,name,33,-16,16,74,0,73);
     name="LED Energy HBHEHF";                     Energy2Dhbhehf = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="LED Energy HO";                         Energy2Dho     = m_dbe->book2D(name,name,33,-16,16,74,0,73);
     name="HBP Average over HPD LED Ref";          HBPphi = m_dbe->book2D(name,name,180,1,73,400,0,2);
     name="HBM Average over HPD LED Ref";          HBMphi = m_dbe->book2D(name,name,180,1,73,400,0,2);
     name="HEP Average over HPD LED Ref";          HEPphi = m_dbe->book2D(name,name,180,1,73,400,0,2);
     name="HEM Average over HPD LED Ref";          HEMphi = m_dbe->book2D(name,name,180,1,73,400,0,2);
     name="HFP Average over RM LED Ref";        HFPphi = m_dbe->book2D(name,name,180,1,37,400,0,2);
     name="HFM Average over RM LED Ref";        HFMphi = m_dbe->book2D(name,name,180,1,37,400,0,2);
     name="HO0 Average over HPD LED Ref";          HO0phi = m_dbe->book2D(name,name,180,1,49,400,0,2);
     name="HO1P Average over HPD LED Ref";         HO1Pphi= m_dbe->book2D(name,name,180,1,49,400,0,2);
     name="HO2P Average over HPD LED Ref";         HO2Pphi= m_dbe->book2D(name,name,180,1,49,400,0,2);
     name="HO1M Average over HPD LED Ref";         HO1Mphi= m_dbe->book2D(name,name,180,1,49,400,0,2);
     name="HO2M Average over HPD LED Ref";         HO2Mphi= m_dbe->book2D(name,name,180,1,49,400,0,2);
        
     setupDepthHists2D(ChannelsLEDEnergy,   "Channel LED Energy","");
     setupDepthHists2D(ChannelsLEDEnergyRef,"Channel LED Energy Reference","");
     
     m_dbe->setCurrentFolder(baseFolder_+"/channel status");
     setupDepthHists2D(ChannelStatusMissingChannels,  "Channel Status Missing Channels","");
     setupDepthHists2D(ChannelStatusUnstableChannels, "Channel Status Unstable Channels","");
     setupDepthHists2D(ChannelStatusUnstableLEDsignal,"Channel Status Unstable LED","");
     setupDepthHists2D(ChannelStatusLEDMean,          "Channel Status LED Mean","");
     setupDepthHists2D(ChannelStatusLEDRMS,           "Channel Status LED RMS","");
     setupDepthHists2D(ChannelStatusTimeMean,         "Channel Status Time Mean","");
     setupDepthHists2D(ChannelStatusTimeRMS,          "Channel Status Time RMS","");
  } 
  ReferenceRun="UNKNOWN";
  LoadReference();
  m_dbe->setCurrentFolder(baseFolder_);
  RefRun_= m_dbe->bookString("HcalDetDiagLEDMonitor Reference Run",ReferenceRun);
  
  lmap =new HcalLogicalMap(gen.createMap());
  emap=lmap->generateHcalElectronicsMap();
  return;
} 

void HcalDetDiagLEDMonitor::processEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HcalDbService& cond){
int  eta,phi,depth,nTS;
   
   if(!m_dbe) return; 
   bool LEDEvent=false;
   bool LocalRun=false;
   // for local runs 
   try{
       edm::Handle<HcalTBTriggerData> trigger_data;
       iEvent.getByType(trigger_data);
       if(trigger_data->triggerWord()==6) LEDEvent=true;
       LocalRun=true;
   }catch(...){}
   
   if(!LocalRun) return;  
   if(!LEDEvent) return; 
   
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
   try{
         edm::Handle<HcalCalibDigiCollection> calib;
         iEvent.getByType(calib);
         for(HcalCalibDigiCollection::const_iterator digi=calib->begin();digi!=calib->end();digi++){
	    if(digi->id().cboxChannel()!=0 || digi->id().hcalSubdet()==0) continue; 
	    nTS=digi->size();
	    double e=0; 
	    for(int i=0;i<nTS;i++){ data[i]=adc2fC[digi->sample(i).adc()&0xff]; e+=data[i];}
	    if(e<15000) calib_data[digi->id().hcalSubdet()][digi->id().ieta()+2][digi->id().iphi()-1].add_statistics(data,nTS);
	 }   
   }catch(...){} 
   if(((ievt_)%500)==0){
       fillHistos();
       CheckStatus(); 
   }
   return;
}


void HcalDetDiagLEDMonitor::fillHistos(){
char *subdet[4]={"HB","HE","HO","HF"};
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
            double ave,rms,time,time_rms;
	    hb_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    hb_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    TimeRMS->Fill(time_rms);
	    T+=time; nT++; E+=ave; nE++;
	    if(GetCalib("HB",eta,phi)->get_statistics()>100){
	      double ave_calib,rms_calib;
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
            double ave,rms,time,time_rms;
	    he_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    he_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    T+=time; nT++; E+=ave; nE++;
	    TimeRMS->Fill(time_rms);
	    if(GetCalib("HE",eta,phi)->get_statistics()>100){
	      double ave_calib,rms_calib;
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
            double ave,rms,time,time_rms;
	    hf_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    hf_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    EnergyHF->Fill(ave);
	    if(ave>0)EnergyRMSHF->Fill(rms/ave);
	    TimeHF->Fill(time);
	    T+=time; nT++; E+=ave; nE++;
	    TimeRMSHF->Fill(time_rms);
	    if(GetCalib("HF",eta,phi)->get_statistics()>100){
	      double ave_calib,rms_calib;
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
   for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++){
      double T=0,nT=0,E=0,nE=0;
      for(int depth=4;depth<=4;depth++){
         if(ho_data[eta+42][phi-1][depth-1].get_statistics()>100){
            double ave,rms,time,time_rms;
	    ho_data[eta+42][phi-1][depth-1].get_average_led(&ave,&rms);
	    ho_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    Energy->Fill(ave);
	    if(ave>0)EnergyRMS->Fill(rms/ave);
	    Time->Fill(time);
	    T+=time; nT++; E+=ave; nE++;
	    TimeRMS->Fill(time_rms);
	    if(GetCalib("HO",eta,phi)->get_statistics()>100){
	      double ave_calib,rms_calib;
	      GetCalib("HO",eta,phi)->get_average_led(&ave_calib,&rms_calib);
	      fill_energy("HO",eta,phi,depth,ave/ave_calib,1);
	      EnergyCorr->Fill(ave_calib/ave);
	    }
         }
      }
      if(nT>0){ Time2Dho->Fill(eta,phi,T/nT); Energy2Dho->Fill(eta,phi+1,E/nE) ;}
   } 
   ///////////////////////////////////
   double ave,rms,ave_calib,rms_calib;
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
   for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++) for(int depth=4;depth<=4;depth++){
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
  
  for(int sd=0;sd<4;sd++){
      int feta=0,teta=0,fdepth=0,tdepth=0;
      if(sd==0){ feta=-16; teta=16 ;fdepth=1; tdepth=2;}
      if(sd==1){ feta=-29; teta=29 ;fdepth=1; tdepth=3;} 
      if(sd==2){ feta=-15; teta=15 ;fdepth=4; tdepth=4;} 
      if(sd==3){ feta=-42; teta=42 ;fdepth=1; tdepth=2;} 
      for(int phi=1;phi<=72;phi++) for(int depth=fdepth;depth<=tdepth;depth++) for(int eta=feta;eta<=teta;eta++){
         if(sd==3 && eta>-29 && eta<29) continue;
         double ave =get_energy(subdet[sd],eta,phi,depth,1);
         double ref =get_energy(subdet[sd],eta,phi,depth,2);
         try{
	    HcalDetId *detid=0;
            if(sd==0) detid=new HcalDetId(HcalBarrel,eta,phi,depth);
            if(sd==1) detid=new HcalDetId(HcalEndcap,eta,phi,depth);
            if(sd==2) detid=new HcalDetId(HcalOuter,eta,phi,depth);
            if(sd==3) detid=new HcalDetId(HcalForward,eta,phi,depth);
	    HcalFrontEndId    lmap_entry=lmap->getHcalFrontEndId(*detid);
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
	    delete detid;
	 }catch(...){ continue;}
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
double led,rms,time,time_rms;
int    Eta,Phi,Depth,Statistic,Status=0;
char   Subdet[10],str[500];
    if(UseDB==false){
       sprintf(str,"%sHcalDetDiagLEDData_run%06i_%i.root",OutputFilePath.c_str(),run_number,dataset_seq_number);
       TFile *theFile = new TFile(str, "RECREATE");
       if(!theFile->IsOpen()) return;
       theFile->cd();
       char str[100]; 
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
       tree->Branch("time",     &time,           "time/D");
       tree->Branch("time_rms", &time_rms,       "time_rms/D");
       sprintf(Subdet,"HB");
       for(int eta=-16;eta<=16;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
          if((Statistic=hb_data[eta+42][phi-1][depth-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hb_data[eta+42][phi-1][depth-1].get_status();
	     hb_data[eta+42][phi-1][depth-1].get_average_led(&led,&rms);
	     hb_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"HE");
       for(int eta=-29;eta<=29;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=3;depth++){
         if((Statistic=he_data[eta+42][phi-1][depth-1].get_statistics())>100){
            Eta=eta; Phi=phi; Depth=depth;
	    Status=he_data[eta+42][phi-1][depth-1].get_status();
	    he_data[eta+42][phi-1][depth-1].get_average_led(&led,&rms);
	    he_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	    tree->Fill();
         }
       } 
       sprintf(Subdet,"HO");
       for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++) for(int depth=4;depth<=4;depth++){
         if((Statistic=ho_data[eta+42][phi-1][depth-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=ho_data[eta+42][phi-1][depth-1].get_status();
	     ho_data[eta+42][phi-1][depth-1].get_average_led(&led,&rms);
	     ho_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	     tree->Fill();
         }
       } 
       sprintf(Subdet,"HF");
       for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++) for(int depth=1;depth<=2;depth++){
         if((Statistic=hf_data[eta+42][phi-1][depth-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=depth;
	     Status=hf_data[eta+42][phi-1][depth-1].get_status();
	     hf_data[eta+42][phi-1][depth-1].get_average_led(&led,&rms);
	     hf_data[eta+42][phi-1][depth-1].get_average_time(&time,&time_rms);
	     tree->Fill();
         }
       }
       sprintf(Subdet,"CALIB_HB");
       for(int eta=-1;eta<=1;eta++) for(int phi=1;phi<=72;phi++){
          if((calib_data[1][eta+2][phi-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[1][eta+2][phi-1].get_status();
 	     calib_data[1][eta+2][phi-1].get_average_led(&led,&rms);
	     calib_data[1][eta+2][phi-1].get_average_time(&time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"CALIB_HE");
       for(int eta=-1;eta<=1;eta++) for(int phi=1;phi<=72;phi++){
          if((calib_data[2][eta+2][phi-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[2][eta+2][phi-1].get_status();
 	     calib_data[2][eta+2][phi-1].get_average_led(&led,&rms);
	     calib_data[2][eta+2][phi-1].get_average_time(&time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"CALIB_HO");
       for(int eta=-2;eta<=2;eta++) for(int phi=1;phi<=72;phi++){
          if((calib_data[3][eta+2][phi-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[3][eta+2][phi-1].get_status();
 	     calib_data[3][eta+2][phi-1].get_average_led(&led,&rms);
	     calib_data[3][eta+2][phi-1].get_average_time(&time,&time_rms);
	     tree->Fill();
          }
       } 
       sprintf(Subdet,"CALIB_HF");
       for(int eta=-2;eta<=2;eta++) for(int phi=1;phi<=72;phi++){
          if((calib_data[4][eta+2][phi-1].get_statistics())>100){
             Eta=eta; Phi=phi; Depth=0;
	     Status=calib_data[4][eta+2][phi-1].get_status();
 	     calib_data[4][eta+2][phi-1].get_average_led(&led,&rms);
	     calib_data[4][eta+2][phi-1].get_average_time(&time,&time_rms);
	     tree->Fill();
          }
       } 
       theFile->Write();
       theFile->Close();
   }
   dataset_seq_number++;
}

void HcalDetDiagLEDMonitor::LoadReference(){
double led,rms;
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
} 
void HcalDetDiagLEDMonitor::CheckStatus(){
   for(int i=0;i<6;i++){
      ChannelStatusMissingChannels[i]->Reset();
      ChannelStatusUnstableChannels[i]->Reset();
      ChannelStatusUnstableLEDsignal[i]->Reset();
      ChannelStatusLEDMean[i]->Reset();
      ChannelStatusLEDRMS[i]->Reset();
      ChannelStatusTimeMean[i]->Reset();
      ChannelStatusTimeRMS[i]->Reset();
   }
 
   
   std::vector <HcalElectronicsId> AllElIds = emap.allElectronicsIdPrecision();
   for (std::vector <HcalElectronicsId>::iterator eid = AllElIds.begin(); eid != AllElIds.end(); eid++) {
      DetId detid=emap.lookup(*eid);
      int eta=0,phi=0,depth=0;
      try{
        HcalDetId hid(detid);
        eta=hid.ieta();
        phi=hid.iphi();
        depth=hid.depth(); 
      }catch(...){ continue; } 
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
	     double ave,rms;
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
	     double ave,rms;
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
	     double ave,rms;
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
	     double ave,rms;
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
void HcalDetDiagLEDMonitor::fill_energy(char *subdet,int eta,int phi,int depth,double e,int type){ 
   int ind=-1;
   if(eta>42 || eta<-42 || eta==0) return;
   if(strcmp(subdet,"HB")==0 || strcmp(subdet,"HF")==0) if(depth==1) ind=0; else ind=1;
   else if(strcmp(subdet,"HE")==0) if(depth==3) ind=2; else ind=3+depth;
   else if(strcmp(subdet,"HO")==0) ind=3; 
   if(ind==-1) return;
   if(type==1) ChannelsLEDEnergy[ind]   ->setBinContent(eta+42,phi+1,e);
   if(type==2) ChannelsLEDEnergyRef[ind]->setBinContent(eta+42,phi+1,e);
}
double HcalDetDiagLEDMonitor::get_energy(char *subdet,int eta,int phi,int depth,int type){
   int ind=-1;
   if(strcmp(subdet,"HB")==0 || strcmp(subdet,"HF")==0) if(depth==1) ind=0; else ind=1;
   else if(strcmp(subdet,"HE")==0) if(depth==3) ind=2; else ind=3+depth;
   else if(strcmp(subdet,"HO")==0) ind=3; 
   if(ind==-1) return -1.0;
   if(type==1) return ChannelsLEDEnergy[ind]  ->getBinContent(eta+42,phi+1);
   if(type==2) return ChannelsLEDEnergyRef[ind] ->getBinContent(eta+42,phi+1);
   return -1.0;
}

void HcalDetDiagLEDMonitor::fill_channel_status(char *subdet,int eta,int phi,int depth,int type,double status){
   int ind=-1;
   if(eta>42 || eta<-42 || eta==0) return;
   if(strcmp(subdet,"HB")==0 || strcmp(subdet,"HF")==0) if(depth==1) ind=0; else ind=1;
   else if(strcmp(subdet,"HE")==0) if(depth==3) ind=2; else ind=3+depth;
   else if(strcmp(subdet,"HO")==0) ind=3; 
   if(ind==-1) return;
   if(type==1) ChannelStatusMissingChannels[ind]  ->setBinContent(eta+42,phi+1,status);
   if(type==2) ChannelStatusUnstableChannels[ind] ->setBinContent(eta+42,phi+1,status);
   if(type==3) ChannelStatusUnstableLEDsignal[ind]->setBinContent(eta+42,phi+1,status);
   if(type==4) ChannelStatusLEDMean[ind]          ->setBinContent(eta+42,phi+1,status);
   if(type==5) ChannelStatusLEDRMS[ind]           ->setBinContent(eta+42,phi+1,status);
   if(type==6) ChannelStatusTimeMean[ind]         ->setBinContent(eta+42,phi+1,status);
   if(type==7) ChannelStatusTimeRMS[ind]          ->setBinContent(eta+42,phi+1,status);
}
void HcalDetDiagLEDMonitor::done(){   
   if(ievt_>=100){
      fillHistos();
      CheckStatus();
      SaveReference();
   }   
} 
