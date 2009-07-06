#include "DQM/HcalMonitorTasks/interface/HcalDetDiagPedestalMonitor.h"
#include "TFile.h"
#include "TTree.h"


#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"
                                                                                                                                                  

HcalDetDiagPedestalMonitor::HcalDetDiagPedestalMonitor() {
  ievt_=0;
  time_min=time_max=0;
  dataset_seq_number=1;
  run_number=-1;
  IsReference=false;
}

HcalDetDiagPedestalMonitor::~HcalDetDiagPedestalMonitor(){}

void HcalDetDiagPedestalMonitor::clearME(){
  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();
    m_dbe = 0;
  }
} 
void HcalDetDiagPedestalMonitor::reset(){}

void HcalDetDiagPedestalMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  m_dbe=NULL;
  ievt_=0;
  if(dbe!=NULL) m_dbe=dbe;
  clearME();
  inputLabelDigi_  = ps.getParameter<edm::InputTag>("digiLabel");
  
  HBMeanTreshold   = ps.getUntrackedParameter<double>("HBMeanPedestalTreshold" , 0.1);
  HBRmsTreshold    = ps.getUntrackedParameter<double>("HBRmsPedestalTreshold"  , 0.1);
  HEMeanTreshold   = ps.getUntrackedParameter<double>("HEMeanPedestalTreshold" , 0.1);
  HERmsTreshold    = ps.getUntrackedParameter<double>("HERmsPedestalTreshold"  , 0.1);
  HOMeanTreshold   = ps.getUntrackedParameter<double>("HOMeanPedestalTreshold" , 0.1);
  HORmsTreshold    = ps.getUntrackedParameter<double>("HORmsPedestalTreshold"  , 0.1);
  HFMeanTreshold   = ps.getUntrackedParameter<double>("HFMeanPedestalTreshold" , 0.1);
  HFRmsTreshold    = ps.getUntrackedParameter<double>("HFRmsPedestalTreshold"  , 0.1);
  UseDB            = ps.getUntrackedParameter<bool>  ("UseDB"  , false);

  ReferenceData    = ps.getUntrackedParameter<string>("PedReferenceData" , "");
  OutputFilePath   = ps.getUntrackedParameter<string>("OutputFilePath", "");
 
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"HcalDetDiagPedestalMonitor";
  char *name;
  if(m_dbe!=NULL){    
     m_dbe->setCurrentFolder(baseFolder_);   
     meEVT_ = m_dbe->bookInt("HcalDetDiagPedestalMonitor Event Number");
     
     m_dbe->setCurrentFolder(baseFolder_+"/Summary Plots");
     name="HB Pedestal Distribution (avarage over 4 caps)";           PedestalsAve4HB = m_dbe->book1D(name,name,200,0,6);
     name="HE Pedestal Distribution (avarage over 4 caps)";           PedestalsAve4HE = m_dbe->book1D(name,name,200,0,6);
     name="HO Pedestal Distribution (avarage over 4 caps)";           PedestalsAve4HO = m_dbe->book1D(name,name,200,0,6);
     name="HF Pedestal Distribution (avarage over 4 caps)";           PedestalsAve4HF = m_dbe->book1D(name,name,200,0,6);
     name="SIPM Pedestal Distribution (avarage over 4 caps)";         PedestalsAve4Simp = m_dbe->book1D(name,name,200,5,15);
     name="ZDC Pedestal Distribution (avarage over 4 caps)";          PedestalsAve4ZDC  = m_dbe->book1D(name,name,200,0,15);
     name="HB Pedestal Reference Distribution (avarage over 4 caps)"; PedestalsRefAve4HB = m_dbe->book1D(name,name,200,0,6);
     name="HE Pedestal Reference Distribution (avarage over 4 caps)"; PedestalsRefAve4HE = m_dbe->book1D(name,name,200,0,6);
     name="HO Pedestal Reference Distribution (avarage over 4 caps)"; PedestalsRefAve4HO = m_dbe->book1D(name,name,200,0,6);
     name="HF Pedestal Reference Distribution (avarage over 4 caps)"; PedestalsRefAve4HF = m_dbe->book1D(name,name,200,0,6);
     name="SIPM Pedestal Reference Distribution (avarage over 4 caps)"; PedestalsRefAve4Simp = m_dbe->book1D(name,name,200,5,15);
     name="ZDC Pedestal Reference Distribution (avarage over 4 caps)";  PedestalsRefAve4ZDC  = m_dbe->book1D(name,name,200,0,15);
     
     name="HB Pedestal-Reference Distribution (avarage over 4 caps)"; PedestalsAve4HBref= m_dbe->book1D(name,name,1500,-3,3);
     name="HE Pedestal-Reference Distribution (avarage over 4 caps)"; PedestalsAve4HEref= m_dbe->book1D(name,name,1500,-3,3);
     name="HO Pedestal-Reference Distribution (avarage over 4 caps)"; PedestalsAve4HOref= m_dbe->book1D(name,name,1500,-3,3);
     name="HF Pedestal-Reference Distribution (avarage over 4 caps)"; PedestalsAve4HFref= m_dbe->book1D(name,name,1500,-3,3);
    
     name="HB Pedestal RMS Distribution (individual cap)";            PedestalsRmsHB = m_dbe->book1D(name,name,200,0,2);
     name="HE Pedestal RMS Distribution (individual cap)";            PedestalsRmsHE = m_dbe->book1D(name,name,200,0,2);
     name="HO Pedestal RMS Distribution (individual cap)";            PedestalsRmsHO = m_dbe->book1D(name,name,200,0,2);
     name="HF Pedestal RMS Distribution (individual cap)";            PedestalsRmsHF = m_dbe->book1D(name,name,200,0,2);
     name="SIPM Pedestal RMS Distribution (individual cap)";          PedestalsRmsSimp = m_dbe->book1D(name,name,200,0,4);
     name="ZDC Pedestal RMS Distribution (individual cap)";           PedestalsRmsZDC = m_dbe->book1D(name,name,200,0,3);
     
     name="HB Pedestal Reference RMS Distribution (individual cap)";  PedestalsRmsRefHB = m_dbe->book1D(name,name,200,0,2);
     name="HE Pedestal Reference RMS Distribution (individual cap)";  PedestalsRmsRefHE = m_dbe->book1D(name,name,200,0,2);
     name="HO Pedestal Reference RMS Distribution (individual cap)";  PedestalsRmsRefHO = m_dbe->book1D(name,name,200,0,2);
     name="HF Pedestal Reference RMS Distribution (individual cap)";  PedestalsRmsRefHF = m_dbe->book1D(name,name,200,0,2);
     name="SIPM Pedestal Reference RMS Distribution (individual cap)";PedestalsRmsRefSimp = m_dbe->book1D(name,name,200,0,4);
     name="ZDC Pedestal Reference RMS Distribution (individual cap)"; PedestalsRmsRefZDC = m_dbe->book1D(name,name,200,0,3);
     name="HB Pedestal_rms-Reference_rms Distribution";               PedestalsRmsHBref = m_dbe->book1D(name,name,1500,-3,3);
     name="HE Pedestal_rms-Reference_rms Distribution";               PedestalsRmsHEref = m_dbe->book1D(name,name,1500,-3,3);
     name="HO Pedestal_rms-Reference_rms Distribution";               PedestalsRmsHOref = m_dbe->book1D(name,name,1500,-3,3);
     name="HF Pedestal_rms-Reference_rms Distribution";               PedestalsRmsHFref = m_dbe->book1D(name,name,1500,-3,3);
     
     name="HBHEHF pedestal mean map";       Pedestals2DHBHEHF      = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="HO pedestal mean map";           Pedestals2DHO          = m_dbe->book2D(name,name,33,-16,16,74,0,73);
     name="HBHEHF pedestal rms map";        Pedestals2DRmsHBHEHF   = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="HO pedestal rms map";            Pedestals2DRmsHO       = m_dbe->book2D(name,name,33,-16,16,74,0,73);
     name="HBHEHF pedestal problems map";   Pedestals2DErrorHBHEHF = m_dbe->book2D(name,name,87,-43,43,74,0,73);
     name="HO pedestal problems map";       Pedestals2DErrorHO     = m_dbe->book2D(name,name,33,-16,16,74,0,73);

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
     PedestalsAve4ZDC->setAxisTitle("ADC counts",1);
     PedestalsRefAve4HB->setAxisTitle("ADC counts",1);
     PedestalsRefAve4HE->setAxisTitle("ADC counts",1);
     PedestalsRefAve4HO->setAxisTitle("ADC counts",1);
     PedestalsRefAve4HF->setAxisTitle("ADC counts",1);
     PedestalsRefAve4Simp->setAxisTitle("ADC counts",1);
     PedestalsRefAve4ZDC->setAxisTitle("ADC counts",1);
     PedestalsAve4HBref->setAxisTitle("ADC counts",1);
     PedestalsAve4HEref->setAxisTitle("ADC counts",1);
     PedestalsAve4HOref->setAxisTitle("ADC counts",1);
     PedestalsAve4HFref->setAxisTitle("ADC counts",1);
     PedestalsRmsHB->setAxisTitle("ADC counts",1);
     PedestalsRmsHE->setAxisTitle("ADC counts",1);
     PedestalsRmsHO->setAxisTitle("ADC counts",1);
     PedestalsRmsHF->setAxisTitle("ADC counts",1);
     PedestalsRmsSimp->setAxisTitle("ADC counts",1);
     PedestalsRmsZDC->setAxisTitle("ADC counts",1);
     PedestalsRmsRefHB->setAxisTitle("ADC counts",1);
     PedestalsRmsRefHE->setAxisTitle("ADC counts",1);
     PedestalsRmsRefHO->setAxisTitle("ADC counts",1);
     PedestalsRmsRefHF->setAxisTitle("ADC counts",1);
     PedestalsRmsRefSimp->setAxisTitle("ADC counts",1);
     PedestalsRmsRefZDC->setAxisTitle("ADC counts",1);
     PedestalsRmsHBref->setAxisTitle("ADC counts",1);
     PedestalsRmsHEref->setAxisTitle("ADC counts",1);
     PedestalsRmsHOref->setAxisTitle("ADC counts",1);
     PedestalsRmsHFref->setAxisTitle("ADC counts",1);
  
     m_dbe->setCurrentFolder(baseFolder_+"/channel status");
     setupDepthHists2D(ChannelStatusMissingChannels,  "Channel Status Missing Channels","");
     setupDepthHists2D(ChannelStatusUnstableChannels, "Channel Status Unstable Channels","");
     setupDepthHists2D(ChannelStatusBadPedestalMean,  "Channel Status Pedestal Mean","");
     setupDepthHists2D(ChannelStatusBadPedestalRMS,   "Channel Status Pedestal RMS","");
     
     ReferenceRun="UNKNOWN";
     LoadReference();
     m_dbe->setCurrentFolder(baseFolder_);
     RefRun_= m_dbe->bookString("HcalDetDiagPedestalMonitor Reference Run",ReferenceRun);
  }
  emap=0;
  return;
} 

void HcalDetDiagPedestalMonitor::processEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HcalDbService& cond){
int  eta,phi,depth,side,chan,nTS;
   if(emap==0) emap=cond.getHcalMapping();
   if(!m_dbe) return; 
   bool PedestalEvent=false;
   bool LocalRun=false;
    
   // for local runs 
   try{
       edm::Handle<HcalTBTriggerData> trigger_data;
       iEvent.getByType(trigger_data);
       if(trigger_data->triggerWord()==5) PedestalEvent=true;
       LocalRun=true;
   }catch(...){}
   if(LocalRun && !PedestalEvent) return; 
  
   // Abort Gap pedestals 
   int calibType = -1 ;
   if(LocalRun==false){
       edm::Handle<FEDRawDataCollection> rawdata;
       iEvent.getByType(rawdata);
       //checking FEDs for calibration information
       for (int i=FEDNumbering::getHcalFEDIds().first;i<=FEDNumbering::getHcalFEDIds().second; i++) {
          const FEDRawData& fedData = rawdata->FEDData(i) ;
          if ( fedData.size() < 24 ) continue ;
          int value = ((const HcalDCCHeader*)(fedData.data()))->getCalibType() ;
          if ( calibType < 0 )  calibType = value ;
       }
       if(calibType!=1) return; 
   }
   if(iEvent.time()>time_max) time_max=iEvent.time();
   if(iEvent.time()<time_min) time_min=iEvent.time();
   
   ievt_++;
   meEVT_->Fill(ievt_);
   run_number=iEvent.id().run();
  
   try{
         edm::Handle<HBHEDigiCollection> hbhe; 
         iEvent.getByLabel(inputLabelDigi_,hbhe);
	 if(hbhe->size()<10 && calibType==1){
             ievt_--;
             meEVT_->Fill(ievt_);
             return;	 
	 }
         for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
             if(nTS>8) nTS=8;
	     if(nTS<8) continue;
	     if(digi->id().subdet()==HcalBarrel){
		for(int i=0;i<nTS;i++) hb_data[eta+42][phi-1][depth-1][digi->sample(i).capid()].add_statistics(digi->sample(i).adc());
	     }	 
             if(digi->id().subdet()==HcalEndcap){
		for(int i=0;i<nTS;i++) he_data[eta+42][phi-1][depth-1][digi->sample(i).capid()].add_statistics(digi->sample(i).adc());
	     }
         }   
   }catch(...){}      
   try{
         edm::Handle<HODigiCollection> ho; 
         iEvent.getByLabel(inputLabelDigi_,ho);
         for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     if(nTS>8) nTS=8;
	     if(nTS<8) continue;
             for(int i=0;i<nTS;i++) ho_data[eta+42][phi-1][depth-1][digi->sample(i).capid()].add_statistics(digi->sample(i).adc());
         }   
   }catch(...){}  
   try{
         edm::Handle<HFDigiCollection> hf;
         iEvent.getByLabel(inputLabelDigi_,hf);
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
             eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	     if(nTS>8) nTS=8;
	     if(nTS<8) continue;
	     for(int i=0;i<nTS;i++) hf_data[eta+42][phi-1][depth-1][digi->sample(i).capid()].add_statistics(digi->sample(i).adc());
         }   
   }catch(...){}    
//    try{
//          edm::Handle<HcalCalibDigiCollection> calib;
//          iEvent.getByType(calib);
//          //for(HcalCalibDigiCollection::const_iterator digi=calib->begin();digi!=calib->end();digi++){}   
//    }catch(...){} 
   try{
         edm::Handle<ZDCDigiCollection> zdc;
         iEvent.getByLabel(inputLabelDigi_,zdc);
         for(ZDCDigiCollection::const_iterator digi=zdc->begin();digi!=zdc->end();digi++){
             side=digi->id().zside(); depth=digi->id().depth(); chan=digi->id().channel(); nTS=digi->size();
	     if(nTS>8) nTS=8;
	     if(nTS<8) continue;
	     if(side<-1 || side>1 || chan<1 || chan>5 || depth<1 || depth>5) continue;
	     for(int i=0;i<nTS;i++) zdc_data[side+1][chan][depth][digi->sample(i).capid()].add_statistics(digi->sample(i).adc());
         }   
   }catch(...){}    
   
   
   if(((ievt_)%100)==0){
       fillHistos();
       CheckStatus(); 
   }
   return;
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
   PedestalsRmsZDC->Reset();
   PedestalsAve4ZDC->Reset();
   
   Pedestals2DRmsHBHEHF->Reset();
   Pedestals2DRmsHO->Reset();
   Pedestals2DHBHEHF->Reset();
   Pedestals2DHO->Reset();
   // HBHEHF summary map
   for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++){ 
      double PED=0,RMS=0,nped=0,nrms=0,ave,rms;
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
   for(int eta=-42;eta<=42;eta++) for(int phi=1;phi<=72;phi++){ 
      double PED=0,RMS=0,nped=0,nrms=0,ave,rms;
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
          double ave,rms,sum=0;
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
          double ave,rms,sum=0;
	  he_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); sum+=ave; PedestalsRmsHE->Fill(rms);
	  he_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); sum+=ave; PedestalsRmsHE->Fill(rms);
	  he_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); sum+=ave; PedestalsRmsHE->Fill(rms);
	  he_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); sum+=ave; PedestalsRmsHE->Fill(rms);
	  PedestalsAve4HE->Fill(sum/4.0);
      }
   } 
   // HO histograms
   for(int eta=-15;eta<=15;eta++) for(int phi=1;phi<=72;phi++) for(int depth=4;depth<=4;depth++){
      if(ho_data[eta+42][phi-1][depth-1][0].get_statistics()>100){
          double ave,rms,sum=0;
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
          double ave,rms,sum=0;
	  hf_data[eta+42][phi-1][depth-1][0].get_average(&ave,&rms); sum+=ave; PedestalsRmsHF->Fill(rms);
	  hf_data[eta+42][phi-1][depth-1][1].get_average(&ave,&rms); sum+=ave; PedestalsRmsHF->Fill(rms);
	  hf_data[eta+42][phi-1][depth-1][2].get_average(&ave,&rms); sum+=ave; PedestalsRmsHF->Fill(rms);
	  hf_data[eta+42][phi-1][depth-1][3].get_average(&ave,&rms); sum+=ave; PedestalsRmsHF->Fill(rms);
	  PedestalsAve4HF->Fill(sum/4.0);
      }
   } 
   // ZDC histograms
   for(int side=-1;side<=1;side++) for(int chan=1;chan<=5;chan++) for(int depth=1;depth<=5;depth++){
      if(zdc_data[side+1][chan][depth][0].get_statistics()>100){
          double ave,rms,sum=0;
	  zdc_data[side+1][chan][depth][0].get_average(&ave,&rms); sum+=ave; PedestalsRmsZDC->Fill(rms);
	  zdc_data[side+1][chan][depth][1].get_average(&ave,&rms); sum+=ave; PedestalsRmsZDC->Fill(rms);
	  zdc_data[side+1][chan][depth][2].get_average(&ave,&rms); sum+=ave; PedestalsRmsZDC->Fill(rms);
	  zdc_data[side+1][chan][depth][3].get_average(&ave,&rms); sum+=ave; PedestalsRmsZDC->Fill(rms);
	  PedestalsAve4ZDC->Fill(sum/4.0);
      }
   } 
} 

void HcalDetDiagPedestalMonitor::SaveReference(){
double ped[4],rms[4];
int    Eta,Phi,Depth,Statistic,Status=0;
char   Subdet[10],str[500];
    if(UseDB==false){
       sprintf(str,"%sHcalDetDiagPedestalData_run%06i_%i.root",OutputFilePath.c_str(),run_number,dataset_seq_number);
       TFile *theFile = new TFile(str, "RECREATE");
       if(!theFile->IsOpen()) return;
       theFile->cd();
       char str[100]; 
       sprintf(str,"%d",run_number); TObjString run(str);    run.Write("run number");
       sprintf(str,"%d",ievt_);      TObjString events(str); events.Write("Total events processed");
       
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
      sprintf(Subdet,"ZDC");
      for(int side=-1;side<=1;side++) for(int chan=1;chan<=5;chan++) for(int depth=1;depth<=5;depth++){
         if((Statistic=zdc_data[side+1][chan][depth][0].get_statistics())>100){
             Eta=side; Phi=chan; Depth=depth;
	     Status=zdc_data[side+1][chan][depth][0].get_status();
	     zdc_data[side+1][chan][depth][0].get_average(&ped[0],&rms[0]);
	     zdc_data[side+1][chan][depth][1].get_average(&ped[1],&rms[1]);
	     zdc_data[side+1][chan][depth][2].get_average(&ped[2],&rms[2]);
	     zdc_data[side+1][chan][depth][3].get_average(&ped[3],&rms[3]);
	     tree->Fill();
         }
      }     
      theFile->Write();
      theFile->Close();
   }
   dataset_seq_number++;
}

void HcalDetDiagPedestalMonitor::LoadReference(){
double ped[4],rms[4];
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
	    PedestalsRefAve4HB->Fill((ped[0]+ped[1]+ped[2]+ped[3])/4.0);
	    PedestalsRmsRefHB->Fill(rms[0]);
	    PedestalsRmsRefHB->Fill(rms[1]);
	    PedestalsRmsRefHB->Fill(rms[2]);
	    PedestalsRmsRefHB->Fill(rms[3]);
	    hb_data[Eta+42][Phi-1][Depth-1][0].set_reference(ped[0],rms[0]);
	    hb_data[Eta+42][Phi-1][Depth-1][1].set_reference(ped[1],rms[1]);
	    hb_data[Eta+42][Phi-1][Depth-1][2].set_reference(ped[2],rms[2]);
	    hb_data[Eta+42][Phi-1][Depth-1][3].set_reference(ped[3],rms[3]);
	 }
	 if(strcmp(subdet,"HE")==0){
	    PedestalsRefAve4HE->Fill((ped[0]+ped[1]+ped[2]+ped[3])/4.0);
	    PedestalsRmsRefHE->Fill(rms[0]);
	    PedestalsRmsRefHE->Fill(rms[1]);
	    PedestalsRmsRefHE->Fill(rms[2]);
	    PedestalsRmsRefHE->Fill(rms[3]);
	    he_data[Eta+42][Phi-1][Depth-1][0].set_reference(ped[0],rms[0]);
	    he_data[Eta+42][Phi-1][Depth-1][1].set_reference(ped[1],rms[1]);
	    he_data[Eta+42][Phi-1][Depth-1][2].set_reference(ped[2],rms[2]);
	    he_data[Eta+42][Phi-1][Depth-1][3].set_reference(ped[3],rms[3]);
	 }
	 if(strcmp(subdet,"HO")==0){
	    if((Eta>=11 && Eta<=15 && Phi>=59 && Phi<=70) || (Eta>=5 && Eta<=10 && Phi>=47 && Phi<=58)){
	       PedestalsRefAve4Simp->Fill((ped[0]+ped[1]+ped[2]+ped[3])/4.0);
	       PedestalsRmsRefSimp->Fill(rms[0]);
	       PedestalsRmsRefSimp->Fill(rms[1]);
	       PedestalsRmsRefSimp->Fill(rms[2]);
	       PedestalsRmsRefSimp->Fill(rms[3]);
	    }else{
	       PedestalsRefAve4HO->Fill((ped[0]+ped[1]+ped[2]+ped[3])/4.0);
	       PedestalsRmsRefHO->Fill(rms[0]);
	       PedestalsRmsRefHO->Fill(rms[1]);
	       PedestalsRmsRefHO->Fill(rms[2]);
	       PedestalsRmsRefHO->Fill(rms[3]);	    
	    }
	    ho_data[Eta+42][Phi-1][Depth-1][0].set_reference(ped[0],rms[0]);
	    ho_data[Eta+42][Phi-1][Depth-1][1].set_reference(ped[1],rms[1]);
	    ho_data[Eta+42][Phi-1][Depth-1][2].set_reference(ped[2],rms[2]);
	    ho_data[Eta+42][Phi-1][Depth-1][3].set_reference(ped[3],rms[3]);
	 }
	 if(strcmp(subdet,"HF")==0){
	    PedestalsRefAve4HF->Fill((ped[0]+ped[1]+ped[2]+ped[3])/4.0);
	    PedestalsRmsRefHF->Fill(rms[0]);
	    PedestalsRmsRefHF->Fill(rms[1]);
	    PedestalsRmsRefHF->Fill(rms[2]);
	    PedestalsRmsRefHF->Fill(rms[3]);
	    hf_data[Eta+42][Phi-1][Depth-1][0].set_reference(ped[0],rms[0]);
	    hf_data[Eta+42][Phi-1][Depth-1][1].set_reference(ped[1],rms[1]);
	    hf_data[Eta+42][Phi-1][Depth-1][2].set_reference(ped[2],rms[2]);
	    hf_data[Eta+42][Phi-1][Depth-1][3].set_reference(ped[3],rms[3]);
	 }
	 if(strcmp(subdet,"ZDC")==0){
	    if(Eta<-1 || Eta>1 || Phi<1 || Phi>5 || Depth<1 || Depth>5) continue;
	    PedestalsRefAve4ZDC->Fill((ped[0]+ped[1]+ped[2]+ped[3])/4.0);
	    PedestalsRmsRefZDC->Fill(rms[0]);
	    PedestalsRmsRefZDC->Fill(rms[1]);
	    PedestalsRmsRefZDC->Fill(rms[2]);
	    PedestalsRmsRefZDC->Fill(rms[3]);
	    zdc_data[Eta+1][Phi][Depth][0].set_reference(ped[0],rms[0]);
	    zdc_data[Eta+1][Phi][Depth][1].set_reference(ped[1],rms[1]);
	    zdc_data[Eta+1][Phi][Depth][2].set_reference(ped[2],rms[2]);
	    zdc_data[Eta+1][Phi][Depth][3].set_reference(ped[3],rms[3]);
	 }
      }
      f->Close();
      IsReference=true;
   }
} 

void HcalDetDiagPedestalMonitor::CheckStatus(){
   for(int i=0;i<6;i++){
      ChannelStatusMissingChannels[i]->Reset();
      ChannelStatusUnstableChannels[i]->Reset();
      ChannelStatusBadPedestalMean[i]->Reset();
      ChannelStatusBadPedestalRMS[i]->Reset();
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
     int eta=0,phi=0,depth=0;
     try{
       HcalDetId hid(detid);
       eta=hid.ieta();
       phi=hid.iphi();
       depth=hid.depth(); 
     }catch(...){ continue; } 
   
     if(detid.subdetId()==HcalBarrel){
          int ovf=hb_data[eta+42][phi-1][depth-1][0].get_overflow();
	  int stat=hb_data[eta+42][phi-1][depth-1][0].get_statistics()+ovf;
	  double status=0;
	  double ped[4],rms[4],ped_ref[4],rms_ref[4]; 
	  if(stat==0){ status=1;                                                fill_channel_status("HB",eta,phi,depth,1,status); }
	  if(status) hb_data[eta+42][phi-1][depth-1][0].change_status(1); 
	  if(stat>0 && stat!=(ievt_*2)){ status=(double)stat/(double)(ievt_*2); 
	      if(status<0.995){ 
	        fill_channel_status("HB",eta,phi,depth,2,status); 
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
	        double tmp=rms[0]-rms_ref[0]; PedestalsRmsHBref->Fill(tmp); if(tmp<0) tmp=-tmp;
		if(tmp>deltaRms) deltaRms=tmp;
	     }
	     if(deltaPed>HBMeanTreshold){ fill_channel_status("HB",eta,phi,depth,3,deltaPed); Pedestals2DErrorHBHEHF->Fill(eta,phi,1);}
	     if(deltaRms>HBRmsTreshold){  fill_channel_status("HB",eta,phi,depth,4,deltaRms); Pedestals2DErrorHBHEHF->Fill(eta,phi,1);}
	  } 
      }
      if(detid.subdetId()==HcalEndcap){
          int ovf=he_data[eta+42][phi-1][depth-1][0].get_overflow();
	  int stat=he_data[eta+42][phi-1][depth-1][0].get_statistics()+ovf;
	  double status=0; 
	  double ped[4],rms[4],ped_ref[4],rms_ref[4]; 
	  if(stat==0){ status=1;                                                fill_channel_status("HE",eta,phi,depth,1,status); }
	  if(status) he_data[eta+42][phi-1][depth-1][0].change_status(1); 
	  if(stat>0 && stat!=(ievt_*2)){ status=(double)stat/(double)(ievt_*2);
	     if(status<0.995){ 
	        fill_channel_status("HE",eta,phi,depth,2,status); 
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
	        double tmp=rms[0]-rms_ref[0]; PedestalsRmsHEref->Fill(tmp); if(tmp<0) tmp=-tmp;
		if(tmp>deltaRms) deltaRms=tmp;
	     }
	     if(deltaPed>HEMeanTreshold){ fill_channel_status("HE",eta,phi,depth,3,deltaPed); Pedestals2DErrorHBHEHF->Fill(eta,phi,1);}
	     if(deltaRms>HERmsTreshold){  fill_channel_status("HE",eta,phi,depth,4,deltaRms); Pedestals2DErrorHBHEHF->Fill(eta,phi,1);}
	  } 
      }
      if(detid.subdetId()==HcalOuter){
          int ovf=ho_data[eta+42][phi-1][depth-1][0].get_overflow(); 
	  int stat=ho_data[eta+42][phi-1][depth-1][0].get_statistics()+ovf;
	  double status=0; 
	  double ped[4],rms[4],ped_ref[4],rms_ref[4]; 
	  if(stat==0){ status=1;                                                fill_channel_status("HO",eta,phi,depth,1,status); }
	  if(status) ho_data[eta+42][phi-1][depth-1][0].change_status(1); 
	  if(stat>0 && stat!=(ievt_*2)){ status=(double)stat/(double)(ievt_*2); 
	     if(status<0.995){ 
	       fill_channel_status("HO",eta,phi,depth,2,status); 
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
	        double tmp=rms[0]-rms_ref[0]; PedestalsRmsHOref->Fill(tmp); if(tmp<0) tmp=-tmp;
		if(tmp>deltaRms) deltaRms=tmp;
	     }
	     if(deltaPed>HOMeanTreshold){ fill_channel_status("HO",eta,phi,depth,3,deltaPed); Pedestals2DErrorHO->Fill(eta,phi,1);}
	     if(deltaRms>THRESTHOLD){  fill_channel_status("HO",eta,phi,depth,4,deltaRms); Pedestals2DErrorHO->Fill(eta,phi,1);}
	  } 
      }
      if(detid.subdetId()==HcalForward){
          int ovf=hf_data[eta+42][phi-1][depth-1][0].get_overflow();
	  int stat=hf_data[eta+42][phi-1][depth-1][0].get_statistics()+ovf;
	  double status=0; 
	  double ped[4],rms[4],ped_ref[4],rms_ref[4]; 
	  if(stat==0){ status=1;                                                fill_channel_status("HF",eta,phi,depth,1,status); }
	  if(status) hf_data[eta+42][phi-1][depth-1][0].change_status(1); 
	  if(stat>0 && stat!=(ievt_*2)){ status=(double)stat/(double)(ievt_*2); 
	     if(status<0.995){ 
	        fill_channel_status("HF",eta,phi,depth,2,status); 
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
	        double tmp=rms[0]-rms_ref[0]; PedestalsRmsHFref->Fill(tmp); if(tmp<0) tmp=-tmp;
		if(tmp>deltaRms) deltaRms=tmp;
	     }
	     if(deltaPed>HFMeanTreshold){ fill_channel_status("HF",eta,phi,depth,3,deltaPed); Pedestals2DErrorHBHEHF->Fill(eta,phi,1);}
	     if(deltaRms>HFRmsTreshold){  fill_channel_status("HF",eta,phi,depth,4,deltaRms); Pedestals2DErrorHBHEHF->Fill(eta,phi,1);}
	  } 
      }
   }
}

void HcalDetDiagPedestalMonitor::fill_channel_status(char *subdet,int eta,int phi,int depth,int type,double status){
   int ind=-1;
   if(eta>42 || eta<-42 || eta==0) return;
   if(strcmp(subdet,"HB")==0 || strcmp(subdet,"HF")==0) if(depth==1) ind=0; else ind=1;
   else if(strcmp(subdet,"HE")==0) if(depth==3) ind=2; else ind=3+depth;
   else if(strcmp(subdet,"HO")==0) ind=3; 
   if(ind==-1) return;
   if(type==1) ChannelStatusMissingChannels[ind] ->setBinContent(eta+42,phi+1,status);
   if(type==2) ChannelStatusUnstableChannels[ind]->setBinContent(eta+42,phi+1,status);
   if(type==3) ChannelStatusBadPedestalMean[ind] ->setBinContent(eta+42,phi+1,status);
   if(type==4) ChannelStatusBadPedestalRMS[ind]  ->setBinContent(eta+42,phi+1,status);
}
void HcalDetDiagPedestalMonitor::done(){   
   if(ievt_>=500){
      CheckStatus();
      SaveReference();
   }   
} 
