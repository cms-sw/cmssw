#include "DQM/HcalMonitorTasks/interface/HcalDetDiagNoiseMonitor.h"

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
static std::string subdets[11]={"HBM","HBP","HEM","HEP","HO2M","HO1M","HO0","HO1P","HO2P","HFM","HFP"};
static std::string HB_RBX[36]={
"HBM01","HBM02","HBM03","HBM04","HBM05","HBM06","HBM07","HBM08","HBM09","HBM10","HBM11","HBM12","HBM13","HBM14","HBM15","HBM16","HBM17","HBM18",
"HBP01","HBP02","HBP03","HBP04","HBP05","HBP06","HBP07","HBP08","HBP09","HBP10","HBP11","HBP12","HBP13","HBP14","HBP15","HBP16","HBP17","HBP18"};
static std::string HE_RBX[36]={
"HEM01","HEM02","HEM03","HEM04","HEM05","HEM06","HEM07","HEM08","HEM09","HEM10","HEM11","HEM12","HEM13","HEM14","HEM15","HEM16","HEM17","HEM18",
"HEP01","HEP02","HEP03","HEP04","HEP05","HEP06","HEP07","HEP08","HEP09","HEP10","HEP11","HEP12","HEP13","HEP14","HEP15","HEP16","HEP17","HEP18"};
static std::string HO_RBX[36]={
"HO2M02","HO2M04","HO2M06","HO2M08","HO2M10","HO2M12","HO1M02","HO1M04","HO1M06","HO1M08","HO1M10","HO1M12",
"HO001","HO002","HO003","HO004","HO005","HO006","HO007","HO008","HO009","HO010","HO011","HO012",
"HO1P02","HO1P04","HO1P06","HO1P08","HO1P10","HO1P12","HO2P02","HO2P04","HO2P06","HO2P08","HO2P10","HO2P12",
};

HcalDetDiagNoiseMonitor::HcalDetDiagNoiseMonitor() {
  ievt_=0;
  run_number=-1;
  NoisyEvents=0;
}

HcalDetDiagNoiseMonitor::~HcalDetDiagNoiseMonitor(){}

void HcalDetDiagNoiseMonitor::clearME(){
  if(m_dbe){
    m_dbe->setCurrentFolder(baseFolder_);
    m_dbe->removeContents();
    m_dbe = 0;
  }
} 
void HcalDetDiagNoiseMonitor::reset(){}

void HcalDetDiagNoiseMonitor::setup(const edm::ParameterSet& ps, DQMStore* dbe){
  m_dbe=NULL;
  ievt_=0;
  if(dbe!=NULL) m_dbe=dbe;
  clearME();
 
  UseDB            = ps.getUntrackedParameter<bool>  ("UseDB"  , false);
  ReferenceData    = ps.getUntrackedParameter<string>("NoiseReferenceData" ,"");
  OutputFilePath   = ps.getUntrackedParameter<string>("OutputFilePath", "");
  HPDthresholdHi   = ps.getUntrackedParameter<double>("NoiseThresholdHPDhi",30.0);
  HPDthresholdLo   = ps.getUntrackedParameter<double>("NoiseThresholdHPDlo",12.0);
  SiPMthreshold    = ps.getUntrackedParameter<double>("NoiseThresholdSiPM",150.0);
  SpikeThreshold   = ps.getUntrackedParameter<double>("NoiseThresholdSpike",0.06);
  UpdateEvents     = ps.getUntrackedParameter<int>   ("NoiseUpdateEvents",200);
  
  FEDRawDataCollection_ = ps.getUntrackedParameter<edm::InputTag>("FEDRawDataCollection",edm::InputTag("source",""));
  inputLabelDigi_       = ps.getParameter<edm::InputTag>         ("digiLabel");
  
  HcalBaseMonitor::setup(ps,dbe);
  baseFolder_ = rootFolder_+"HcalNoiseMonitor";
  char *name;
  if(m_dbe!=NULL){    
     m_dbe->setCurrentFolder(baseFolder_);   
     meEVT_ = m_dbe->bookInt("HcalNoiseMonitor Event Number");
     m_dbe->setCurrentFolder(baseFolder_+"/Summary Plots");
     
     name="RBX Pixel multiplisity";   PixelMult        = m_dbe->book1D(name,name,73,0,73);
     name="HPD energy";               HPDEnergy        = m_dbe->book1D(name,name,200,0,2500);
     name="RBX energy";               RBXEnergy        = m_dbe->book1D(name,name,200,0,3500);
     name="HB RM Noise Fraction Map"; HB_RBXmapRatio   = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HB RM Spike Map";          HB_RBXmapSpikeCnt= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HB RM Spike Amplitude Map";HB_RBXmapSpikeAmp= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Noise Fraction Map"; HE_RBXmapRatio   = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Spike Map";          HE_RBXmapSpikeCnt= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Spike Amplitude Map";HE_RBXmapSpikeAmp= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Noise Fraction Map"; HO_RBXmapRatio   = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Spike Map";          HO_RBXmapSpikeCnt= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Spike Amplitude Map";HO_RBXmapSpikeAmp= m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
 
     m_dbe->setCurrentFolder(baseFolder_+"/Current Plots");
     name="HB RM Noise Fraction Map (current status)"; HB_RBXmapRatioCur = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HE RM Noise Fraction Map (current status)"; HE_RBXmapRatioCur = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     name="HO RM Noise Fraction Map (current status)"; HO_RBXmapRatioCur = m_dbe->book2D(name,name,4,0.5,4.5,36,0.5,36.5);
     
     std::string title="RM";
     HB_RBXmapRatio->setAxisTitle(title);
     HB_RBXmapRatioCur->setAxisTitle(title);
     HB_RBXmapSpikeAmp->setAxisTitle(title);
     HB_RBXmapSpikeCnt->setAxisTitle(title);
     HE_RBXmapRatio->setAxisTitle(title);
     HE_RBXmapRatioCur->setAxisTitle(title);
     HE_RBXmapSpikeAmp->setAxisTitle(title);
     HE_RBXmapSpikeCnt->setAxisTitle(title);
     HO_RBXmapRatio->setAxisTitle(title);
     HO_RBXmapRatioCur->setAxisTitle(title);
     HO_RBXmapSpikeAmp->setAxisTitle(title);
     HO_RBXmapSpikeCnt->setAxisTitle(title);
         
     for(int i=0;i<36;i++){
        HB_RBXmapRatio->setBinLabel(i+1,HB_RBX[i],2);
        HB_RBXmapRatioCur->setBinLabel(i+1,HB_RBX[i],2);
        HB_RBXmapSpikeAmp->setBinLabel(i+1,HB_RBX[i],2); 
        HB_RBXmapSpikeCnt->setBinLabel(i+1,HB_RBX[i],2);
        HE_RBXmapRatio->setBinLabel(i+1,HE_RBX[i],2);
        HE_RBXmapRatioCur->setBinLabel(i+1,HE_RBX[i],2);
        HE_RBXmapSpikeAmp->setBinLabel(i+1,HE_RBX[i],2);
        HE_RBXmapSpikeCnt->setBinLabel(i+1,HE_RBX[i],2);
        HO_RBXmapRatio->setBinLabel(i+1,HO_RBX[i],2);
        HO_RBXmapRatioCur->setBinLabel(i+1,HO_RBX[i],2);
        HO_RBXmapSpikeAmp->setBinLabel(i+1,HO_RBX[i],2);
        HO_RBXmapSpikeCnt->setBinLabel(i+1,HO_RBX[i],2);
     }
  } 
  ReferenceRun="UNKNOWN";
  IsReference=false;
  //LoadReference();
  lmap =new HcalLogicalMap(gen.createMap());
  return;
} 

void HcalDetDiagNoiseMonitor::processEvent(const edm::Event& iEvent, const edm::EventSetup& iSetup, const HcalDbService& cond){
bool isNoiseEvent=false;   
   if(!m_dbe) return;
   
   ievt_++;
   meEVT_->Fill(ievt_);
   run_number=iEvent.id().run();

   // We do not want to look at Abort Gap events
   edm::Handle<FEDRawDataCollection> rawdata;
   iEvent.getByLabel(FEDRawDataCollection_,rawdata);
   //checking FEDs for calibration information
   for(int i=FEDNumbering::MINHCALFEDID;i<=FEDNumbering::MAXHCALFEDID; i++) {
       const FEDRawData& fedData = rawdata->FEDData(i) ;
       if ( fedData.size() < 24 ) continue ;
       if(((const HcalDCCHeader*)(fedData.data()))->getCalibType()!=hc_Null) return;
   }
  
   HcalDetDiagNoiseRMData RMs[HcalFrontEndId::maxRmIndex];
   
   try{
         edm::Handle<HBHEDigiCollection> hbhe; 
         iEvent.getByLabel(inputLabelDigi_,hbhe);
         for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
	     double max=-100,sum,energy=0;
	     for(int i=0;i<digi->size()-1;i++){
	        sum=adc2fC[digi->sample(i).adc()&0xff]+adc2fC[digi->sample(i+1).adc()&0xff]; 
		if(max<sum) max=sum;
             }
	     if(max>HPDthresholdLo){
	        for(int i=0;i<digi->size();i++) energy+=adc2fC[digi->sample(i).adc()&0xff]-2.5;
	        HcalFrontEndId lmap_entry=lmap->getHcalFrontEndId(digi->id());
	        int index=lmap_entry.rmIndex(); if(index>=HcalFrontEndId::maxRmIndex) continue;
	        RMs[index].n_th_lo++;
	        if(max>HPDthresholdHi){ RMs[index].n_th_hi++; isNoiseEvent=true;}
		RMs[index].energy+=energy;
	     }
         }   
   }catch(...){}      
   try{
         edm::Handle<HODigiCollection> ho; 
         iEvent.getByLabel(inputLabelDigi_,ho);
         for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
 	     double max=-100,energy=0; int Eta=digi->id().ieta(); int Phi=digi->id().iphi();
	     for(int i=0;i<digi->size()-1;i++){
		if(max<adc2fC[digi->sample(i).adc()&0xff]) max=adc2fC[digi->sample(i).adc()&0xff];
             }
	     if((Eta>=11 && Eta<=15 && Phi>=59 && Phi<=70) || (Eta>=5 && Eta<=10 && Phi>=47 && Phi<=58)){
  	        if(max>SiPMthreshold){
	          for(int i=0;i<digi->size();i++) energy+=adc2fC[digi->sample(i).adc()&0xff]-11.0;
	          HcalFrontEndId lmap_entry=lmap->getHcalFrontEndId(digi->id());
	          int index=lmap_entry.rmIndex(); if(index>=HcalFrontEndId::maxRmIndex) continue;
	          RMs[index].n_th_hi++; isNoiseEvent=true;
	          RMs[index].energy+=energy;
	        }	          
	     }else{
	        if(max>HPDthresholdLo){
	          for(int i=0;i<digi->size();i++) energy+=adc2fC[digi->sample(i).adc()&0xff]-2.5;
	          HcalFrontEndId lmap_entry=lmap->getHcalFrontEndId(digi->id());
	          int index=lmap_entry.rmIndex(); if(index>=HcalFrontEndId::maxRmIndex) continue;
	          RMs[index].n_th_lo++;
	          if(max>HPDthresholdHi){ RMs[index].n_th_hi++; isNoiseEvent=true;}
		  RMs[index].energy+=energy;
	        }
	     }		          
         }   
   }catch(...){}  
//    try{ //curently we don't want to look at PMTs
//          edm::Handle<HFDigiCollection> hf;
//          iEvent.getByType(hf);
//          for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
//             
// 	     for(int i=0;i<digi->size();i++); 
// 	     
//          }   
//    }catch(...){}    
   if(isNoiseEvent){
      NoisyEvents++;
      
      // RMs loop
      for(int i=0;i<HcalFrontEndId::maxRmIndex;i++){
        if(RMs[i].n_th_hi>0){
	   RBXCurrentSummary.AddNoiseStat(i);
	   RBXSummary.AddNoiseStat(i);
	   HPDEnergy->Fill(RMs[i].energy);
	}
      }
    }  
    // RBX loop
    for(int sd=0;sd<9;sd++) for(int sect=1;sect<=18;sect++){
       std::stringstream tempss;
       tempss << std::setw(2) << std::setfill('0') << sect;
       std::string rbx= subdets[sd]+tempss.str();
	 
       double rbx_energy=0;int pix_mult=0; bool isValidRBX=false;
       for(int rm=1;rm<=4;rm++){
         int index=RBXSummary.GetRMindex(rbx,rm);
	 if(index>0 && index<HcalFrontEndId::maxRmIndex){
	    rbx_energy+=RMs[index].energy;
            pix_mult+=RMs[index].n_th_lo; 
	    isValidRBX=true;
         }
       }
       if(isValidRBX){
         PixelMult->Fill(pix_mult);
         RBXEnergy->Fill(rbx_energy);
       }
   }
   
   UpdateHistos();
       
   if((ievt_%100)==0) printf("%i\t%i\n",ievt_,NoisyEvents);
   return;
}

void HcalDetDiagNoiseMonitor::UpdateHistos(){
int first_rbx=0,last_rbx=0;  
  for(int sd=0;sd<9;sd++){
     if(RBXCurrentSummary.GetStat(sd)>=UpdateEvents){
        if(sd==0){ first_rbx=0;  last_rbx=18;} //HBM
        if(sd==1){ first_rbx=18; last_rbx=36;} //HBP
        if(sd==0 || sd==1){  // update HB plots
           for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
              double val1=0,val2=0;
              if(RBXSummary.GetRMStatusValue(HB_RBX[rbx],rm,&val1)){
	        HB_RBXmapRatio->setBinContent(rm,rbx+1,val1);
                if(RBXCurrentSummary.GetRMStatusValue(HB_RBX[rbx],rm,&val2)){
	           HB_RBXmapRatioCur->setBinContent(rm,rbx+1,val2);
		   if((val2-val1)>SpikeThreshold){
		      double n=HB_RBXmapSpikeCnt->getBinContent(rm,rbx+1);
		      double a=HB_RBXmapSpikeAmp->getBinContent(rm,rbx+1);
		      HB_RBXmapSpikeCnt->Fill(rm,rbx+1,1);
		      HB_RBXmapSpikeAmp->setBinContent(rm,rbx+1,((val2-val1)+a*n)/(n+1));
	           }
		}
	      }
           }	
	}
	if(sd==2){ first_rbx=0;  last_rbx=18;} //HEM
        if(sd==3){ first_rbx=18; last_rbx=36;} //HEP
        if(sd==2 || sd==3){  // update HB plots
           for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
              double val1=0,val2=0;
              if(RBXSummary.GetRMStatusValue(HE_RBX[rbx],rm,&val1)){
	        HE_RBXmapRatio->setBinContent(rm,rbx+1,val1);
                if(RBXCurrentSummary.GetRMStatusValue(HE_RBX[rbx],rm,&val2)){
		   HE_RBXmapRatioCur->setBinContent(rm,rbx+1,val2);
	           if((val2-val1)>SpikeThreshold){
		      double n=HE_RBXmapSpikeCnt->getBinContent(rm,rbx+1);
		      double a=HE_RBXmapSpikeAmp->getBinContent(rm,rbx+1);
		      HE_RBXmapSpikeCnt->Fill(rm,rbx+1,1);
		      HE_RBXmapSpikeAmp->setBinContent(rm,rbx+1,((val2-val1)+a*n)/(n+1));
	           }
	        }
	      }
           }	
	}
        if(sd==4){ first_rbx=0;  last_rbx=6;}   //HO2M
	if(sd==5){ first_rbx=6;  last_rbx=12;}  //HO1M
	if(sd==6){ first_rbx=12;  last_rbx=24;} //HO0
	if(sd==7){ first_rbx=24;  last_rbx=30;} //HO1P
	if(sd==8){ first_rbx=30;  last_rbx=36;} //HO2P
	if(sd>3){ // update HO plots
           for(int rbx=first_rbx;rbx<last_rbx;rbx++)for(int rm=1;rm<=4;rm++){
              double val1=0,val2=0;
              if(RBXSummary.GetRMStatusValue(HO_RBX[rbx],rm,&val1)){
	        HO_RBXmapRatio->setBinContent(rm,rbx+1,val1);
                if(RBXCurrentSummary.GetRMStatusValue(HO_RBX[rbx],rm,&val2)){
		   HO_RBXmapRatioCur->setBinContent(rm,rbx+1,val2);
	           if((val2-val1)>SpikeThreshold){
		      double n=HO_RBXmapSpikeCnt->getBinContent(rm,rbx+1);
		      double a=HO_RBXmapSpikeAmp->getBinContent(rm,rbx+1);
		      HO_RBXmapSpikeCnt->Fill(rm,rbx+1,1);
		      HO_RBXmapSpikeAmp->setBinContent(rm,rbx+1,((val2-val1)+a*n)/(n+1));
	           }
	        }
	      }
           }		
	}
	
        RBXCurrentSummary.reset(sd); 
        printf("update %i\n",sd); 
     }
  } 
} 

void HcalDetDiagNoiseMonitor::SaveReference(){
char   RBX[20];
int    RM_INDEX,RM;
double VAL;
    if(UseDB==false){
       char str[100]; 
       sprintf(str,"%sHcalDetDiagNoiseData_run%06i.root",OutputFilePath.c_str(),run_number);
       TFile *theFile = new TFile(str, "RECREATE");
       if(!theFile->IsOpen()) return;
       theFile->cd();
       sprintf(str,"%d",run_number); TObjString run(str);    run.Write("run number");
       sprintf(str,"%d",ievt_);      TObjString events(str); events.Write("Total events processed");
       
       TTree *tree   =new TTree("HCAL Noise data","HCAL Noise data");
       if(tree==0)   return;
       tree->Branch("RBX",            &RBX,      "RBX/C");
       tree->Branch("rm",             &RM,       "rm/I");
       tree->Branch("rm_index",       &RM_INDEX, "rm_index/I");
       tree->Branch("relative_noise", &VAL,      "relative_noise/D");
       for(int sd=0;sd<9;sd++) for(int sect=1;sect<=18;sect++) for(int rm=1;rm<=4;rm++){
           std::stringstream tempss;
           tempss << std::setw(2) << std::setfill('0') << sect;
           std::string rbx= subdets[sd]+tempss.str();
           double val;
           if(RBXCurrentSummary.GetRMStatusValue(rbx,rm,&val)){
	       sprintf(RBX,"%s",(char *)rbx.c_str());
	       RM=rm;
	       RM_INDEX=RBXCurrentSummary.GetRMindex(rbx,rm);
	       val=VAL;
               tree->Fill();
           }
       }     
       theFile->Write();
       theFile->Close();
   }
}

void HcalDetDiagNoiseMonitor::LoadReference(){
TFile *f;
int    RM_INDEX;
double VAL;
   if(UseDB==false){
      try{ 
         f = new TFile(ReferenceData.c_str(),"READ");
      }catch(...){ return ;}
      if(!f->IsOpen()){ return ;}
      TObjString *STR=(TObjString *)f->Get("run number");
      
      if(STR){ string Ref(STR->String()); ReferenceRun=Ref;}
      
      TTree*  t=(TTree*)f->Get("HCAL Noise data");
      if(!t) return;
      t->SetBranchAddress("rm_index",       &RM_INDEX);
      t->SetBranchAddress("relative_noise", &VAL);
      for(int ievt=0;ievt<t->GetEntries();ievt++){
         t->GetEntry(ievt);
	 RBXCurrentSummary.SetReference(RM_INDEX,VAL);
	 RBXSummary.SetReference(RM_INDEX,VAL);
      }
      f->Close();
      IsReference=true;
   }
} 

void HcalDetDiagNoiseMonitor::done(){   /*SaveReference();*/ } 
