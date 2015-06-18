#include "DQM/HcalMonitorTasks/interface/HcalDetDiagTimingMonitor.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HcalDigi/interface/HcalCalibrationEventTypes.h"
#include "EventFilter/HcalRawToDigi/interface/HcalDCCHeader.h"

#include <math.h>

// this is to retrieve GT digi's 
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"



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
static const int MAXRPC =20;
static const int MAXDTBX=20;
static const int MAXCSC =20;    
static const int TRIG_DT =1;
static const int TRIG_RPC=2;
static const int TRIG_GCT=4;
static const int TRIG_CSC=8;
static const int TRIG_RPCF=16;

HcalDetDiagTimingMonitor::HcalDetDiagTimingMonitor(const edm::ParameterSet& ps) :HcalBaseDQMonitor(ps)
{
  Online_                = ps.getUntrackedParameter<bool>("online",false);
  mergeRuns_             = ps.getUntrackedParameter<bool>("mergeRuns",false);
  enableCleanup_         = ps.getUntrackedParameter<bool>("enableCleanup",false);
  debug_                 = ps.getUntrackedParameter<int>("debug",0);
  prefixME_              = ps.getUntrackedParameter<std::string>("subSystemFolder","Hcal/");
  if (prefixME_.substr(prefixME_.size()-1,prefixME_.size())!="/")
    prefixME_.append("/");
  subdir_                = ps.getUntrackedParameter<std::string>("TaskFolder","DetDiagTimingMonitor_Hcal");
  if (subdir_.size()>0 && subdir_.substr(subdir_.size()-1,subdir_.size())!="/")
    subdir_.append("/");
  subdir_=prefixME_+subdir_;
  AllowedCalibTypes_     = ps.getUntrackedParameter<std::vector<int> > ("AllowedCalibTypes");
  skipOutOfOrderLS_      = ps.getUntrackedParameter<bool>("skipOutOfOrderLS",false);
  NLumiBlocks_           = ps.getUntrackedParameter<int>("NLumiBlocks",4000);
  makeDiagnostics_       = ps.getUntrackedParameter<bool>("makeDiagnostics",false);

  GCTTriggerBit1_= ps.getUntrackedParameter<int>("GCTTriggerBit1", 15);         
  GCTTriggerBit2_= ps.getUntrackedParameter<int>("GCTTriggerBit2", 16);         
  GCTTriggerBit3_= ps.getUntrackedParameter<int>("GCTTriggerBit3", 17);         
  GCTTriggerBit4_= ps.getUntrackedParameter<int>("GCTTriggerBit4", 18);         
  GCTTriggerBit5_= ps.getUntrackedParameter<int>("GCTTriggerBit5", 16);         
  CosmicsCorr_   = ps.getUntrackedParameter<bool>("CosmicsCorr", true); 
  
  L1ADataLabel_  = ps.getUntrackedParameter<edm::InputTag>("gtLabel");
  inputLabelDigi_= ps.getUntrackedParameter<edm::InputTag>("digiLabel");

  // register for data access
  tok_raw_ = consumes<FEDRawDataCollection>(ps.getUntrackedParameter<edm::InputTag>("FEDRawDataCollection",edm::InputTag("source","")));
  tok_l1_ = consumes<L1GlobalTriggerReadoutRecord>(L1ADataLabel_);
  tok_l1mu_ = consumes<L1MuGMTReadoutCollection>(L1ADataLabel_);
  tok_hbhe_ = consumes<HBHEDigiCollection>(inputLabelDigi_);
  tok_ho_ = consumes<HODigiCollection>(inputLabelDigi_);
  tok_hf_ = consumes<HFDigiCollection>(inputLabelDigi_);
}

HcalDetDiagTimingMonitor::~HcalDetDiagTimingMonitor(){}

void HcalDetDiagTimingMonitor::reset(){}

void HcalDetDiagTimingMonitor::bookHistograms(DQMStore::IBooker &ib, const edm::Run& run, const edm::EventSetup& c)
{
  if (debug_>1) std::cout <<"HcalDetDiagTimingMonitor::bookHistograms"<<std::endl;
  HcalBaseDQMonitor::bookHistograms(ib,run,c);

  if (tevt_==0) this->setup(ib); // set up histograms if they have not been created before
  if (mergeRuns_==false)
    this->reset();

  return;

} // void HcalNDetDiagTimingMonitor::bookHistograms(...)



void HcalDetDiagTimingMonitor::setup(DQMStore::IBooker &ib)
{
  
  HcalBaseDQMonitor::setup(ib);

  std::string str;
     ib.setCurrentFolder(subdir_);   
     str="Hcal Timing summary"; Summary = ib.book2D(str,str,6,0,6,6,0,6); 
     Summary->setBinLabel(1,"DT",1);
     Summary->setBinLabel(2,"RPC",1);
     Summary->setBinLabel(3,"GCT",1);
     Summary->setBinLabel(4,"CSC",1);
     Summary->setBinLabel(5,"RPCf",1);
     Summary->setBinLabel(6,"bit11",1);
     Summary->setBinLabel(1,"HB",2);
     Summary->setBinLabel(2,"HO",2);
     Summary->setBinLabel(3,"HEM",2);
     Summary->setBinLabel(4,"HEP",2);
     Summary->setBinLabel(5,"HFM",2);
     Summary->setBinLabel(6,"HFP",2);
     for(int i=1;i<=6;i++) for(int j=1;j<=6;j++) Summary->setBinContent(i,j,-1);
     
     ib.setCurrentFolder(subdir_+"Timing Plots");
     str="HB Timing (DT Trigger)";                      HBTimeDT  = ib.book1D(str,str,100,0,10); 
     str="HO Timing (DT Trigger)";                      HOTimeDT  = ib.book1D(str,str,100,0,10); 
     str="HB Timing (RPC Trigger)";                     HBTimeRPC = ib.book1D(str,str,100,0,10); 
     str="HO Timing (RPC Trigger)";                     HOTimeRPC = ib.book1D(str,str,100,0,10); 
     str="HB Timing (HO SelfTrigger tech bit 11)";      HBTimeHO  = ib.book1D(str,str,100,0,10); 
     str="HO Timing (HO SelfTrigger tech bit 11)";      HOTimeHO  = ib.book1D(str,str,100,0,10); 
     
     str="HB Timing (GCT Trigger alg bit 15 16 17 18)"; HBTimeGCT  =ib.book1D(str,str,100,0,10); 
     str="HO Timing (GCT Trigger alg bit 15 16 17 18)"; HOTimeGCT  =ib.book1D(str,str,100,0,10); 
     
     str="HEP Timing (CSC Trigger)";                    HETimeCSCp =ib.book1D(str,str,100,0,10); 
     str="HEM Timing (CSC Trigger)";                    HETimeCSCm =ib.book1D(str,str,100,0,10);
     str="HEP Timing (RPCf Trigger)";                   HETimeRPCp =ib.book1D(str,str,100,0,10); 
     str="HEM Timing (RPCf Trigger)";                   HETimeRPCm =ib.book1D(str,str,100,0,10);
     str="HFP Timing (CSC Trigger)";                    HFTimeCSCp =ib.book1D(str,str,100,0,10); 
     str="HFM Timing (CSC Trigger)";                    HFTimeCSCm =ib.book1D(str,str,100,0,10);     
     str="HBHE Shape";                                  HBHEShape  =ib.book1D(str,str,10,-0.5,9.5);
     str="HO Shape";                                    HOShape    =ib.book1D(str,str,10,-0.5,9.5);

} 

void HcalDetDiagTimingMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  HcalBaseDQMonitor::analyze(iEvent, iSetup);
  if (!IsAllowedCalibType()) return;
  if (LumiInOrder(iEvent.luminosityBlock())==false) return;
//  HcalBaseDQMonitor::analyze(iEvent, iSetup);
  
  int eta,phi,depth,nTS,BXinEVENT=1,TRIGGER=0;
  
  // We do not want to look at Abort Gap events
  edm::Handle<FEDRawDataCollection> rawdata;
  iEvent.getByToken(tok_raw_,rawdata);
  //checking FEDs for calibration information
  if(!rawdata.isValid()) return;
  for(int i=FEDNumbering::MINHCALFEDID;
		  i<=FEDNumbering::MAXHCALuTCAFEDID; i++) 
  {
	if (i>FEDNumbering::MAXHCALFEDID && i<FEDNumbering::MINHCALuTCAFEDID)
		continue;

    const FEDRawData& fedData = rawdata->FEDData(i) ;
    if ( fedData.size() < 24 ) continue ;
    if(((const HcalDCCHeader*)(fedData.data()))->getCalibType()!=hc_Null) return;
  }
  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////
  bool GCTTrigger1=false,GCTTrigger2=false,GCTTrigger3=false,GCTTrigger4=false,GCTTrigger5=false,HOselfTrigger=false;
  // Check GCT trigger bits
  edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
  iEvent.getByToken(tok_l1_, gtRecord);
  if(gtRecord.isValid()){

    const TechnicalTriggerWord tWord = gtRecord->technicalTriggerWord();
    const DecisionWord         dWord = gtRecord->decisionWord();
    //bool HFselfTrigger   = tWord.at(9);
    if (!tWord.empty()) HOselfTrigger    = tWord.at(11);

    if (!dWord.empty()) 
      {
	GCTTrigger1      = dWord.at(GCTTriggerBit1_);     
	GCTTrigger2      = dWord.at(GCTTriggerBit2_);     
	GCTTrigger3      = dWord.at(GCTTriggerBit3_);     
	GCTTrigger4      = dWord.at(GCTTriggerBit4_);     
	GCTTrigger5      = dWord.at(GCTTriggerBit5_);     
      }

    // define trigger trigger source (example from GMT group)
    edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle; 
    iEvent.getByToken(tok_l1mu_,gmtrc_handle);
    if(!gmtrc_handle.isValid()) return;
    L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();
   
    int idt   =0;
    int icsc  =0;
    int irpcb =0;
    int irpcf =0;
    int ndt[5]   = {0,0,0,0,0};
    int ncsc[5]  = {0,0,0,0,0};
    int nrpcb[5] = {0,0,0,0,0};
    int nrpcf[5] = {0,0,0,0,0};
    int N;	
    std::vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
    std::vector<L1MuGMTReadoutRecord>::const_iterator igmtrr;
    N=0;
    int NN=0;
    for(igmtrr=gmt_records.begin(); igmtrr!=gmt_records.end(); igmtrr++) {
      if(igmtrr->getBxInEvent()==0) BXinEVENT=NN;
      NN++;
      std::vector<L1MuRegionalCand>::const_iterator iter1;
      std::vector<L1MuRegionalCand> rmc;
      // DTBX Trigger
      rmc = igmtrr->getDTBXCands(); 
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
	if ( idt < MAXDTBX && !(*iter1).empty() ) {
	  idt++; 
	  if(N<5) ndt[N]++; 
				 
	} 	 
      }
      // CSC Trigger
      rmc = igmtrr->getCSCCands(); 
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
	if ( icsc < MAXCSC && !(*iter1).empty() ) {
	  icsc++; 
	  if(N<5) ncsc[N]++; 
	} 
      }
      // RPCb Trigger
      rmc = igmtrr->getBrlRPCCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
	if ( irpcb < MAXRPC && !(*iter1).empty() ) {
	  irpcb++;
	  if(N<5) nrpcb[N]++;
				
	}  
      }
      // RPCfwd Trigger
      rmc = igmtrr->getFwdRPCCands();
      for(iter1=rmc.begin(); iter1!=rmc.end(); iter1++) {
	if ( irpcf < MAXRPC && !(*iter1).empty() ) {
	  irpcf++;
	  if(N<5) nrpcf[N]++;
				
	}  
      }
		
      N++;
    }
    if(ncsc[BXinEVENT]>0 ) {   TRIGGER=+TRIG_CSC;  }
    if(ndt[BXinEVENT]>0  ) {   TRIGGER=+TRIG_DT;   }
    if(nrpcb[BXinEVENT]>0) {   TRIGGER=+TRIG_RPC;  }
    if(nrpcf[BXinEVENT]>0) {   TRIGGER=+TRIG_RPCF; }
    if(GCTTrigger1 || GCTTrigger2 || GCTTrigger3 || GCTTrigger4 || GCTTrigger5) {  TRIGGER=+TRIG_GCT; }
  }
  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////   
  if(ievt_<100){
    edm::Handle<HBHEDigiCollection> hbhe; 
    iEvent.getByToken(tok_hbhe_,hbhe);
    if(hbhe.isValid()){   
      for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
	eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	for(int i=0;i<nTS;i++) if(digi->sample(i).adc()<20) set_hbhe(eta,phi,depth,digi->sample(i).capid(),adc2fC[digi->sample(i).adc()]);
      }   
    }    
    edm::Handle<HODigiCollection> ho; 
    iEvent.getByToken(tok_ho_,ho);
    if(ho.isValid()){
      for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
	eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	for(int i=0;i<nTS;i++) if(digi->sample(i).adc()<20) set_ho(eta,phi,depth,digi->sample(i).capid(),adc2fC[digi->sample(i).adc()]);
      }   
    }
    edm::Handle<HFDigiCollection> hf;
    iEvent.getByToken(tok_hf_,hf);
    if(hf.isValid()){
      for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
	eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	for(int i=0;i<nTS;i++) if(digi->sample(i).adc()<20) set_hf(eta,phi,depth,digi->sample(i).capid(),adc2fC[digi->sample(i).adc()]);
      }   
    }
    return;   
  }
  /////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////   
  double data[20];
  edm::Handle<HBHEDigiCollection> hbhe; 
  iEvent.getByToken(tok_hbhe_,hbhe);
  if(hbhe.isValid()){ 
    for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
      eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
      for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()]-get_ped_hbhe(eta,phi,depth,digi->sample(i).capid());

      double energy=0;
      for(int i=0;i<nTS;i++) energy+=data[i]; 
      if(energy>20) for(int i=0;i<nTS;i++) HBHEShape->Fill(i,data[i]);

      if(!isSignal(data,nTS)) continue;
	  
      occHBHE[eta+50][phi][depth]+=1.0; occSum+=1.0;
      if((occHBHE[eta+50][phi][depth]/(double)(ievt_))>0.001) continue;
	     
      double Time=GetTime(data,nTS);
      if(digi->id().subdet()==HcalBarrel){
	if(TRIGGER==TRIG_GCT) HBTimeGCT->Fill(Time);
	if(CosmicsCorr_) Time+=(7.5*sin((phi*5.0)/180.0*3.14159))/25.0;
	if(TRIGGER==TRIG_DT)  HBTimeDT->Fill(Time);
	if(HOselfTrigger)     HBTimeHO->Fill(Time);
	if(TRIGGER==TRIG_RPC) HBTimeRPC->Fill(Time);
      }
      if(digi->id().subdet()==HcalEndcap){
	if(CosmicsCorr_) Time+=(3.5*sin((phi*5.0)/180.0*3.14159))/25.0; 
	if(TRIGGER==TRIG_CSC && eta>0)  HETimeCSCp->Fill(Time);
	if(TRIGGER==TRIG_CSC && eta<0)  HETimeCSCm->Fill(Time);  
	if(TRIGGER==TRIG_RPCF && eta>0) HETimeRPCp->Fill(Time);
	if(TRIGGER==TRIG_RPCF && eta<0) HETimeRPCm->Fill(Time);  	     
      }
	 
    }    
  }
  edm::Handle<HODigiCollection> ho; 
  iEvent.getByToken(tok_ho_,ho);
  if(ho.isValid()){ 
    for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
      eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
      for(int i=0;i<nTS;i++) data[i]=adc2fC[digi->sample(i).adc()]-get_ped_ho(eta,phi,depth,digi->sample(i).capid());

      double energy=0;
      for(int i=0;i<nTS;i++) energy+=data[i]; 
      if(energy>100) for(int i=0;i<nTS;i++) HOShape->Fill(i,data[i]);
      if(energy<100) continue;

      occHO[eta+50][phi][depth]+=1.0;
      occSum+=1.0;
      if((occHO[eta+50][phi][depth]/(double)(ievt_))>0.001) continue;
	     
      double Time=GetTime(data,nTS);
      if(CosmicsCorr_) Time+=(12.0*sin((phi*5.0)/180.0*3.14159))/25.0;  
      if(TRIGGER==TRIG_DT)  HOTimeDT ->Fill(Time);
      if(HOselfTrigger)     HOTimeHO ->Fill(Time);
      if(TRIGGER==TRIG_RPC) HOTimeRPC->Fill(Time);
      if(TRIGGER==TRIG_GCT) HOTimeGCT->Fill(Time);
    }   
  }
  edm::Handle<HFDigiCollection> hf; 
  iEvent.getByToken(tok_hf_,hf);
  if(hf.isValid()){ 
    for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
      eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
      double energy=0;
      for(int i=0;i<nTS;i++){
	data[i]=adc2fC[digi->sample(i).adc()]-get_ped_hf(eta,phi,depth,digi->sample(i).capid());
	energy+=data[i]; 
      }
	    
      if(energy<25.0) continue;
      occHF[eta+50][phi][depth]+=1.0;
      occSum+=1.0;
	    
      double Time=GetTime(data,nTS);
      if((occHF[eta+50][phi][depth]/(double)(ievt_))>0.01) continue;
	    
      if(TRIGGER==TRIG_CSC && eta>0) HFTimeCSCp->Fill(Time); 
      if(TRIGGER==TRIG_CSC && eta<0) HFTimeCSCm->Fill(Time); 
    }   
  }
  if((ievt_%500)==0){
    CheckTiming();
    //printf("Run: %i, Evants processed: %i\n",iEvent.run(),ievt_);       
  }
}

void HcalDetDiagTimingMonitor::CheckTiming(){
  if(HBTimeDT->getEntries()>10){
    Summary->setBinContent(1,1,1); 
  }
  if(HBTimeRPC->getEntries()>10){
    Summary->setBinContent(2,1,1); 
  } 
  if(HBTimeGCT->getEntries()>10){
    Summary->setBinContent(3,1,1); 
  } 
  if(HBTimeHO->getEntries()>10){
    Summary->setBinContent(6,1,1); 
  } 
  if(HOTimeDT->getEntries()>10){
    Summary->setBinContent(1,2,1); 
  }
  if(HOTimeRPC->getEntries()>10){
    Summary->setBinContent(2,2,1); 
  } 
  if(HOTimeGCT->getEntries()>10){
    Summary->setBinContent(3,2,1); 
  } 
  if(HOTimeHO->getEntries()>10){
    Summary->setBinContent(6,2,1); 
  } 
  if(HETimeCSCp->getEntries()>10){
    Summary->setBinContent(4,4,1); 
  } 
  if(HETimeCSCm->getEntries()>10){
    Summary->setBinContent(4,3,1); 
  } 
  if(HETimeRPCp->getEntries()>10){
    Summary->setBinContent(5,4,1); 
  } 
  if(HETimeRPCm->getEntries()>10){
    Summary->setBinContent(5,3,1); 
  } 
  if(HFTimeCSCp->getEntries()>10){
    Summary->setBinContent(4,6,1); 
  } 
  if(HFTimeCSCm->getEntries()>10){
    Summary->setBinContent(4,4,1); 
  } 
}

void HcalDetDiagTimingMonitor::done(){   } 
DEFINE_FWK_MODULE (HcalDetDiagTimingMonitor);
