// -*- C++ -*-
//
// Package:    HcalTimingMonitorModule
// Class:      HcalTimingMonitorModule
// 
/**\class HcalTimingMonitorModule HcalTimingMonitorModule.cc DQM/HcalMonitorModule/src/HcalTimingMonitorModule.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmitry Vishnevskiy
//         Created:  Thu Mar 27 08:12:02 CET 2008
// $Id: HcalTimingMonitorModule.cc,v 1.11 2012/10/09 18:16:14 wdd Exp $
//
//

static const int MAXGEN =10;
static const int MAXRPC =20;
static const int MAXDTBX=20;
static const int MAXCSC =20;    
static const int MAXGMT =20;
static const int TRIG_DT =1;
static const int TRIG_RPC=2;
static const int TRIG_GCT=4;
static const int TRIG_CSC=8;

// system include files
#include <iostream>
#include <fstream>
#include <cmath>
#include <iosfwd>
#include <bitset>
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// this is to retrieve HCAL digi's
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"

// this is to retrieve GT digi's 
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtPsbWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"

// DQM stuff
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
 
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


class HcalTimingMonitorModule : public edm::EDAnalyzer {
   public:
      explicit HcalTimingMonitorModule(const edm::ParameterSet&);
      ~HcalTimingMonitorModule();
      void   initialize();
   
   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      double GetTime(double *data,int n){
             int MaxI=-100; double Time=0,SumT=0,MaxT=-10;
             for(int j=0;j<n;++j) if(MaxT<data[j]){ MaxT=data[j]; MaxI=j; }
	     if (MaxI>=0)
	       {
		 Time=MaxI*data[MaxI];
		 SumT=data[MaxI];
		 if(MaxI>0){ Time+=(MaxI-1)*data[MaxI-1]; SumT+=data[MaxI-1]; }
		 if(MaxI<(n-1)){ Time+=(MaxI+1)*data[MaxI+1]; SumT+=data[MaxI+1]; }
		 Time=Time/SumT;
	       }
             return Time;
      }
      bool isSignal(double *data,int n){
        int Imax=-1; double max=-100;
        for(int i=0;i<n;i++) if(data[i]>max){max=data[i]; Imax=i;}
        if(Imax==0 && Imax==(n-1)) return false;
        float sum=data[Imax-1]+data[Imax+1];
        if(data[Imax]>5.5 && sum>(data[Imax]*0.20)) return true;
        return false;
     }
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
     // to be undependent from th DB we want to calculate pedestals using first 100 events
     void set_hbhe(int eta,int phi,int depth,int cap,float val){
       HBHE[eta+50][phi][depth][cap]+=val;
       nHBHE[eta+50][phi][depth][cap]+=1.0;
     }   
     void set_ho(int eta,int phi,int depth,int cap,float val){
       HO[eta+50][phi][depth][cap]+=val;
       nHO[eta+50][phi][depth][cap]+=1.0;
     }   
     void set_hf(int eta,int phi,int depth,int cap,float val){
       HF[eta+50][phi][depth][cap]+=val;
       nHF[eta+50][phi][depth][cap]+=1.0;
     }
     double get_ped_hbhe(int eta,int phi,int depth,int cup){
      if(nHBHE[eta+50][phi][depth][cup]<10) return 2.5; 
      if(nHBHE[eta+50][phi][depth][cup]!=0){
         double ped=HBHE[eta+50][phi][depth][cup]/nHBHE[eta+50][phi][depth][cup];
         if(ped>1.5 && ped<4.5) return ped;
      } 
      return 99999; 
     }   
     double get_ped_ho(int eta,int phi,int depth,int cup){
      if(nHO[eta+50][phi][depth][cup]<10) return 2.5; 
      if(nHO[eta+50][phi][depth][cup]!=0){
         double ped=HO[eta+50][phi][depth][cup]/nHO[eta+50][phi][depth][cup];
         if(ped>1.5 && ped<4.5) return ped;
      }
      return 99999; 
     }   
     double get_ped_hf(int eta,int phi,int depth,int cup){
      if(nHF[eta+50][phi][depth][cup]<10) return 2.5; 
      if(nHF[eta+50][phi][depth][cup]!=0){
         double ped=HF[eta+50][phi][depth][cup]/nHF[eta+50][phi][depth][cup];
         if(ped>1.5 && ped<4.5) return ped;
      }
      return 99999; 
     }   
     double HBHE[100][73][5][4];
     double nHBHE[100][73][5][4];
     double HO[100][73][5][4];
     double nHO[100][73][5][4];   
     double HF[100][73][5][4];
     double nHF[100][73][5][4];
///////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////
    int counterEvt_;
    int run_number; 
   
    int TrigCSC,TrigDT,TrigRPC,TrigGCT;
    
    edm::ParameterSet parameters_;
    DQMStore          *dbe_;
    std::string       monitorName_;
    int               prescaleLS_,prescaleEvt_;
    int               GCTTriggerBit1_;
    int               GCTTriggerBit2_;
    int               GCTTriggerBit3_;
    int               GCTTriggerBit4_;
    int               GCTTriggerBit5_;
    bool              CosmicsCorr_;
    bool              Debug_;
    
    MonitorElement *HBEnergy,*HEEnergy,*HOEnergy,*HFEnergy;
    MonitorElement *DTcand,*RPCbcand,*RPCfcand,*CSCcand,*OR; 
  
    MonitorElement *HBShapeDT; 
    MonitorElement *HBShapeRPC; 
    MonitorElement *HBShapeGCT; 
    MonitorElement *HOShapeDT; 
    MonitorElement *HOShapeRPC; 
    MonitorElement *HOShapeGCT; 
    MonitorElement *HEShapeCSCp; 
    MonitorElement *HEShapeCSCm; 
    MonitorElement *HFShapeCSCp; 
    MonitorElement *HFShapeCSCm; 
    
    MonitorElement *HBTimeDT; 
    MonitorElement *HBTimeRPC; 
    MonitorElement *HBTimeGCT; 
    MonitorElement *HOTimeDT; 
    MonitorElement *HOTimeRPC; 
    MonitorElement *HOTimeGCT; 
    MonitorElement *HETimeCSCp; 
    MonitorElement *HETimeCSCm;
    MonitorElement *HFTimeCSCp; 
    MonitorElement *HFTimeCSCm;
    
    std::string L1ADataLabel;

    edm::InputTag hbheDigiCollectionTag_;
    edm::InputTag hoDigiCollectionTag_;
    edm::InputTag hfDigiCollectionTag_;
};

HcalTimingMonitorModule::HcalTimingMonitorModule(const edm::ParameterSet& iConfig) :
   hbheDigiCollectionTag_(iConfig.getParameter<edm::InputTag>("hbheDigiCollectionTag")),
   hoDigiCollectionTag_(iConfig.getParameter<edm::InputTag>("hoDigiCollectionTag")),
   hfDigiCollectionTag_(iConfig.getParameter<edm::InputTag>("hfDigiCollectionTag")) {

  std::string str;   
   parameters_ = iConfig;
   dbe_ = edm::Service<DQMStore>().operator->();
   // Base folder for the contents of this job
   std::string subsystemname = parameters_.getUntrackedParameter<std::string>("subSystemFolder", "HcalTiming") ;
   
   monitorName_ = parameters_.getUntrackedParameter<std::string>("monitorName","HcalTiming");
   if (monitorName_ != "" ) monitorName_ =subsystemname+"/"+monitorName_+"/" ;
   counterEvt_=0;
   
   // some currently dummy things for compartability with GUI
   dbe_->setCurrentFolder(subsystemname+"/EventInfo/");
   str="reportSummary";
   dbe_->bookFloat(str)->Fill(1);     // Unknown status by default
   str="reportSummaryMap";
   MonitorElement* me=dbe_->book2D(str,str,5,0,5,1,0,1); // Unknown status by default
   TH2F* myhist=me->getTH2F();
   myhist->GetXaxis()->SetBinLabel(1,"HB");
   myhist->GetXaxis()->SetBinLabel(2,"HE");
   myhist->GetXaxis()->SetBinLabel(3,"HO");
   myhist->GetXaxis()->SetBinLabel(4,"HF");
   myhist->GetYaxis()->SetBinLabel(1,"Status");
   // Unknown status by default
   myhist->SetBinContent(1,1,-1);
   myhist->SetBinContent(2,1,-1);
   myhist->SetBinContent(3,1,-1);
   myhist->SetBinContent(4,1,-1);
   // Add ZDC at some point
   myhist->GetXaxis()->SetBinLabel(5,"ZDC");
   myhist->SetBinContent(5,1,-1); // no ZDC info known
   myhist->SetOption("textcolz");
     
   run_number=0;
   TrigCSC=TrigDT=TrigRPC=TrigGCT=0;
   L1ADataLabel   = iConfig.getUntrackedParameter<std::string>("L1ADataLabel" , "l1GtUnpack");
   prescaleLS_    = parameters_.getUntrackedParameter<int>("prescaleLS",  1);
   prescaleEvt_   = parameters_.getUntrackedParameter<int>("prescaleEvt", 1);
   GCTTriggerBit1_= parameters_.getUntrackedParameter<int>("GCTTriggerBit1", -1);         
   GCTTriggerBit2_= parameters_.getUntrackedParameter<int>("GCTTriggerBit2", -1);         
   GCTTriggerBit3_= parameters_.getUntrackedParameter<int>("GCTTriggerBit3", -1);         
   GCTTriggerBit4_= parameters_.getUntrackedParameter<int>("GCTTriggerBit4", -1);         
   GCTTriggerBit5_= parameters_.getUntrackedParameter<int>("GCTTriggerBit5", -1);         
   CosmicsCorr_   = parameters_.getUntrackedParameter<bool>("CosmicsCorr", true); 
   Debug_         = parameters_.getUntrackedParameter<bool>("Debug", true);    
   initialize();
}

HcalTimingMonitorModule::~HcalTimingMonitorModule(){}

// ------------ method called once each job just before starting event loop  ------------
void HcalTimingMonitorModule::beginJob(){}
// ------------ method called once each job just after ending the event loop  ------------
void HcalTimingMonitorModule::endJob(){}

void HcalTimingMonitorModule::initialize(){
  std::string str;
  dbe_->setCurrentFolder(monitorName_+"DebugPlots");
  str="L1MuGMTReadoutRecord_getDTBXCands";   DTcand  =dbe_->book1D(str,str,5,-0.5,4.5);
  str="L1MuGMTReadoutRecord_getBrlRPCCands"; RPCbcand=dbe_->book1D(str,str,5,-0.5,4.5);
  str="L1MuGMTReadoutRecord_getFwdRPCCands"; RPCfcand=dbe_->book1D(str,str,5,-0.5,4.5);
  str="L1MuGMTReadoutRecord_getCSCCands";    CSCcand =dbe_->book1D(str,str,5,-0.5,4.5);
  str="DT_OR_RPCb_OR_RPCf_OR_CSC";           OR      =dbe_->book1D(str,str,5,-0.5,4.5);
  
  str="HB Tower Energy (LinADC-PED)"; HBEnergy=dbe_->book1D(str,str,1000,-10,90);
  str="HE Tower Energy (LinADC-PED)"; HEEnergy=dbe_->book1D(str,str,1000,-10,90);
  str="HO Tower Energy (LinADC-PED)"; HOEnergy=dbe_->book1D(str,str,1000,-10,90);
  str="HF Tower Energy (LinADC-PED)"; HFEnergy=dbe_->book1D(str,str,1000,-10,90);
  
  dbe_->setCurrentFolder(monitorName_+"ShapePlots");
  str="HB Shape (DT Trigger)";        HBShapeDT  =dbe_->book1D(str,str,10,-0.5,9.5); 
  str="HB Shape (RPC Trigger)";       HBShapeRPC =dbe_->book1D(str,str,10,-0.5,9.5); 
  str="HB Shape (GCT Trigger)";       HBShapeGCT =dbe_->book1D(str,str,10,-0.5,9.5); 
  str="HO Shape (DT Trigger)";        HOShapeDT  =dbe_->book1D(str,str,10,-0.5,9.5); 
  str="HO Shape (RPC Trigger)";       HOShapeRPC =dbe_->book1D(str,str,10,-0.5,9.5); 
  str="HO Shape (GCT Trigger)";       HOShapeGCT =dbe_->book1D(str,str,10,-0.5,9.5); 
  str="HE+ Shape (CSC Trigger)";      HEShapeCSCp=dbe_->book1D(str,str,10,-0.5,9.5); 
  str="HE- Shape (CSC Trigger)";      HEShapeCSCm=dbe_->book1D(str,str,10,-0.5,9.5); 
  str="HF+ Shape (CSC Trigger)";      HFShapeCSCp=dbe_->book1D(str,str,10,-0.5,9.5); 
  str="HF- Shape (CSC Trigger)";      HFShapeCSCm=dbe_->book1D(str,str,10,-0.5,9.5); 
  
  dbe_->setCurrentFolder(monitorName_+"TimingPlots");
  str="HB Timing (DT Trigger)";       HBTimeDT   =dbe_->book1D(str,str,100,0,10);
  str="HB Timing (RPC Trigger)";      HBTimeRPC  =dbe_->book1D(str,str,100,0,10);
  str="HB Timing (GCT Trigger)";      HBTimeGCT  =dbe_->book1D(str,str,100,0,10);
  str="HO Timing (DT Trigger)";       HOTimeDT   =dbe_->book1D(str,str,100,0,10);
  str="HO Timing (RPC Trigger)";      HOTimeRPC  =dbe_->book1D(str,str,100,0,10);
  str="HO Timing (GCT Trigger)";      HOTimeGCT  =dbe_->book1D(str,str,100,0,10);
  str="HE+ Timing (CSC Trigger)";     HETimeCSCp =dbe_->book1D(str,str,100,0,10);
  str="HE- Timing (CSC Trigger)";     HETimeCSCm =dbe_->book1D(str,str,100,0,10);
  str="HF+ Timing (CSC Trigger)";     HFTimeCSCp =dbe_->book1D(str,str,100,0,10);
  str="HF- Timing (CSC Trigger)";     HFTimeCSCm =dbe_->book1D(str,str,100,0,10);
}

void HcalTimingMonitorModule::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
int HBcnt=0,HEcnt=0,HOcnt=0,HFcnt=0,eta,phi,depth,nTS;
int TRIGGER=0;
   counterEvt_++;
   if (prescaleEvt_<1)  return;
   if (counterEvt_%prescaleEvt_!=0)  return;

   run_number=iEvent.id().run();
   // Check GCT trigger bits
   edm::Handle< L1GlobalTriggerReadoutRecord > gtRecord;
   
   if (!iEvent.getByLabel( L1ADataLabel, gtRecord))
     return;
   const TechnicalTriggerWord tWord = gtRecord->technicalTriggerWord();
   const DecisionWord         dWord = gtRecord->decisionWord();
   //bool HFselfTrigger   = tWord.at(9);
   //bool HOselfTrigger   = tWord.at(11);
   bool GCTTrigger1      = dWord.at(GCTTriggerBit1_);     
   bool GCTTrigger2      = dWord.at(GCTTriggerBit2_);     
   bool GCTTrigger3      = dWord.at(GCTTriggerBit3_);     
   bool GCTTrigger4      = dWord.at(GCTTriggerBit4_);     
   bool GCTTrigger5      = dWord.at(GCTTriggerBit5_);     
   if(GCTTrigger1 || GCTTrigger2 || GCTTrigger3 || GCTTrigger4 || GCTTrigger5){ TrigGCT++; TRIGGER=+TRIG_GCT; }
   
   /////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////
   // define trigger trigger source (example from GMT group)
   edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle; 
   if (!iEvent.getByLabel(L1ADataLabel,gmtrc_handle)) return;
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
   	for(igmtrr=gmt_records.begin(); igmtrr!=gmt_records.end(); igmtrr++) {
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
	if(ndt[0]) DTcand->Fill(0);
	if(ndt[1]) DTcand->Fill(1);
	if(ndt[2]) DTcand->Fill(2);
	if(ndt[3]) DTcand->Fill(3);
	if(ndt[4]) DTcand->Fill(4);
	if(ncsc[0]) CSCcand->Fill(0);
	if(ncsc[1]) CSCcand->Fill(1);
	if(ncsc[2]) CSCcand->Fill(2);
	if(ncsc[3]) CSCcand->Fill(3);
	if(ncsc[4]) CSCcand->Fill(4);
	if(nrpcb[0]) RPCbcand->Fill(0);
	if(nrpcb[1]) RPCbcand->Fill(1);
	if(nrpcb[2]) RPCbcand->Fill(2);
	if(nrpcb[3]) RPCbcand->Fill(3);
	if(nrpcb[4]) RPCbcand->Fill(4);
	if(nrpcf[0]) RPCfcand->Fill(0);
	if(nrpcf[1]) RPCfcand->Fill(1);
	if(nrpcf[2]) RPCfcand->Fill(2);
	if(nrpcf[3]) RPCfcand->Fill(3);
	if(nrpcf[4]) RPCfcand->Fill(4);
	if(ndt[0]||nrpcb[0]||nrpcf[0]||ncsc[0]) OR->Fill(0);
	if(ndt[1]||nrpcb[1]||nrpcf[1]||ncsc[1]) OR->Fill(1);
	if(ndt[2]||nrpcb[2]||nrpcf[2]||ncsc[2]) OR->Fill(2);
	if(ndt[3]||nrpcb[3]||nrpcf[3]||ncsc[3]) OR->Fill(3);
	if(ndt[4]||nrpcb[4]||nrpcf[4]||ncsc[4]) OR->Fill(4);
	
  	if(ncsc[1]>0 ) { TrigCSC++;   TRIGGER=+TRIG_CSC;  }
  	if(ndt[1]>0  ) { TrigDT++;    TRIGGER=+TRIG_DT;   }
  	if(nrpcb[1]>0) { TrigRPC++;   TRIGGER=+TRIG_RPC;  }

   /////////////////////////////////////////////////////////////////////////////////////////
   /////////////////////////////////////////////////////////////////////////////////////////   
   if(counterEvt_<100){
     edm::Handle<HBHEDigiCollection> hbhe; 
     iEvent.getByLabel(hbheDigiCollectionTag_, hbhe);
     if (hbhe.isValid())
       {
	 for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
	   eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	   if(digi->id().subdet()==HcalBarrel) HBcnt++;
	   if(digi->id().subdet()==HcalEndcap) HEcnt++;
	   for(int i=0;i<nTS;i++)
	     if(digi->sample(i).adc()<20) set_hbhe(eta,phi,depth,digi->sample(i).capid(),adc2fC[digi->sample(i).adc()]);
	 } 
       }  
     edm::Handle<HODigiCollection> ho; 
     iEvent.getByLabel(hoDigiCollectionTag_, ho);
     if (ho.isValid())
     {
       for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
	 eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	 HOcnt++;
	 for(int i=0;i<nTS;i++)
	        if(digi->sample(i).adc()<20) set_ho(eta,phi,depth,digi->sample(i).capid(),adc2fC[digi->sample(i).adc()]);
       }   
     } // if

     edm::Handle<HFDigiCollection> hf;
     iEvent.getByLabel(hfDigiCollectionTag_, hf);
     if (hf.isValid())
       {
         for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
	   eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	   HFcnt++;
	   for(int i=0;i<nTS;i++) 
	     if(digi->sample(i).adc()<20) set_hf(eta,phi,depth,digi->sample(i).capid(),adc2fC[digi->sample(i).adc()]);
         }   
       }
   } // if (counterEvt<100)
   else{
     ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      double data[10];
      
      edm::Handle<HBHEDigiCollection> hbhe; 
      iEvent.getByLabel(hbheDigiCollectionTag_, hbhe);
      if (hbhe.isValid())
	{
	  for(HBHEDigiCollection::const_iterator digi=hbhe->begin();digi!=hbhe->end();digi++){
	    eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	    if(nTS>10) nTS=10;
	    if(digi->id().subdet()==HcalBarrel) HBcnt++;
	    if(digi->id().subdet()==HcalEndcap) HEcnt++;
	    double energy=0;
	    for(int i=0;i<nTS;i++){
	      data[i]=adc2fC[digi->sample(i).adc()]-get_ped_hbhe(eta,phi,depth,digi->sample(i).capid());
	      energy+=data[i];
	    }
	    if(digi->id().subdet()==HcalBarrel) HBEnergy->Fill(energy); 
	    if(digi->id().subdet()==HcalEndcap) HEEnergy->Fill(energy); 
	    if(!isSignal(data,nTS)) continue;
	    for(int i=0;i<nTS;i++){
	      if(data[i]>-1.0){
		if(digi->id().subdet()==HcalBarrel && (TRIGGER|TRIG_DT)==TRIG_DT)             HBShapeDT->Fill(i,data[i]);
		if(digi->id().subdet()==HcalBarrel && (TRIGGER|TRIG_RPC)==TRIG_RPC)           HBShapeRPC->Fill(i,data[i]);
		if(digi->id().subdet()==HcalBarrel && (TRIGGER|TRIG_GCT)==TRIG_GCT)           HBShapeGCT->Fill(i,data[i]); 
		if(digi->id().subdet()==HcalEndcap && (TRIGGER|TRIG_CSC)==TRIG_CSC && eta>0)  HEShapeCSCp->Fill(i,data[i]);
		if(digi->id().subdet()==HcalEndcap && (TRIGGER|TRIG_CSC)==TRIG_CSC && eta<0)  HEShapeCSCm->Fill(i,data[i]);   
	      }  
	    }
	    double Time=GetTime(data,nTS);
	    if(digi->id().subdet()==HcalBarrel){
	      if(CosmicsCorr_) Time+=(7.5*sin((phi*5.0)/180.0*3.14159))/25.0;
	      if((TRIGGER&TRIG_DT)==TRIG_DT)  HBTimeDT ->Fill(GetTime(data,nTS));
	      if((TRIGGER&TRIG_RPC)==TRIG_RPC) HBTimeRPC->Fill(GetTime(data,nTS));
	      if((TRIGGER&TRIG_GCT)==TRIG_GCT) HBTimeGCT->Fill(GetTime(data,nTS));
	    }else{
	      if(CosmicsCorr_) Time+=(3.5*sin((phi*5.0)/180.0*3.14159))/25.0; 
	      if(digi->id().subdet()==HcalEndcap && (TRIGGER&TRIG_CSC)==TRIG_CSC && eta>0) HETimeCSCp->Fill(Time);
	      if(digi->id().subdet()==HcalEndcap && (TRIGGER&TRIG_CSC)==TRIG_CSC && eta<0) HETimeCSCm->Fill(Time);  
	    }
	  }    
	} // if (...)

      edm::Handle<HODigiCollection> ho; 
      iEvent.getByLabel(hoDigiCollectionTag_, ho);
      if (ho.isValid())
	{
	  for(HODigiCollection::const_iterator digi=ho->begin();digi!=ho->end();digi++){
	    eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	    if(nTS>10) nTS=10;
	    HOcnt++; 
	    double energy=0;
	    for(int i=0;i<nTS;i++){
	      data[i]=adc2fC[digi->sample(i).adc()]-get_ped_ho(eta,phi,depth,digi->sample(i).capid());
	      energy+=data[i];
	    }     
	    HOEnergy->Fill(energy); 
	    if(!isSignal(data,nTS)) continue;
	    for(int i=0;i<nTS;i++){
	      if(data[i]>-1.0){
		if((TRIGGER&TRIG_DT)==TRIG_DT)    HOShapeDT->Fill(i,data[i]);
		if((TRIGGER&TRIG_RPC)==TRIG_RPC)  HOShapeRPC->Fill(i,data[i]);
		if((TRIGGER&TRIG_GCT)==TRIG_GCT)  HOShapeGCT->Fill(i,data[i]); 	     
	      }
	    }  
	    double Time=GetTime(data,nTS);
	    if(CosmicsCorr_) Time+=(12.0*sin((phi*5.0)/180.0*3.14159))/25.0;    
	    if((TRIGGER&TRIG_DT)==TRIG_DT)   HOTimeDT->Fill(Time);
	    if((TRIGGER&TRIG_RPC)==TRIG_RPC) HOTimeRPC->Fill(Time);
	    if((TRIGGER&TRIG_GCT)==TRIG_GCT) HOTimeGCT->Fill(Time);
	  }   
	}// if (ho)

      edm::Handle<HFDigiCollection> hf; 
      iEvent.getByLabel(hfDigiCollectionTag_, hf);
      if (hf.isValid())
	{
	  for(HFDigiCollection::const_iterator digi=hf->begin();digi!=hf->end();digi++){
            eta=digi->id().ieta(); phi=digi->id().iphi(); depth=digi->id().depth(); nTS=digi->size();
	    if(nTS>10) nTS=10;
            HFcnt++; 
	    double energy=0;
	    for(int i=0;i<nTS;i++){
	       data[i]=adc2fC[digi->sample(i).adc()]-get_ped_hf(eta,phi,depth,digi->sample(i).capid());
	       energy+=data[i]; 
	    }
	    HFEnergy->Fill(energy); 
	    if(energy<15.0) continue;
	    for(int i=0;i<nTS;i++){
	      if(data[i]>-1.0){
 	        if((TRIGGER&TRIG_CSC)==TRIG_CSC && eta>0)  HFShapeCSCp->Fill(i,data[i]); 
	        if((TRIGGER&TRIG_CSC)==TRIG_CSC && eta<0)  HFShapeCSCm->Fill(i,data[i]);    
	      } 
	    }
	    if((TRIGGER&TRIG_CSC)==TRIG_CSC && eta>0) HFTimeCSCp->Fill(GetTime(data,nTS)); 
 	    if((TRIGGER&TRIG_CSC)==TRIG_CSC && eta<0) HFTimeCSCm->Fill(GetTime(data,nTS)); 
        }   
	} // if (hf)
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   }
   if(Debug_) if((counterEvt_%100)==0) printf("Run: %i,Events processed: %i (HB: %i towers,HE: %i towers,HO: %i towers,HF: %i towers)"
                                        " CSC: %i DT: %i RPC: %i GCT: %i\n",
                                        run_number,counterEvt_,HBcnt,HEcnt,HOcnt,HFcnt,TrigCSC,TrigDT,TrigRPC,TrigGCT);
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
   //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
}
//define this as a plug-in
DEFINE_FWK_MODULE(HcalTimingMonitorModule);


