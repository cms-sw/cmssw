/**************************************
 *         Autor David Lomidze        *
 *          INFN di Napoli            *
 *************************************/

#include <string>
#include <sstream>
#include <map>
#include <DQM/RPCMonitorClient/interface/RPCChamberQuality.h>
#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"
#include "DQM/RPCMonitorDigi/interface/utils.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//DQM Services
#include "DQMServices/Core/interface/DQMStore.h"
//DataFormats
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
//Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
using namespace edm;
using namespace std;
RPCChamberQuality::RPCChamberQuality(const ParameterSet& ps ){
  LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Constructor";
  
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 9);
  prefixDir_ = ps.getUntrackedParameter<string>("RPCGlobalFolder", "RPC/RecHits/SummaryHistograms");
  minEvents = ps.getUntrackedParameter<int>("MinimumRPCEvents", 10000);
}

RPCChamberQuality::~RPCChamberQuality(){
  LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Destructor ";
  dbe_=0;
}

void RPCChamberQuality::beginJob(){
  LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Begin job ";
  dbe_ = Service<DQMStore>().operator->();
}


// void RPCGoodessTest::endJob(void){
//   if(saveRootFile) dbe->save(RootFileName); 
//   dbe = 0;
// }



void RPCChamberQuality::beginRun(const Run& r, const EventSetup& iSetup){
  LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Begin run";
  
  init_ = false;  
  
  MonitorElement* bq; //barrel quality histo
  MonitorElement* epq; //endcap+ quality histo
  MonitorElement* enq; //endcap- quality histo
  
  MonitorElement* me;
  dbe_->setCurrentFolder(prefixDir_);
  
  stringstream histoName;
  
  rpcdqm::utils rpcUtils;

  histoName.str("");
  histoName<<"RPCChamberQuality_Barrel"; 
  bq = dbe_->get(prefixDir_+"/"+ histoName.str());
    if (0!=bq) {
      dbe_->removeElement(bq->getName());
    }
    bq = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  7, 0.5, 7.5);
    bq->setBinLabel(1, "Good", 1);
    bq->setBinLabel(2, "OFF", 1);
    bq->setBinLabel(3, "Nois.St", 1);
    bq->setBinLabel(4, "Nois.Ch", 1);
    bq->setBinLabel(5, "Part.Dead", 1);
    bq->setBinLabel(6, "Dead", 1);
    bq->setBinLabel(7, "Bad.Shape", 1);
    
    histoName.str("");
    histoName<<"RPCChamberQuality_EndcapPositive"; 
    epq = dbe_->get(prefixDir_+"/"+ histoName.str());
    if (0!=epq) {
      dbe_->removeElement(epq->getName());
    }
    epq = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  7, 0.5, 7.5);
    epq->setBinLabel(1, "Good", 1);
    epq->setBinLabel(2, "OFF", 1);
    epq->setBinLabel(3, "Nois.St", 1);
    epq->setBinLabel(4, "Nois.Ch", 1);
    epq->setBinLabel(5, "Part.Dead", 1);
    epq->setBinLabel(6, "Dead", 1);
    epq->setBinLabel(7, "Bad.Shape", 1);
    
    histoName.str("");
    histoName<<"RPCChamberQuality_EndcapNegative"; 
    enq = dbe_->get(prefixDir_+"/"+ histoName.str());
    if (0!=enq) {
      dbe_->removeElement(enq->getName());
    }
    enq = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  7, 0.5, 7.5);
    enq->setBinLabel(1, "Good", 1);
    enq->setBinLabel(2, "OFF", 1);
    enq->setBinLabel(3, "Nois.St", 1);
    enq->setBinLabel(4, "Nois.Ch", 1);
    enq->setBinLabel(5, "Part.Dead", 1);
    enq->setBinLabel(6, "Dead", 1);
    enq->setBinLabel(7, "Bad.Shape", 1);


  histoName.str("");
  histoName<<"RPC_System_Quality_Overview"; 
  me = dbe_->get(prefixDir_+"/"+ histoName.str());
    if (0!=me) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  7, 0.5, 7.5, 3, 0.5, 3.5);
    me->setBinLabel(1, "E+", 2);
    me->setBinLabel(2, "B", 2);
    me->setBinLabel(3, "E-", 2);
    
    me->setBinLabel(1, "Good", 1);
    me->setBinLabel(2, "OFF", 1);
    me->setBinLabel(3, "Nois.St", 1);
    me->setBinLabel(4, "Nois.Ch", 1);
    me->setBinLabel(5, "Part.Dead", 1);
    me->setBinLabel(6, "Dead", 1);
    me->setBinLabel(7, "Bad.Shape", 1);
    
    for(int w=-2; w<3;w++){
    
    
    histoName.str("");
    histoName<<"RPCChamberQuality_Roll_vs_Sector_Wheel"<<w;       //  2D histo for RPC Qtest
    me = 0;
    me = dbe_->get(prefixDir_+"/"+ histoName.str());
    if (0!=me) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
    for(int bin =1; bin<13;bin++) {
      histoName.str("");
      histoName<<"Sec"<<bin;
      me->setBinLabel(bin,histoName.str().c_str(),1);
    }
    
    me->setBinLabel(1, "RB1in_B", 2);
    me->setBinLabel(2, "RB1in_F", 2);
    me->setBinLabel(3, "RB1out_B", 2);
    me->setBinLabel(4, "RB1out_F", 2);
    me->setBinLabel(5, "RB2in_B", 2);
    me->setBinLabel(6, "RB2in_F", 2);
    me->setBinLabel(7, "RB2in_M", 2);
    me->setBinLabel(8, "RB2out_B", 2);
    me->setBinLabel(9, "RB2out_F", 2);
    me->setBinLabel(10, "RB3-_B", 2);
    me->setBinLabel(11, "RB3-_F", 2);
    me->setBinLabel(12, "RB3+_B", 2);
    me->setBinLabel(13, "RB3+_F", 2);
    me->setBinLabel(14, "RB4,-,--_B", 2);
    me->setBinLabel(15, "RB4,-,--_F", 2);
    me->setBinLabel(16, "RB4+,-+_B", 2);
    me->setBinLabel(17, "RB4+,-+_F", 2);
    me->setBinLabel(18, "RB4+-_B", 2);
    me->setBinLabel(19, "RB1+-_F", 2);
    me->setBinLabel(20, "RB4++_B", 2);
    me->setBinLabel(21, "RB1++_F", 2);
   
    histoName.str("");
    histoName<<"RPCChamberQuality_Distribution_Wheel"<<w;       //  ClusterSize in first bin, distribution
    me=0;
    me = dbe_->get(prefixDir_+"/"+ histoName.str());
    if (0!=me ) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  7, 0.5, 7.5);
    me->setBinLabel(1, "Good", 1);
    me->setBinLabel(2, "OFF", 1);
    me->setBinLabel(3, "Nois.St", 1);
    me->setBinLabel(4, "Nois.Ch", 1);
    me->setBinLabel(5, "Part.Dead", 1);
    me->setBinLabel(6, "Dead", 1);
    me->setBinLabel(7, "Bad.Shape", 1);
    
  }//end loop on wheels

  for(int i=-3; i<4; i++) { // ENDCAP
    if(i!=0) {
      histoName.str("");
      histoName<<"RPCChamberQuality_Ring_vs_Segment_Disk"<<i;       //  2D histo for RPC Qtest
      me = 0;
      me = dbe_->get(prefixDir_+"/"+ histoName.str());
      if (0!=me) {
	dbe_->removeElement(me->getName());
      }
      me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  36, 0.5, 36.5, 6, 0.5, 6.5);
      rpcUtils.labelXAxisSegment(me);
      rpcUtils.labelYAxisRing(me, 2);
    } //if 
  } // for
}

void RPCChamberQuality::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCChamberQuality::analyze(const Event& iEvent, const EventSetup& c) {}

 void RPCChamberQuality::endRun(const edm::Run& r, const edm::EventSetup& iSetup)  {   

  LogVerbatim ("rpceventsummary") <<"[RPCChamberQuality]: End of LS transition, performing DQM client operation";

   MonitorElement * RpcEvents = NULL;
   stringstream meName;
   
   meName.str("");
   meName<<"RPC/RecHits/SummaryHistograms/RPCEvents"; 
   int rpcEvents=0;
   RpcEvents = dbe_->get(meName.str());

   if(RpcEvents) rpcEvents= (int)RpcEvents->getEntries();


   if(rpcEvents < minEvents) return;

    
    ESHandle<RPCGeometry> rpcGeo;
    iSetup.get<MuonGeometryRecord>().get(rpcGeo);
    
    MonitorElement * RCQ=NULL;          // Monitoring Element RPC Chamber Quality (RCQ)
    MonitorElement * RCQD=NULL;         // Monitoring Element RPC Chamber Quality Distr (RCQD)
     
   	    
    // RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
    stringstream mme;
    MonitorElement * myMe=NULL;
    MonitorElement * CLS=NULL;
    MonitorElement * MULT=NULL;
    MonitorElement * NoisySt=NULL;
    MonitorElement * Chip=NULL;
    MonitorElement * HV=NULL;
    MonitorElement * LV=NULL;
    MonitorElement* bq = NULL;
    MonitorElement* enq = NULL;
    MonitorElement* epq = NULL;
    
    float dead =0;
    float firstbin= 0;
    float noisystrips = 0;
    float mult = 0;
    float asy = 0;
    //int good, off, ns, nc, pd, d, as; 

    meName.str("");
    meName<<"RPC/RecHits/SummaryHistograms/RPCChamberQuality_EndcapNegative"; 
    enq = dbe_ -> get(meName.str());
    
    meName.str("");
    meName<<"RPC/RecHits/SummaryHistograms/RPCChamberQuality_EndcapPositive"; 
    epq = dbe_ -> get(meName.str());
    
    meName.str("");
    meName<<"RPC/RecHits/SummaryHistograms/RPCChamberQuality_Barrel"; 
    bq  = dbe_ -> get(meName.str());
    
    for (int i=-2; i<3; i++) {    
      
      meName.str("");
      meName<<"RPC/RecHits/SummaryHistograms/RPCChamberQuality_Roll_vs_Sector_Wheel"<<i; 
      RCQ = dbe_ -> get(meName.str());
      
      meName.str("");
      meName<<"RPC/RecHits/SummaryHistograms/RPCChamberQuality_Distribution_Wheel"<<i; 
      RCQD = dbe_ -> get(meName.str());
      
      RCQD->Reset();
      RCQ->Reset();
      //get HV Histo
      meName.str("");                        
      meName<<"RPC/RecHits/SummaryHistograms/HVStatus_Wheel_"<<i;
      HV = dbe_ -> get(meName.str());
	
      //get LV Histo
      meName.str("");                     
      meName<<"RPC/RecHits/SummaryHistograms/LVStatus_Wheel_"<<i; 
      LV = dbe_ -> get(meName.str());
      
      mme.str("");
      mme << "RPC/RecHits/SummaryHistograms/DeadChannelFraction_Roll_vs_Sector_Wheel"<<i;
      myMe = dbe_->get(mme.str());
	
   
	
      for(int x=1; x<13; x++) {
	int roll;
	if(x==4) roll=22;
	else if(x==9 || x==11) roll=16;
	else roll=18;
	
	for(int y=1; y<roll; y++) {
	  int hv=0;
	  int lv=0;
	  bool flag=false;
	  
	  if(HV && LV) {
	    hv = (int)HV ->getBinContent(x,y);
	    lv = (int)LV ->getBinContent(x,y);
	    flag = true;
	  }
  
	  if(flag && (hv!=1 || lv!=1)) {                                        //HV & LV
	    // Chamber OFF
	    if (RCQ) RCQ -> setBinContent(x,y, 2);
	    if (RCQD) { 
	      RCQD -> Fill(2, 1); bq->Fill(2,1); 
	    }
	  }else {                                                              //DEAD
	    
	    if(myMe) dead= myMe -> getBinContent(x,y);
	    if(dead>=0.80 && rpcEvents>50000) {
	      // declare as DEAD chamber. fill map by a number
	      if (RCQ)	RCQ -> setBinContent(x,y, 6);
	      if (RCQD)	{RCQD -> Fill(6, 1); bq->Fill(6,1); }
	    }else if (0.33<=dead && dead<0.80 && rpcEvents>=20000){
	      //Partially DEAD!!! Fill map by a number 
	      //do dead FEB/CHIP s calculation
	      if (RCQ)	RCQ -> setBinContent(x,y, 5);
	      if (RCQD)	{
		RCQD -> Fill(5, 1);
		bq->Fill(5,1); 
	      }
	    }else {                                                             //1Bin
	      //check 1bin
	      meName.str("");
	      meName<<"RPC/RecHits/SummaryHistograms/ClusterSizeIn1Bin_Roll_vs_Sector_Wheel" << i;
	      CLS = dbe_ -> get(meName.str());
		
	      meName.str("");
	      meName<<"RPC/RecHits/SummaryHistograms/RPCNoisyStrips_Roll_vs_Sector_Wheel" << i;
	      NoisySt = dbe_ -> get(meName.str());
		
 	      if(CLS && NoisySt){
		firstbin = CLS -> getBinContent(x,y);
		noisystrips = NoisySt -> getBinContent(x,y);
	      }	

	      if(firstbin >= 0.88) {
		// noisely strip !!! fill map by a number !
		if (RCQ)  RCQ -> setBinContent(x,y, 3);
		if (RCQD)  RCQD -> Fill(3, 1);
	      } else if(noisystrips>0) { 
		if (RCQ)	  RCQ -> setBinContent(x,y, 3);
		if (RCQD)	  { 
		  RCQD -> Fill(3, 1); 
		  bq->Fill(3,1); 
		}
	      }else {
		//check Multiplicity to spot noisely Chamber
		
		meName.str("");
		meName<<"RPC/RecHits/SummaryHistograms/NumberOfDigi_Mean_Roll_vs_Sector_Wheel" << i;
		//meName<<"RPC/RecHits/SummaryHistograms/ClusterSizeIn1Bin_Roll_vs_Sector_Wheel" << i;
		MULT = dbe_ -> get(meName.str());
		  
		
		  if(MULT) mult = MULT -> getBinContent(x,y);
		  
		  if(mult>=6) {
		    // Declare chamber as noisely! Fill map by a number !
		    if (RCQ)    RCQ -> setBinContent(x,y, 4);
		    if (RCQD)	  {  RCQD -> Fill(4, 1);
		    bq->Fill(4,1); 
		    }
		  }else {
		    meName.str("");
		    meName<<"RPC/RecHits/SummaryHistograms/AsymmetryLeftRight_Roll_vs_Sector_Wheel" << i;
		    Chip = dbe_ -> get(meName.str());
		   

		    if (Chip) asy = Chip->getBinContent(x,y);

		    if(asy>0.35) {
		 	if (RCQ)     RCQ -> setBinContent(x,y, 7);
		 	if (RCQ)    { RCQD -> Fill(7, 1); 
			bq->Fill(7,1); 
			}
		    }else {
		      if (RCQ)  RCQ -> setBinContent(x,y, 1);
		      if (RCQ)  { RCQD -> Fill(1, 1);
		      bq->Fill(1,1); 
		      }
		    }
		  } 
	      }
	    }
	  }
	}
      } // loop by chamber
    } // loop by Wheels


    // Endcap
    MonitorElement * meDEAD = NULL;
    myMe=NULL;
    CLS=NULL;
    MULT=NULL;
    NoisySt=NULL;
    Chip=NULL;
    HV=NULL;
    LV=NULL;
    
    for(int i=-3; i<4; i++) {
      if(i!=0) {
	meName.str("");
	meName<<"RPC/RecHits/SummaryHistograms/RPCChamberQuality_Ring_vs_Segment_Disk"<<i; 
	RCQ = dbe_ -> get(meName.str());
	if(RCQ) RCQ->Reset();
	mme.str("");
	mme << "RPC/RecHits/SummaryHistograms/DeadChannelFraction_Ring_vs_Segment_Disk"<<i;
	meDEAD = dbe_->get(mme.str());
	
	for(int x=1; x<37; x++) {
	  for (int y=1; y<7; y++) {
	    dead=0;
	    
	    if (meDEAD) dead = meDEAD->getBinContent(x,y);
	    
	    if(dead>=0.8) { 
	      RCQ -> setBinContent(x,y,6); 
	      if(i<0) enq -> Fill(6,1) ;
	      else epq -> Fill(6,1);
	    }
	    else if (dead>=0.33 && dead<0.8) RCQ->setBinContent(x,y,5);
	    else {
	      mme.str("");
	      mme << "RPC/RecHits/SummaryHistograms/ClusterSizeIn1Bin_Ring_vs_Segment_Disk"<<i;
	      CLS = dbe_->get(mme.str());
	      
	      mme.str("");
	      mme << "RPC/RecHits/SummaryHistograms/RPCNoisyStrips_Ring_vs_Segment_Disk"<<i;
	      NoisySt = dbe_->get(mme.str());
	      if(CLS && NoisySt) {
		firstbin = CLS->getBinContent(x,y);
		noisystrips = NoisySt -> getBinContent(x,y);
	      }
	      if(firstbin>0.88) { 
		RCQ -> setBinContent(x,y,3);
		if(i<0) enq -> Fill(3,1) ;
		else epq -> Fill(3,1);
	      }
	      else if(noisystrips>0) {
		RCQ -> setBinContent(x,y,3);
		if(i<0) enq -> Fill(6,1) ;
		else epq -> Fill(6,1);
	      }
	      else {
		mme.str("");
		mme << "RPC/RecHits/SummaryHistograms/NumberOfDigi_Mean_Ring_vs_Segment_Disk"<<i;
		MULT = dbe_->get(mme.str());
		
		if(MULT) mult=MULT->getBinContent(x,y);
		if(mult>=6) {
		  RCQ->setBinContent(x,y,4);
		  if(i<0) enq -> Fill(4,1) ;
		  else epq -> Fill(4,1);
		}
		else {
		  mme.str("");
		  mme << "RPC/RecHits/SummaryHistograms/AsymmetryLeftRight_Ring_vs_Segment_Disk"<<i;
		  myMe = dbe_->get(mme.str());
		  if(myMe) asy = myMe->getBinContent(x,y);
		  if(asy>0.35) {
		    RCQ->setBinContent(x,y,7);
		    if(i<0) enq -> Fill(7,1) ;
		    else epq -> Fill(7,1);
		  }
		  else {
		    RCQ->setBinContent(x,y,1);
		    if(i<0) enq -> Fill(1,1) ;
		    else epq -> Fill(1,1);
		  }
		}
	      }
	    }
	    
	  } //loop by Xaxis
	} //loop by Yaxis
      } //if on 0
    } // loop by Endcap
    
    MonitorElement * rpcperc=NULL;
    mme.str("");
    mme<<"RPC/RecHits/SummaryHistograms/RPC_System_Quality_Overview"; 
    rpcperc = dbe_->get(mme.str());
    
    //    float totperc=0;
    int b_ch = 0;
    int ep_ch = 0;
    int en_ch =0;
    double perc=0;
    
    b_ch = bq -> getEntries();
    en_ch = enq -> getEntries();
    ep_ch = epq -> getEntries();
    
        
    for(int i=1; i<8; i++) {
      
           
      perc = ((bq->getBinContent(i)) * 100)/b_ch;
      rpcperc -> setBinContent(i, 2, perc);
      
      perc = ((enq->getBinContent(i)) * 100)/en_ch;
      rpcperc -> setBinContent(i, 3, perc);

      perc = ((epq->getBinContent(i)) * 100)/ep_ch;
      rpcperc -> setBinContent(i, 1, perc);
    }
    
    

}

void RPCChamberQuality::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {    }




