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
  
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 1);
  prefixDir_ = ps.getUntrackedParameter<string>("RPCPrefixDir", "RPC/RecHits/SummaryHistograms/");
  verbose_=ps.getUntrackedParameter<bool>("VerboseLevel", 0);

  // saveRootFile  = pset.getUntrackedParameter<bool>("DigiDQMSaveRootFile", true); 
  //  RootFileName  = pset.getUntrackedParameter<string>("RootFileNameDigi", "RPCClient.root"); 
  
}

RPCChamberQuality::~RPCChamberQuality(){
  LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Destructor ";
  dbe_=0;
}

void RPCChamberQuality::beginJob(const EventSetup& iSetup){
  LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Begin job ";
  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(verbose_);
}


// void RPCGoodessTest::endJob(void){
//   if(saveRootFile) dbe->save(RootFileName); 
//   dbe = 0;
// }



void RPCChamberQuality::beginRun(const Run& r, const EventSetup& c){
  LogVerbatim ("rpceventsummary") << "[RPCChamberQuality]: Begin run";
  
  
  MonitorElement* me;
  dbe_->setCurrentFolder(prefixDir_);
  
  stringstream histoName;
  
  
  for(int w=-2; w<3;w++){
    
    
    histoName.str("");
    histoName<<"RPCChamberQuality_Roll_vs_Sector_Wheel"<<w;       //  2D histo for RPC Qtest
    if ( me = dbe_->get(prefixDir_ + histoName.str()) ) {
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
    me->setBinLabel(15, "RB4,-,--,F", 2);
    me->setBinLabel(16, "RB4+,-+_B", 2);
    me->setBinLabel(17, "RB4+,-+_F", 2);
    me->setBinLabel(18, "RB4+-_B", 2);
    me->setBinLabel(19, "RB1+-_F", 2);
    me->setBinLabel(20, "RB4++_B", 2);
    me->setBinLabel(21, "RB1++_F", 2);
    
    histoName.str("");
    histoName<<"RPCChamberQuality_Distribution_Wheel"<<w;       //  ClusterSize in first bin, distribution
    if ( me = dbe_->get(prefixDir_+ histoName.str()) ) {
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
  
}

void RPCChamberQuality::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCChamberQuality::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCChamberQuality::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCChamberQualit]: End of LS transition, performing DQM client operation";
  
  // counts number of lumiSegs 
  nLumiSegs_ = lumiSeg.id().luminosityBlock();
  
  //check some statements and prescale Factor
  if(nLumiSegs_%prescaleFactor_ == 0) {
    
    ESHandle<RPCGeometry> rpcGeo;
    iSetup.get<MuonGeometryRecord>().get(rpcGeo);
    
    MonitorElement * RCQ;          // Monitoring Element RPC Chamber Quality (RCQ)
    MonitorElement * RCQD;         // Monitoring Element RPC Chamber Quality Distr (RCQD)
     
    stringstream meName;
    //Loop on chambers
    // for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    // if( dynamic_cast< RPChamber* >( *it ) != 0 ){
    //	RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
    //	std::vector< const RPCRoll*> roles = (ch->rolls());
	
	
	//Loop on rolls in given chamber
	//for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
    // RPCDetId detId = (*r)->id();
    //	  rpcdqm::utils prova;
	  
	  
    // int nr = prova.detId2RollNr(detId);
    //if(detId.region() !=0) continue;
	  
	  //Get Occupancy ME for roll
    //  RPCGeomServ RPCname(detId);
	  
    //  if (detId.region()==0){
	    
    //    string YLabel = RPCname.shortname();

    
  

	    
    RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
    stringstream mme;
    MonitorElement * myMe;
    MonitorElement * CLS;
    MonitorElement * MULT;
    MonitorElement * NoisySt;
    MonitorElement * Chip;
    
    for (int i=-2; i<3; i++) {    
      
      meName.str("");
      meName<<"RPC/RecHits/SummaryHistograms/RPCChamberQuality_Roll_vs_Sector_Wheel"<<i; 
      RCQ = dbe_ -> get(meName.str());
      
      meName.str("");
      meName<<"RPC/RecHits/SummaryHistograms/RPCChamberQuality_Distribution_Wheel"<<i; 
      RCQD = dbe_ -> get(meName.str());
      
      
      
      mme.str("");
      mme << "RPC/RecHits/SummaryHistograms/DeadChannelFraction_Roll_vs_Sector_Wheel"<<i;
      
      myMe = dbe_->get(mme.str());
      //if (RCQ) cout<<"found me"<<endl;
      
      //if (!myMe)continue;
      
      if (myMe) {
	for(int x=1; x<13; x++) {
	  int roll;
	  if(x==4) roll=22;
	  else if(x==9 || x==11) roll=16;
	  else roll=18;
	  for(int y=1; y<roll; y++) {
	    //cout<<" xy "<< myMe -> getBinContent(x,y)<<endl;
	    float dead = myMe -> getBinContent(x,y);
	    if(dead>=80) {
	      // declare as DEAD chamber. fill map by a number
	      RCQ -> setBinContent(x,y, 6);
	      RCQD -> Fill(6, 1);
	    }
	    
	    else if (33<=dead && dead<80){
	      //Partially DEAD!!! Fill map by a number 
	      //do dead FEB/CHIP s calculation
	      RCQ -> setBinContent(x,y, 5);
	      RCQD -> Fill(5, 1);
	      
	    }
	    
	    else {
	      //check 1bin
	      meName.str("");
	      meName<<"RPC/RecHits/SummaryHistograms/ClusterSizeIn1Bin_Roll_vs_Sector_Wheel" << i;
	      CLS = dbe_ -> get(meName.str());
	      
	      meName.str("");
	      meName<<"RPC/RecHits/SummaryHistograms/RPCNoisyStrips_Roll_vs_Sector_Wheel" << i;
	      NoisySt = dbe_ -> get(meName.str());


	      float firstbin = CLS -> getBinContent(x,y);
	      int noisystrips = NoisySt -> getBinContent(x,y);
	      
	      if(firstbin >= 0.88) {
		// noisely strip !!! fill map by a number !
		RCQ -> setBinContent(x,y, 3);
		RCQD -> Fill(3, 1);
	      } 
	      
	      else if(noisystrips>0) { 
	      	RCQ -> setBinContent(x,y, 3);
		RCQD -> Fill(3, 1);
	      }
	      
	      else {
		//check Multiplicity to spot noisely Chamber
		
		meName.str("");
		meName<<"RPC/RecHits/SummaryHistograms/NumberOfDigi_Roll_vs_Sector_Wheel" << i;
		//meName<<"RPC/RecHits/SummaryHistograms/ClusterSizeIn1Bin_Roll_vs_Sector_Wheel" << i;
		MULT = dbe_ -> get(meName.str());
		
		
		float mult = MULT -> getBinContent(x,y);
		
		if(mult>=6) {
		  // Declare chamber as noisely! Fill map by a number !
		  RCQ -> setBinContent(x,y, 4);
		  RCQD -> Fill(4, 1);
		}
		else {
		  //Declare Chember as GOOD !!!! Fill map by a number !
		  meName.str("");
		  meName<<"RPC/RecHits/SummaryHistograms/AsymmetryLeftRight_Roll_vs_Sector_Wheel" << i;
		  Chip = dbe_ -> get(meName.str());
		  
		  if(Chip->getBinContent(x,y)>0.35) {
		    RCQ -> setBinContent(x,y, 7);
		    RCQD -> Fill(7, 1);
		  }
		  else {
		    RCQ -> setBinContent(x,y, 1);
		    RCQD -> Fill(1, 1);
		  }
		  
		} 
	      }
	    }
	    
	    //MEAND->Fill( MEAN->getBinContent(x,y));
	    //if (CLSD) CLSD->Fill(CLS->getBinContent(x,y));
	  }
	}
	
      }
      
      //	  }//End loop on Barrel
      //	} // end loop on rolls in given chamber
      // }//End loop on chamber
      
      
    }
  }
}


