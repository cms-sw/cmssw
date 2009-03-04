/**************************************
 *         Autor David Lomidze        *
 *          INFN di Napoli            *
 *************************************/

#include <string>
#include <sstream>
#include <map>
#include <DQM/RPCMonitorClient/interface/RPCClusterSizeTest.h>
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
RPCClusterSizeTest::RPCClusterSizeTest(const ParameterSet& ps ){
  LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: Constructor";
  
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 1);
  prefixDir_ = ps.getUntrackedParameter<string>("RPCPrefixDir", "RPC/RecHits/SummaryHistograms/");
  verbose_=ps.getUntrackedParameter<bool>("VerboseLevel", 0);
  
}

RPCClusterSizeTest::~RPCClusterSizeTest(){
  LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: Destructor ";
  dbe_=0;
}

void RPCClusterSizeTest::beginJob(const EventSetup& iSetup){
  LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: Begin job ";
  dbe_ = Service<DQMStore>().operator->();
  dbe_->setVerbose(verbose_);
}

void RPCClusterSizeTest::beginRun(const Run& r, const EventSetup& c){
  LogVerbatim ("rpceventsummary") << "[RPCClusterSizeTest]: Begin run";
  
  
  MonitorElement* me;
  dbe_->setCurrentFolder(prefixDir_);
  
  stringstream histoName;
  
  
  for(int w=-2; w<3;w++){
    
    
    histoName.str("");
    histoName<<"ClusterSizeIn1Bin_Roll_vs_Sector_Wheel"<<w;       // ClusterSize in first bin norm. by Entries (2D Roll vs Sector)       
    if ( me = dbe_->get(prefixDir_ + histoName.str()) ) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
    for(int bin =1; bin<13;bin++) {
      histoName.str("");
      histoName<<"Sec"<<bin;
      me->setBinLabel(bin,histoName.str().c_str(),1);
    }
    
    
    histoName.str("");
    histoName<<"ClusterSizeIn1Bin_Distribution_Wheel"<<w;       //  ClusterSize in first bin, distribution
    if ( me = dbe_->get(prefixDir_+ histoName.str()) ) {
      dbe_->removeElement(me->getName());
    }
    me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, 0.02, 1.02);
    
    
  //   histoName.str("");
//     histoName<<"ClusterSizeMean_Roll_vs_Sector_Wheel"<<w;       // Avarage ClusterSize (2D Roll vs Sector)   
//     if ( me = dbe_->get(prefixDir_ + histoName.str()) ) {
//       dbe_->removeElement(me->getName());
//     }
//     me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
//     for(int bin =1; bin<13;bin++) {
//       histoName.str("");
//       histoName<<"Sec"<<bin;
//       me->setBinLabel(bin,histoName.str().c_str(),1);
//     }
    
    
//     histoName.str("");
//     histoName<<"ClusterSizeMean_Distribution_Wheel"<<w;       //  Avarage ClusterSize Distribution
//     if ( me = dbe_->get(prefixDir_ + histoName.str()) ) {
//       dbe_->removeElement(me->getName());
//     }
//     me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.5, 10.5);
   
    
  }//end loop on wheels
  
}

void RPCClusterSizeTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCClusterSizeTest::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCClusterSizeTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCClusterSizeTest]: End of LS transition, performing DQM client operation";
  
  // counts number of lumiSegs 
  nLumiSegs_ = lumiSeg.id().luminosityBlock();
  
  //check some statements and prescale Factor
  if(nLumiSegs_%prescaleFactor_ == 0) {
    
    ESHandle<RPCGeometry> rpcGeo;
    iSetup.get<MuonGeometryRecord>().get(rpcGeo);
    
    MonitorElement * CLS;          // ClusterSize in 1 bin, Roll vs Sector
    MonitorElement * CLSD;         // ClusterSize in 1 bin, Distribution
    MonitorElement * MEAN;         // Mean ClusterSize, Roll vs Sector
    MonitorElement * MEAND;        // Mean ClusterSize, Distribution
    
    stringstream meName;
    //Loop on chambers
    for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
      if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
	RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
	std::vector< const RPCRoll*> roles = (ch->rolls());
	
	
	//Loop on rolls in given chamber
	for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	  RPCDetId detId = (*r)->id();
	  rpcdqm::utils prova;
	  
	  
	  int nr = prova.detId2RollNr(detId);
	  if(detId.region() !=0) continue;
	  
	  //Get Occupancy ME for roll
	  RPCGeomServ RPCname(detId);
	  
	  if (detId.region()==0){
	    
	    string YLabel = RPCname.shortname();
	    
	    RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
	    MonitorElement * myMe = dbe_->get("RPC/RecHits/"+ folderStr->folderStructure(detId)+"/ClusterSize_"+RPCname.name()); 
	    if (!myMe)continue;
	    
	    
	    float NormCLS = myMe->getBinContent(1)/myMe->getEntries(); // Normalization -> # of Entries in first Bin normalaized by total Entries
	    float meanCLS = myMe->getMean();
	    
	    meName.str("");
	    meName<<prefixDir_<<"ClusterSizeIn1Bin_Roll_vs_Sector_Wheel"<<detId.ring();
	    CLS = dbe_->get(meName.str());
	    if (CLS) {
	      CLS -> setBinContent(detId.sector(), nr, NormCLS);
	      CLS->setBinLabel(nr, YLabel, 2);
	    }
	    
	    
	    meName.str("");
	    meName<<prefixDir_<<"ClusterSizeIn1Bin_Distribution_Wheel"<<detId.ring();
	    CLSD = dbe_->get(meName.str());
	    CLSD->Fill(NormCLS);
	    	    	    
	  //   meName.str("");
// 	    meName<<prefixDir_<<"ClusterSizeMean_Roll_vs_Sector_Wheel"<<detId.ring();  
// 	    MEAN= dbe_->get(meName.str());
// 	    if(MEAN) {
// 	      MEAN -> setBinContent(detId.sector(), nr, meanCLS);
// 	      MEAN -> setBinLabel(nr, YLabel, 2);
// 	    }
	    
	    
	   //  meName.str("");
// 	    meName<<prefixDir_<<"ClusterSizeMean_Distribution_Wheel"<<detId.ring();
// 	    MEAND = dbe_->get(meName.str());
// 	    MEAND->Reset();
// 	    if (MEAND) {
// 	      for(int x=1; x<13; x++) {
// 		int roll;
// 		if(x==4) roll=22;
// 		else if(x==9 || x==11) roll=16;
// 		else roll=18;
// 		for(int y=1; y<roll; y++) {
// 		  MEAND->Fill( MEAN->getBinContent(x,y));
// 		  if (CLSD) CLSD->Fill(CLS->getBinContent(x,y));
// 		}
// 	      }
	      
// 	    }
	    
	    
	  }//End loop on Barrel
	} // end loop on rolls in given chamber
      }//End loop on chamber
      
      
    }
  }
}


