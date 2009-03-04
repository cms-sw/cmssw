/*  \author Anna Cimmino*/
#include <string>
#include <sstream>
#include <map>
#include <DQM/RPCMonitorClient/interface/RPCOccupancyTest.h>
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
RPCOccupancyTest::RPCOccupancyTest(const ParameterSet& ps ){
  LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: Constructor";
 
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 1);
  prefixDir_ = ps.getUntrackedParameter<string>("RPCPrefixDir", "RPC/RecHits");
  verbose_=ps.getUntrackedParameter<bool>("VerboseLevel", 0);

}

RPCOccupancyTest::~RPCOccupancyTest(){
  LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: Destructor ";
  dbe_=0;
}

void RPCOccupancyTest::beginJob(const EventSetup& iSetup){
 LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: Begin job ";
 dbe_ = Service<DQMStore>().operator->();
 dbe_->setVerbose(verbose_);
}

void RPCOccupancyTest::beginRun(const Run& r, const EventSetup& c){
 LogVerbatim ("rpceventsummary") << "[RPCOccupancyTest]: Begin run";
 
 
 MonitorElement* me;
 dbe_->setCurrentFolder(prefixDir_+"/SummaryHistograms");

 stringstream histoName;


 for(int w=-2; w<3;w++){

   histoName.str("");
   histoName<<"AsymmetryLeftRight_Roll_vs_Sector_Wheel"<<w;

 if ( me = dbe_->get(prefixDir_ +"/SummaryHistograms/"+ histoName.str()) ) {
    dbe_->removeElement(me->getName());
  }

  me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);

 for(int bin =1; bin<13;bin++) {
       histoName.str("");
       histoName<<"Sec"<<bin;
       me->setBinLabel(bin,histoName.str().c_str(),1);
     }

  histoName.str("");
  histoName<<"AsymmetryLeftRight_Distribution_Wheel"<<w;      
  if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
    dbe_->removeElement(me->getName());
  }
  me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, -0.1, 1.1);
  
  /////////////////

  histoName.str("");
  histoName<<"OccupancyNormByGeoAndRPCEvents_Wheel"<<w;

  if ( me = dbe_->get(prefixDir_ +"/SummaryHistograms/"+ histoName.str()) ) {
    dbe_->removeElement(me->getName());
  }
  
  me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
  
  for(int bin =1; bin<13;bin++) {
       histoName.str("");
       histoName<<"Sec"<<bin;
       me->setBinLabel(bin,histoName.str().c_str(),1);
  }
  
  histoName.str("");
  histoName<<"OccupancyNormByGeoAndRPCEvents_Distribution_Wheel"<<w;      
  if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
    dbe_->removeElement(me->getName());
  }
  me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.0, 0.205);


 
 }//end loop on wheels 
}

void RPCOccupancyTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCOccupancyTest::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCOccupancyTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCOccupancyTest]: End of LS transition, performing DQM client operation";

  // counts number of lumiSegs 
   nLumiSegs_ = lumiSeg.id().luminosityBlock();

  //check some statements and prescale Factor
  if(nLumiSegs_%prescaleFactor_ == 0) {
 
    ESHandle<RPCGeometry> rpcGeo;
    iSetup.get<MuonGeometryRecord>().get(rpcGeo);
    
    
    MonitorElement * myAsyMe;      //Left Right Asymetry 
    MonitorElement * NormOccup;
    MonitorElement * NormOccupDist;
    
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

	  MonitorElement * RPCEvents = dbe_->get(prefixDir_+"/SummaryHistograms/RPCEvents");  // 
	  int rpcevents = RPCEvents -> getEntries();                                          //  get RPC events
	  
	  if(detId.region() !=0) continue;
	  
	  //Get Occupancy ME for roll
	  RPCGeomServ RPCname(detId);
	  //	 string Yaxis=RPCname.name();
	  if (detId.region()==0){
	    
	    string YLabel = RPCname.shortname();
	    
	    RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
	    MonitorElement * myMe = dbe_->get(prefixDir_+"/"+ folderStr->folderStructure(detId)+"/Occupancy_"+RPCname.name()); 
	    if (!myMe)continue;
	    int stripInRoll=(*r)->nstrips();
	    float FOccupancy=0;
	    float BOccupancy=0;
	    
	    float  totEnt =  myMe->getEntries();
	    for(int strip = 1 ; strip<=stripInRoll; strip++){
	      if(strip<=stripInRoll/2) FOccupancy+=myMe->getBinContent(strip);
	      else  BOccupancy+=myMe->getBinContent(strip);
	    }
	    
	    float asym =  fabs((FOccupancy - BOccupancy )/totEnt);
	    
	    meName.str("");
	    meName<<prefixDir_<<"/SummaryHistograms/AsymmetryLeftRight_Roll_vs_Sector_Wheel"<<detId.ring();
	    myAsyMe= dbe_->get(meName.str());
	    if(myAsyMe){
	      
	      myAsyMe->setBinContent(detId.sector(),nr,asym );
	      myAsyMe->setBinLabel(nr, YLabel, 2);
	    }
	    
	    meName.str("");
	    meName<<prefixDir_<<"/SummaryHistograms/AsymmetryLeftRight_Distribution_Wheel"<<detId.ring();
	    myMe= dbe_->get(meName.str());
	    if(myMe) myMe->Fill(asym);


	    meName.str("");
	    meName<<prefixDir_<<"/SummaryHistograms/OccupancyNormByGeoAndRPCEvents_Wheel"<<detId.ring();
	    NormOccup= dbe_->get(meName.str());

	    meName.str("");
	    meName<<prefixDir_<<"/SummaryHistograms/OccupancyNormByGeoAndRPCEvents_Distribution_Wheel"<<detId.ring();      
	    NormOccupDist = dbe_->get(meName.str());
	    if(myAsyMe){
	      float normoccup = totEnt/(stripInRoll*rpcevents)*10;
	      NormOccup->setBinContent(detId.sector(),nr, normoccup);
	      NormOccup->setBinLabel(nr, YLabel, 2);
	      // if(normoccup>0.1) normoccup=0.1; //overflow
	      NormOccupDist->Fill(normoccup);
	    }


	  }//End loop on rolls in given chambers
	}
      }//End loop on chamber
    }
  }
}


