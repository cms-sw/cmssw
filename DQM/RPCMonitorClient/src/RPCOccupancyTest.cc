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
   histoName<<"AsymmetryFB_Roll_vs_Sector_Wheel"<<w;

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
  histoName<<"3Occupancy_Entries_Roll_vs_Sector_Wheel"<<w;
  if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
    dbe_->removeElement(me->getName());
  }

  me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  36, 0.5, 36.5, 21, 0.5, 21.5);

  int s=1;

 for(int bin =2; bin<36;bin+=3) {
       histoName.str("");
       histoName<<"Sec"<<s;
       me->setBinLabel(bin,histoName.str().c_str(),1);
       s++;    
 }

 ////////// new
 histoName.str("");
 histoName<<"Asymmetry_Roll_vs_Sector_Wheel"<<w;       // new asymmetry 2D histo       
  if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
    dbe_->removeElement(me->getName());
  }
  me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
  for(int bin =1; bin<13;bin++) {
    histoName.str("");
    histoName<<"Sec"<<s;
    me->setBinLabel(bin,histoName.str().c_str(),1);
    s++;    
  }
  

  histoName.str("");
  histoName<<"Asymmetry_Distribution_Wheel"<<w;       // new asymmetry distribution    
  if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
    dbe_->removeElement(me->getName());
  }
  me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.5, 100.5);
  
  histoName.str("");
  histoName<<"RMS_Roll_vs_Sector_Wheel"<<w;       // Occupancy RMS 
  if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
    dbe_->removeElement(me->getName());
  }
  me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str(),  12, 0.5, 12.5, 21, 0.5, 21.5);
  for(int bin =1; bin<13;bin++) {
    histoName.str("");
    histoName<<"Sec"<<s;
    me->setBinLabel(bin,histoName.str().c_str(),1);
    s++;    
  }


  histoName.str("");
  histoName<<"RMS_Distribution_Wheel"<<w;       // Occupancy RMS 
  if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
    dbe_->removeElement(me->getName());
  }
  me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  100, 0.5, 100.5);
 

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
 

  MonitorElement * myAsyMe;
  MonitorElement * myGlobalMe;
  MonitorElement * MVA;          // Mean Value Asymmetry
  MonitorElement * RMS;          // Occupancy RMS
  MonitorElement * RMSD;         // Occupancy RMS Distribution
  MonitorElement * MVAD;         // Mean Value Asymmetry Distibution
  
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
	 //	 string Yaxis=RPCname.name();
	 if (detId.region()==0){

	  string YLabel = RPCname.shortname();

	 RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
	 MonitorElement * myMe = dbe_->get(prefixDir_+"/"+ folderStr->folderStructure(detId)+"/Occupancy_"+RPCname.name()); 
	 if (!myMe)continue;
	 int stripInRoll=(*r)->nstrips();
	 float FOccupancy=0;
	 float BOccupancy=0;

	 float FirstOccupancy=0;
	 float  SecondOccupancy=0;
	 float  ThirdOccupancy=0;


	 float  totEnt =  myMe->getEntries();
	 for(int strip = 1 ; strip<=stripInRoll; strip++){
	   if(strip<=stripInRoll/2) FOccupancy+=myMe->getBinContent(strip);
	   else  BOccupancy+=myMe->getBinContent(strip);

	   if(strip<=stripInRoll/3)FirstOccupancy+=myMe->getBinContent(strip);
	   else if(strip<=(2*stripInRoll)/3) SecondOccupancy+=myMe->getBinContent(strip);
	   else ThirdOccupancy+=myMe->getBinContent(strip);
	 }

	 float asym =  (FOccupancy - BOccupancy )/totEnt;

	 meName.str("");
	 meName<<prefixDir_<<"/SummaryHistograms/AsymmetryFB_Roll_vs_Sector_Wheel"<<detId.ring();
	 myAsyMe= dbe_->get(meName.str());
	 if(myAsyMe){

	   myAsyMe->setBinContent(detId.sector(),nr,asym );
	   myAsyMe->setBinLabel(nr, YLabel, 2);
	 }

	 meName.str("");
	 meName<<prefixDir_<<"/SummaryHistograms/3Occupancy_Entries_Roll_vs_Sector_Wheel"<<detId.ring();
	 myGlobalMe= dbe_->get(meName.str());
	 if(myGlobalMe){


	 int xBin=(detId.sector()-1)*3;
	 myGlobalMe->setBinContent(xBin+1,nr,FirstOccupancy/totEnt);
	 myGlobalMe->setBinContent(xBin+2,nr,SecondOccupancy/totEnt);
	 myGlobalMe->setBinContent(xBin+3,nr,ThirdOccupancy/totEnt);
	 myGlobalMe->setBinLabel(nr, YLabel, 2);
	 }

	 ///////////// David //////
	 float BinsOnX = myMe->getNbinsX();
	 float Mean = myMe->getMean();
	 float Asymmetry = fabs(BinsOnX-Mean/2);
	 float rms = myMe->getRMS();
	 
	 meName.str("");
	 meName<<prefixDir_<<"/SummaryHistograms/RMS_Roll_vs_Sector_Wheel"<<detId.ring();
	 RMS = dbe_->get(meName.str());
	 if (RMS) { 
	   RMS -> setBinContent(detId.sector(), nr, rms);
	   RMS ->setBinLabel(nr, YLabel, 2);
	 }
	 
	 
	 meName.str("");
	 meName<<prefixDir_<<"/SummaryHistograms/RMS_Distribution_Wheel"<<detId.ring();
	 RMSD = dbe_->get(meName.str());
	 RMSD->Reset();
	 	 

	 meName.str("");
	 meName<<prefixDir_<<"/SummaryHistograms/Asymmetry_Roll_vs_Sector_Wheel"<<detId.ring();  
	 MVA= dbe_->get(meName.str());
	 if(MVA) {
	   MVA -> setBinContent(detId.sector(), nr, Asymmetry);
	   MVA -> setBinLabel(nr, YLabel, 2);
	 }
	 
	 meName.str("");
	 meName<<prefixDir_<<"/SummaryHistograms/Asymmetry_Distribution_Wheel"<<detId.ring();
	 // cout<<meName.str()<<endl;
	 MVAD = dbe_->get(meName.str());
	 MVAD->Reset();
	 if (MVAD) {
	   for(int x=1; x<13; x++) {
	     int roll;
	     if(x==4) roll=22;
	     else if(x==9 || x==11) roll=16;
	     else roll=18;
	     for(int y=1; y<roll; y++) {
	       MVAD->Fill( MVA->getBinContent(x,y));
	       if (RMSD) RMSD->Fill(RMS->getBinContent(x,y));
	     }
	   }
	 }
	 
	 
       }//End loop on rolls in given chambers
    }
    }//End loop on chamber
  
    
    
    
  }
  }
}


