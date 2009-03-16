/**************************************
 *         Autor: David Lomidze       *
 *           INFN di Napoli           *
 *           06 March 2009            *
 *************************************/

#include <string>
#include <sstream>
//#include <ostream>
#include <map>
#include <DQM/RPCMonitorClient/interface/RPCNoisyStripTest.h>
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

RPCNoisyStripTest::RPCNoisyStripTest(const ParameterSet& ps ){
  LogVerbatim ("rpceventsummary") << "[RPCNoisyStripTest]: Constructor";
 
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 1);
  prefixDir_ = ps.getUntrackedParameter<string>("RPCPrefixDir", "RPC/RecHits");
  verbose_=ps.getUntrackedParameter<bool>("VerboseLevel", 0);

}

RPCNoisyStripTest::~RPCNoisyStripTest(){
  LogVerbatim ("rpceventsummary") << "[RPCNoisyStripTest]: Destructor ";
  dbe_=0;
}

void RPCNoisyStripTest::beginJob(const EventSetup& iSetup){
 LogVerbatim ("rpceventsummary") << "[RPCNoisyStripTest]: Begin job ";
 dbe_ = Service<DQMStore>().operator->();
 dbe_->setVerbose(verbose_);
}

void RPCNoisyStripTest::beginRun(const Run& r, const EventSetup& c){
 LogVerbatim ("rpceventsummary") << "[RPCNoisyStripTest]: Begin run";
 
 
 MonitorElement* me;
 dbe_->setCurrentFolder(prefixDir_+"/SummaryHistograms");

 stringstream histoName;


 for(int w=-2; w<3;w++){

   histoName.str("");
   histoName<<"RPCNoisyStrips_Distribution_Wheel"<<w;      
   if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }
   me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  6, -0.5, 5.5);


   histoName.str("");
   histoName<<"RPCStripsDeviation_Distribution_Wheel"<<w;      
   if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }
   me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  101, -0.01, 10.01);

   histoName.str("");
   histoName<<"RPCNoisyStrips_Roll_vs_Sector_Wheel"<<w;
   if ( me = dbe_->get(prefixDir_+"/SummaryHistograms/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }
   
   me = dbe_->book2D(histoName.str().c_str(), histoName.str().c_str() , 12, 0.5, 12.5, 21, 0.5, 21.5);
   
   for(int bin =1; bin<13;bin++) {
     histoName.str("");
     histoName<<"Sec"<<bin;
     me->setBinLabel(bin,histoName.str().c_str(),1);
   }
 }

 //Get NumberOfDigi ME for each roll

ESHandle<RPCGeometry> rpcGeo;
 c.get<MuonGeometryRecord>().get(rpcGeo);

//loop on all geometry and get all histos
 for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
   if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
     RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
     std::vector< const RPCRoll*> roles = (ch->rolls());
     //Loop on rolls in given chamber
     for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
       RPCDetId detId = (*r)->id();
      
       //Get Occupancy ME for roll
       RPCGeomServ RPCname(detId);	   
       
       RPCBookFolderStructure *  folderStr = new RPCBookFolderStructure();
       MonitorElement * myMe = dbe_->get(prefixDir_+"/"+ folderStr->folderStructure(detId)+"/Occupancy_"+RPCname.name()); 
       if (!myMe)continue;

       myOccupancyMe_.push_back(myMe);
	
       myDetIds_.push_back(detId);
       //   myRollNames_.push_back(RPCname.name());
     }
   }
 }//end loop on all geometry and get all histos
}

void RPCNoisyStripTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCNoisyStripTest::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCNoisyStripTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCNoisyStripTest]: End of LS transition, performing DQM client operation";
 
  // counts number of lumiSegs 
  nLumiSegs_ = lumiSeg.id().luminosityBlock();
  
  //check some statements and prescale Factor
  if(nLumiSegs_%prescaleFactor_ == 0) {
    
    ESHandle<RPCGeometry> rpcgeo;
    iSetup.get<MuonGeometryRecord>().get(rpcgeo);
 
    stringstream meName;
    
    MonitorElement * myMe;
    MonitorElement * DTdist;
    MonitorElement * StripDev;

    DTdist = dbe_ -> get(meName.str());
    RPCDetId detId;
    
    //Loop on Occupancy MEs
    for (unsigned int  i = 0 ; i<myOccupancyMe_.size();i++){
      
      detId =  myDetIds_[i];
      myMe = myOccupancyMe_[i];
      
      RPCGeomServ RPCserv(detId);
      
      if (detId.region()==0){
		
	const RPCRoll * rpcRoll = rpcgeo->roll(detId);      
 	unsigned int nstrips =rpcRoll->nstrips();
	
	
	RPCGeomServ RPCname(detId);
	string YLabel = RPCname.shortname();
	
	
	int entries = myMe -> getEntries();
	int bins = myMe ->getNbinsX();
		
	vector<float> myvector;
	
	// count alive strips and alive strip values put in the vector
	for(int xbin =1 ; xbin <= bins ; xbin++) {	  
	  
	  int binContent = myMe->getBinContent(xbin);
	  
	  if (binContent > 0) myvector.push_back(binContent);
	  
	}
	
	
	meName.str("");
	meName<<prefixDir_<<"/SummaryHistograms/RPCNoisyStrips_Distribution_Wheel"<<detId.ring();
	DTdist = dbe_ -> get(meName.str());
	  
	meName.str("");
	meName<<prefixDir_<<"/SummaryHistograms/RPCStripsDeviation_Distribution_Wheel"<<detId.ring();      
	StripDev = dbe_ -> get(meName.str());
	
	
	int noisyStrips=0;
	// calculate mean on YAxis and check diff between bins and mean
	if (myvector.size()>0) {
	  float ymean = entries/myvector.size(); //mean on Yaxis
	  for(int i=0; i<myvector.size(); i++) {
	    float deviation = myvector[i]/ymean;
	    if(deviation > 3.5)  noisyStrips++;
	    if(deviation > 5) deviation = 5; //overflow 
	      StripDev -> Fill(deviation);
	  }
	
	  rpcdqm::utils rollNumber;
	  int nr = rollNumber.detId2RollNr(detId);
	  meName.str("");
	  meName<<prefixDir_<<"/SummaryHistograms/RPCNoisyStrips_Roll_vs_Sector_Wheel"<<detId.ring();
	  myMe =dbe_->get(meName.str()); 
	  
	  if(myMe) {
	    //cout<<detId.sector()<<"  "<<nr<<"  "<<noisyStrips<<endl;
	    myMe->setBinContent(detId.sector(),nr,noisyStrips); 
	    myMe->setBinLabel(nr, YLabel, 2);
	    DTdist ->Fill(noisyStrips);
	  }
	
	  

  
	} // End of if()
	
      }//End Barrel
    }//End loop on Occupancy MEmhipDist[detId.ring()+2]->Fill( v);
  }
}

