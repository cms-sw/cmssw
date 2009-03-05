/*  \author Anna Cimmino*/
#include <string>
#include <sstream>
//#include <ostream>
#include <map>
#include <DQM/RPCMonitorClient/interface/RPCOccupancyChipTest.h>
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
RPCOccupancyChipTest::RPCOccupancyChipTest(const ParameterSet& ps ){
  LogVerbatim ("rpceventsummary") << "[RPCOccupancyChipTest]: Constructor";
 
  prescaleFactor_ =  ps.getUntrackedParameter<int>("PrescaleFactor", 1);
  prefixDir_ = ps.getUntrackedParameter<string>("RPCPrefixDir", "RPC/RecHits");
  verbose_=ps.getUntrackedParameter<bool>("VerboseLevel", 0);

}

RPCOccupancyChipTest::~RPCOccupancyChipTest(){
  LogVerbatim ("rpceventsummary") << "[RPCOccupancyChipTest]: Destructor ";
  dbe_=0;
}

void RPCOccupancyChipTest::beginJob(const EventSetup& iSetup){
 LogVerbatim ("rpceventsummary") << "[RPCOccupancyChipTest]: Begin job ";
 dbe_ = Service<DQMStore>().operator->();
 dbe_->setVerbose(verbose_);
}

void RPCOccupancyChipTest::beginRun(const Run& r, const EventSetup& c){
 LogVerbatim ("rpceventsummary") << "[RPCOccupancyChipTest]: Begin run";
 
 
 MonitorElement* me;
 dbe_->setCurrentFolder(prefixDir_+"/SummaryHistograms");

 stringstream histoName;


 for(int w=-2; w<3;w++){

   histoName.str("");
   histoName<<"ChipOccupancy_Distribution_Wheel"<<w;      
   if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }
   me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  10001, -0.5, 10000.5);


   histoName.str("");
   histoName<<"ChipsOccupancy_Roll_vs_Sector_Wheel"<<w;
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

void RPCOccupancyChipTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCOccupancyChipTest::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCOccupancyChipTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCOccupancyChipTest]: End of LS transition, performing DQM client operation";
 
  // counts number of lumiSegs 
  nLumiSegs_ = lumiSeg.id().luminosityBlock();
  
  //check some statements and prescale Factor
  if(nLumiSegs_%prescaleFactor_ == 0) {
    
    ESHandle<RPCGeometry> rpcgeo;
    iSetup.get<MuonGeometryRecord>().get(rpcgeo);
 
    MonitorElement * myChipDist[5];
    stringstream meName;
    for(int wheel = -2; wheel<3; wheel++){
      meName.str("");
      meName<<prefixDir_<<"/SummaryHistograms/ChipOccupancy_Distribution_Wheel"<<wheel;
      myChipDist[wheel+2]= dbe_->get(meName.str());
      if(!myChipDist[wheel+2])continue;
      myChipDist[wheel+2]->Reset();
    }
    MonitorElement * myMe;
    RPCDetId detId;
    int ui=0;
 
    //Loop on Occupancy MEs
    for (unsigned int  i = 0 ; i<myOccupancyMe_.size();i++){
      
      detId =  myDetIds_[i];
      myMe = myOccupancyMe_[i];
     
      RPCGeomServ RPCserv(detId);
      
      if (detId.region()==0){
	//    string YLabel = RPCname.shortname();
	ui++;
	vector<int> numChip = RPCserv.channelInChip();
	int totalChanPerFEB=0;
	
	for (unsigned int j =0 ; j < numChip.size(); j++){
	  totalChanPerFEB += numChip[j];
	}
	
	const RPCRoll * rpcRoll = rpcgeo->roll(detId);      
 	unsigned int nstrips =rpcRoll->nstrips();
	int totalNumberFEB =0;
	
	RPCGeomServ RPCname(detId);
	string YLabel = RPCname.shortname();

	if (nstrips % totalChanPerFEB ==0)
	  totalNumberFEB = nstrips/totalChanPerFEB; 
	else continue;
	
	vector<int> range;
	range.push_back(numChip[0]);
	unsigned int c= 1;
	while (range.size()<(unsigned int)(2*totalNumberFEB)){
	  if(c>=numChip.size()) c=0;
	  range.push_back(range.back() + numChip[c]);
	  c++;
	}
	;
	vector<float> entriesPerChip(range.size(),0);
	
	for(int xbin =1 ; xbin <= myMe->getNbinsX() ; xbin++){	  
	  unsigned int b = 0;
	  float binContent = myMe->getBinContent(xbin);
	  while(b<range.size() &&  binContent>range[b]){
	    b++;
	  }
	  entriesPerChip[b]+=  myMe->getBinContent(xbin);
	}
	
	float total = myMe->getEntries();
	float totalDiff=0;
	c=0;
	for(unsigned int j = 0; j<entriesPerChip.size(); j++){
	  if(c>=numChip.size()) c=0;
	    totalDiff +=  fabs((entriesPerChip[j] / numChip[c]) - (total/nstrips) );
	    c++;
	}


	rpcdqm::utils rollNumber;
	int nr = rollNumber.detId2RollNr(detId);
	meName.str("");
	meName<<prefixDir_<<"/SummaryHistograms/ChipsOccupancy_Roll_vs_Sector_Wheel"<<detId.ring();
	myMe =dbe_->get(meName.str()); 
	

	if(myMe) {
	  myMe->setBinContent(detId.sector(),nr,totalDiff); 
	
	
	  myMe->setBinLabel(nr, YLabel, 2);

	}
	  if (totalDiff > 10000)totalDiff=10000;	
	  myChipDist[detId.ring()+2]->Fill(totalDiff);
	
      }//End Barrel
    }//End loop on Occupancy MEmyChipDist[detId.ring()+2]->Fill( v);
  }
}

