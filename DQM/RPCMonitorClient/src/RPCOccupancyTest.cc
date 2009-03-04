/*  \author Anna Cimmino*/
#include <string>
#include <sstream>
//#include <ostream>
#include <map>
#include <DQM/RPCMonitorClient/interface/RPCOccupancyTest.h>
//#include "DQM/RPCMonitorDigi/interface/RPCBookFolderStructure.h"
//#include "DQM/RPCMonitorDigi/interface/utils.h"
#include <DQM/RPCMonitorClient/interface/clientTools.h>

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
<<<<<<< RPCOccupancyTest.cc
   histoName<<"ChipOccupancy_Distribution_Wheel"<<w;      
   if ( me = dbe_->get(prefixDir_ +"/"+ histoName.str()) ) {
     dbe_->removeElement(me->getName());
   }
   me = dbe_->book1D(histoName.str().c_str(), histoName.str().c_str(),  20, -0.1, 1.1);
 }

=======
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

>>>>>>> 1.3

<<<<<<< RPCOccupancyTest.cc
 //Get NumberOfDigi ME for each roll

 rpcdqmclient::clientTools tool;
 myOccupancyMe_ = tool.constructMEVector(c, prefixDir_, "NumberOfDigi", dbe_);
 myDetIds_ = tool.getAssociatedRPCdetId();

=======
 
 }//end loop on wheels 
>>>>>>> 1.3
}

void RPCOccupancyTest::beginLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& context){} 

void RPCOccupancyTest::analyze(const Event& iEvent, const EventSetup& c) {}

void RPCOccupancyTest::endLuminosityBlock(LuminosityBlock const& lumiSeg, EventSetup const& iSetup) {  
  LogVerbatim ("rpceventsummary") <<"[RPCOccupancyTest]: End of LS transition, performing DQM client operation";
  
  // counts number of lumiSegs 
  nLumiSegs_ = lumiSeg.id().luminosityBlock();
  
  //check some statements and prescale Factor
  if(nLumiSegs_%prescaleFactor_ == 0) {
<<<<<<< RPCOccupancyTest.cc
    
    ESHandle<RPCGeometry> rpcgeo;
    iSetup.get<MuonGeometryRecord>().get(rpcgeo);
    
    MonitorElement * myChipDist[5];
    stringstream meName;
    for(int wheel = -2; wheel<3; wheel++){
      meName.str("");
      meName<<prefixDir_<<"/SummaryHistograms/ChipOccupancy_Distribution_Wheel"<<wheel;
      myChipDist[wheel+2]= dbe_->get(meName.str());
      if(!myChipDist)continue;
      myChipDist[wheel+2]->Reset();
    }
    MonitorElement * myMe;
    RPCDetId detId;
    
    
    //Loop on Occupancy MEs
    for (unsigned int  i = 0 ; i<myOccupancyMe_.size();i++){
      
      detId =  myDetIds_[i];
      myMe = myOccupancyMe_[i];
      
      
      RPCGeomServ RPCserv(detId);
      
      if (detId.region()==0){
	//    string YLabel = RPCname.shortname();
	
	vector<int> numChip = RPCserv.channelInChip();
	int totalChanPerFEB=0;
	
	for (unsigned int j =0 ; j < numChip.size(); j++){
	  totalChanPerFEB += numChip[j];
	}
	
	
	const RPCRoll * rpcRoll = rpcgeo->roll(detId);      
 	unsigned int nstrips =rpcRoll->nstrips();
	int totalNumberFEB =0;
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
	
	for(int o = 0; o<range.size(); o++){
	  cout<<range[o]<<endl;
	  
	}
	cout<<endl;
	
	int index = 0;
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
	c=0;
	for(unsigned int j = 0; j<entriesPerChip.size(); j++){
	  if(c>=numChip.size()) c=0;
	  if(total !=0)
	    myChipDist[detId.ring()+2]->Fill((( (entriesPerChip[j] / numChip[c]) - (total/nstrips) )/(total/nstrips)));
	  c++;
	}
	
      }//End Barrel
    }//End loop on Occupancy ME
    
    this->OccupancyDist();
  }
}

void RPCOccupancyTest::OccupancyDist(){

  stringstream meName;
  MonitorElement * myMe;
  MonitorElement * myMe1;
  
  dbe_->setCurrentFolder(prefixDir_+"/SummaryHistograms");
  
  for(int ring =-2; ring<=2; ring++){
=======
 
    ESHandle<RPCGeometry> rpcGeo;
    iSetup.get<MuonGeometryRecord>().get(rpcGeo);
>>>>>>> 1.3
    
    meName.str("");
    meName<<prefixDir_<<"/SummaryHistograms/OccupancyNormByGeoAndEvents_Roll_vs_Sector_Wheel"<<"_"<<ring;
    myMe= dbe_->get(meName.str());
    if(!myMe) continue;
    
<<<<<<< RPCOccupancyTest.cc
    meName.str("");
    meName<<"OccupancyNormByGeoAndEvents_Distribution_Wheel"<<ring;
    
    myMe1 = dbe_->get(prefixDir_ +"/SummaryHistograms/"+ meName.str());
      
      if (myMe1)  myMe1->Reset();
      else    myMe1= dbe_->book1D(meName.str().c_str(),meName.str().c_str() ,  501, -0.0005, 0.5005);
    
    
    for(int x =1; x<= myMe->getNbinsX(); x++){
      for(int y =1; y<= myMe->getNbinsY(); y++){
	if (y<16 || x==4 || (y<18 && x!=9 && x!=11) )
	  myMe1->Fill(myMe->getBinContent(x,y));
	
      }
      
    }
=======
    MonitorElement * myAsyMe;      //Left Right Asymetry 
    MonitorElement * NormOccup;
    MonitorElement * NormOccupDist;
>>>>>>> 1.3
    
<<<<<<< RPCOccupancyTest.cc
  }
=======
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
>>>>>>> 1.3
}
 
