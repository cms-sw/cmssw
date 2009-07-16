#include <sstream>
#include <iomanip>
#include <DQM/RPCMonitorClient/interface/RPCMon_SS_Dbx_Global.h>
// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//DQM Services
#include "DQMServices/Core/interface/MonitorElement.h"
//DataFormats
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
//Geometry
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace edm;
using namespace std;


RPCMon_SS_Dbx_Global::RPCMon_SS_Dbx_Global(const ParameterSet& iConfig ){


  LogVerbatim ("rpcmonitorerror") << "[RPCMon_SS_Dbx_Global]: Constructor";


  globalFolder_ = iConfig.getUntrackedParameter<string>("GlobalHistogramsFolder","RPC/RecHits/SummaryHistograms");

  saveRootFile_ =  iConfig.getUntrackedParameter<bool>("SaveRootFile", false);
  rootFileName_ = iConfig.getUntrackedParameter<string>("RootFileName","out.root");
  digiLabel_ = iConfig.getUntrackedParameter<std::string>("DigiLabel","muonRPCDigis");
}

RPCMon_SS_Dbx_Global::~RPCMon_SS_Dbx_Global(){
  LogVerbatim ("rpcmonitorerror") << "[RPCMon_SS_Dbx_Global]: Destructor ";
  dbe_=0;
}

void RPCMon_SS_Dbx_Global::beginJob(const EventSetup& iSetup){
 LogVerbatim ("rpcmonitorerror") << "[RPCMon_SS_Dbx_Global]: Begin job ";
 dbe_ = Service<DQMStore>().operator->();
}

void RPCMon_SS_Dbx_Global::endJob(){}

void RPCMon_SS_Dbx_Global::beginRun(const Run& r, const EventSetup& c){
  LogVerbatim ("rpcmonitorerror") << "[RPCMon_SS_Dbx_Global]: Begin run";

  dbe_->setCurrentFolder(globalFolder_);

  MonitorElement* me;
  me =0;
  me = dbe_->get(globalFolder_ + "/AfterPulseBxDiff");
  if ( 0!=me) 
    dbe_->removeElement(me->getName());
  me = dbe_->book1D("AfterPulseBxDiff","After Pulse Bx Difference",13,-6.5,6.5);
}

void RPCMon_SS_Dbx_Global::analyze(const Event& iEvent, const EventSetup&  iSetup) {

 edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByType(rpcDigis);

 edm::ESHandle<RPCGeometry> pDD;
 iSetup.get<MuonGeometryRecord>().get( pDD );
 
 
 RPCDigiCollection::DigiRangeIterator detUnitIt;
 
 dbe_->setCurrentFolder(globalFolder_);

 // Loop over DetUnit's
 for (detUnitIt=rpcDigis->begin();detUnitIt!=rpcDigis->end();++detUnitIt){      
   const RPCDetId& id = (*detUnitIt).first;
   const RPCRoll* roll = dynamic_cast<const RPCRoll* >( pDD->roll(id));
   const RPCDigiCollection::Range& range = (*detUnitIt).second;
   //     std::cout <<" detector "<< id.region()<< std::endl;
   ostringstream tag;
   ostringstream name;	
   MonitorElement* me;

   //get roll name
   RPCGeomServ RPCname(id);
   string nameRoll = RPCname.name();
   //get roll number      
   RPCGeomServ RPCnumber(id);
   int nr = RPCnumber.chambernr();
   //get segment (valid only if endcap)
   RPCGeomServ RPCsegment(id);
   int seg = RPCsegment.segment();
   
   // Loop over the digis of this DetUnit 
   for (RPCDigiCollection::const_iterator digiIt = range.first;digiIt!=range.second;++digiIt) {
     if (digiIt->strip() < 1 || digiIt->strip() > roll->nstrips() )
       LogVerbatim ("rpcmonitorerror") <<" XXXX  Problem with detector "<< id << ", strip # " << digiIt->strip();
     else {  
       for (RPCDigiCollection::const_iterator digiIt2 = digiIt; digiIt2!=range.second; ++digiIt2) {
	 int dstrip = digiIt->strip() - digiIt2->strip();
	 int dbx = digiIt->bx() - digiIt2->bx();
	 if ( dstrip == 0 && abs(dbx) != 0 ) {
	   //	   std::cout <<" detector 3333 "<< id.region()<<" " << id.ring()<<std::endl;
	   me = dbe_->get(globalFolder_ + "/AfterPulseBxDiff");
	   if(!me) continue;
	   me->Fill(dbx);

	   if (id.region()!=0){//Endcap
	     name.str("");
	     tag.str("");
	     tag << "AfterPulse_DiffBx_Disk#" << id.region()*id.station();
	     name << "Endcap, After Pulse, Diff bx, Disk # " << id.region()*id.station();
	     me = dbe_->get(globalFolder_ + "/"+tag.str());
	     if (!me){
	       me = dbe_->book2D (tag.str(),name.str(),36,1.0,36.0,9,1.0,9.0);
	   
	       for (int chamber=1; chamber<=36 ; chamber++){
		 name.str("");
		 name<<"Ch#"<< setw(2) << setfill('0') <<chamber;
		 me->setBinLabel(chamber, name.str(), 1); 
	       }
	       for (int ring=1; ring<=3 ; ring++){
		 for(int roll =1; roll<=3; roll++){
		   string myRoll;
		   if(roll==1) myRoll="A";
		   else if(roll==2) myRoll="B";
		   else if(roll==3) myRoll="C";
		   else if(roll==3) myRoll="D";
		   name.str("");
		   name<<"Ring"<<ring<<"_"<<myRoll;
		   me->setBinLabel((ring-1)*3+roll, name.str(), 2); 
		 }
	       }
	     }
	     me->Fill(seg, (id.ring()-1)*3+id.roll()); 
	   } else { // Barrel ( region == 0 ).
	
	     name.str("");
	     tag.str("");
	     tag << "AfterPulse_Wheel#" << std::setw(2) << std::setfill('+') << id.ring();
	     name << "Barrel, After Pulse, Diff. bx, Wheel # " << std::setw(2) << std::setfill('+') << id.ring();
	     me = dbe_->get(globalFolder_ + "/"+tag.str());
	     if (!me){
	       me = dbe_->book2D (tag.str(),name.str(),12,0.5,13.5,21,1.0,21.0);
	       for (int sector=1; sector<=12 ; sector++){
		 name.str("");
		 name<<"Sec"<< setw(2) << setfill('0') <<sector;
		 me->setBinLabel(sector, name.str(), 1); 
	       }
	     } 
	     me->Fill(id.sector(),nr);
	     string YLabel = RPCname.shortname();
	     me->setBinLabel(nr, YLabel, 2);
	   }
	 }
       }
     }
   }	    
 }// for  digis in layer
} 
              




