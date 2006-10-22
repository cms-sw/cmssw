/** \file
 *
 *  implementation of RPCMonitorDigi class
 *
 *  $Date: 2006/10/14 14:31:49 $
 *  $Revision: 1.14 $
 *
 * \author Ilaria Segoni
 */

#include <map>
#include <string>
#include <TRandom.h> 

#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>

///Data Format
#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>

///RPCRecHits
#include <DataFormats/RPCRecHit/interface/RPCRecHitCollection.h>
#include <Geometry/Surface/interface/LocalError.h>
#include <Geometry/Vector/interface/LocalPoint.h>

///Geometry
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>


///Log messages
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>


RPCMonitorDigi::RPCMonitorDigi( const edm::ParameterSet& pset ):counter(0){

  foundHitsInChamber.clear();
  nameInLog = pset.getUntrackedParameter<std::string>("moduleLogName", "RPC_DQM");

  saveRootFile  = pset.getUntrackedParameter<bool>("DigiDQMSaveRootFile", false); 
  saveRootFileEventsInterval  = pset.getUntrackedParameter<int>("DigiEventsInterval", 10000); 
  RootFileName  = pset.getUntrackedParameter<std::string>("RootFileNameDigi", "RPCMonitor.root"); 

  
  /// get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();


}


RPCMonitorDigi::~RPCMonitorDigi(){
}

void RPCMonitorDigi::beginJob(edm::EventSetup const&){
 edm::LogInfo (nameInLog) <<"Beginning DQMMonitorDigi " ;
  
  GlobalHistogramsFolder="RPC/RecHits/GlobalHistograms";
  dbe->setCurrentFolder(GlobalHistogramsFolder);  

  GlobalZYHitCoordinates = dbe->book2D("GlobalRecHitZYCoordinates", "Rec Hit Z-Y", 1000, -800, 800, 1000, -800, 800);
  GlobalZXHitCoordinates = dbe->book2D("GlobalRecHitZXCoordinates", "Rec Hit Z-X", 1000, -800, 800, 1000, -800, 800);
  GlobalZPhiHitCoordinates = dbe->book2D("GlobalRecHitZPhiCoordinates", "Rec Hit Z-Phi", 1000, -800, 800, 1000, -4, 4);


  dbe->showDirStructure();
}

void RPCMonitorDigi::endJob(void)
{
  if(saveRootFile) dbe->save(RootFileName);
 
  //std::vector<std::string> contentVec;
  //dbe->getContents(contentVec);
  //std::vector<std::string>::iterator dirItr;
  //for(dirItr=contentVec.begin();dirItr!=contentVec.end(); ++dirItr){
  //	dbe->setCurrentFolder(*dirItr);
  //	dbe->removeContents();
  //}
  
  dbe = 0;
}


void RPCMonitorDigi::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup ){
 counter++;
 edm::LogInfo (nameInLog) <<"Beginning analyzing event " << counter;

 char layerLabel[128];
 char meId [128];

 
 std::map<uint32_t, bool >::iterator mapItrReset;
 for (mapItrReset = foundHitsInChamber.begin(); mapItrReset != foundHitsInChamber.end(); ++ mapItrReset) {
 	mapItrReset->second=false;
 }
 
 

/// RPC Geometry
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

/// DIGI     
 edm::Handle<RPCDigiCollection> rpcdigis;
 iEvent.getByType(rpcdigis);

/// RecHits
 edm::Handle<RPCRecHitCollection> rpcHits;
 iEvent.getByType(rpcHits);


 RPCDigiCollection::DigiRangeIterator collectionItr;
 for(collectionItr=rpcdigis->begin(); collectionItr!=rpcdigis->end(); ++collectionItr){

 RPCDetId detId=(*collectionItr ).first; 
 uint32_t id=detId(); 

 const GeomDet* gdet=rpcGeo->idToDet(detId);
 const BoundPlane & surface = gdet->surface();
 
 char detUnitLabel[128];
 sprintf(detUnitLabel ,"%d",detId());
 sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detId.layer(),detId.subsector(),detId.roll());

 
 std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
 if (meItr == meCollection.end() || (meCollection.size()==0)) {
 	meCollection[id]=bookDetUnitME(detId);
 }
 std::map<std::string, MonitorElement*> meMap=meCollection[id];
 
 int region=detId.region();
 int ring=detId.ring();
 
 std::pair<int,int> regionRing(region,ring);
 std::map<std::pair<int,int>, std::map<std::string,MonitorElement*> >::iterator meRingItr = meWheelDisk.find(regionRing);
 if (meRingItr == meWheelDisk.end() || (meWheelDisk.size()==0)) {
 	meWheelDisk[regionRing]=bookRegionRing(region,ring);
 }
 
 
 std::map<std::string, MonitorElement*> meRingMap=meWheelDisk[regionRing];
 std::string ringType= (detId.region()==0) ? "Wheel" : "Disk";
 	
 
 int numberOfDigi= 0;
	
	std::vector<int> strips;
	std::vector<int> bxs;
	strips.clear(); 
	bxs.clear();
	RPCDigiCollection::const_iterator digiItr; 
	for (digiItr = ((*collectionItr ).second).first;
		digiItr!=((*collectionItr).second).second; ++digiItr){
		
		int strip= (*digiItr).strip();
		strips.push_back(strip);
		int bx=(*digiItr).bx();
		bool bxExists = false;
		for(std::vector<int>::iterator existingBX= bxs.begin();
		   existingBX != bxs.end(); ++existingBX){
		 	if (bx==*existingBX) {
				bxExists=true;
				break;
			}
		}
		if(!bxExists)bxs.push_back(bx);

	        ++numberOfDigi;

		sprintf(meId,"Occupancy_%s",detUnitLabel);
		meMap[meId]->Fill(strip);

		sprintf(meId,"BXN_%s",detUnitLabel);
		meMap[meId]->Fill(bx);
		
		sprintf(meId,"BXN_vs_strip_%s",detUnitLabel);
		meMap[meId]->Fill(strip,bx);
	
	}/// loop on Digi
 	sprintf(meId,"BXWithData_%s",detUnitLabel);
	meMap[meId]->Fill(bxs.size());
	
	for(unsigned int stripIter=0;stripIter<strips.size(); ++stripIter){
		if(strips[stripIter+1]==strips[stripIter]+1) {
			sprintf(meId,"CrossTalkHigh_%s",detUnitLabel);
			meMap[meId]->Fill(strips[stripIter]);	
		}
	}
	
	for(unsigned int stripIter2=1;stripIter2<=strips.size(); ++stripIter2){
		if(strips[stripIter2-1]==strips[stripIter2]-1) {
			sprintf(meId,"CrossTalkLow_%s",detUnitLabel);
			meMap[meId]->Fill(strips[stripIter2]);	
		}
	}

	sprintf(meId,"NumberOfDigi_%s",detUnitLabel);
	meMap[meId]->Fill(numberOfDigi);

	typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
	rangeRecHits recHitCollection =  rpcHits->get(detId);
	
	
	if(recHitCollection.first==recHitCollection.second){
		sprintf(meId,"MissingHits_%s",detUnitLabel);
		meMap[meId]->Fill((int)(counter), 1.0);
	
	}else{
		std::map<uint32_t, bool >::iterator mapItr = foundHitsInChamber.find(id);
		if (mapItr == foundHitsInChamber.end() || (foundHitsInChamber.size()==0)) {
			sprintf(meId,"RecHitCounter_%s",detUnitLabel);
			meMap[meId]->setBinContent(1, counter);
		}		
		foundHitsInChamber[id]=true;
		
		sprintf(meId,"MissingHits_%s",detUnitLabel);
		meMap[meId]->Fill((int)(counter), 0.0);
		
		RPCRecHitCollection::const_iterator it;
		int numberOfHits=0;
	
		int numbOfClusters(0);
		for (it = recHitCollection.first; it != recHitCollection.second ; it++){
 
			numbOfClusters++; 

			RPCDetId detIdRecHits=it->rpcId();
			LocalError error=it->localPositionError();//plot of errors/roll => should be gaussian	
			LocalPoint point=it->localPosition();	  //plot of coordinates/roll =>should be flat
			
			GlobalPoint globalHitPoint=surface.toGlobal(point); 

			sprintf(meId,"GlobalRecHitXYCoordinates_%s_%d",ringType.c_str(),detId.ring());
			meRingMap[meId]->Fill(globalHitPoint.x(),globalHitPoint.y());
			
			GlobalZXHitCoordinates->Fill(globalHitPoint.z(),globalHitPoint.x());			
			GlobalZYHitCoordinates->Fill(globalHitPoint.z(),globalHitPoint.y());
			GlobalZPhiHitCoordinates->Fill(globalHitPoint.z(),globalHitPoint.phi());
			
			
			int mult=it->clusterSize();		  //cluster size plot => should be within 1-3	
			int firstStrip=it->firstClusterStrip();    //plot first Strip => should be flat
			float xposition=point.x();
			//float yposition=point.y();
	
			sprintf(meId,"ClusterSize_%s",detUnitLabel);
			if(mult<=10) meMap[meId]->Fill(mult);
			if(mult>10)  meMap[meId]->Fill(11);
			
			int centralStrip=firstStrip;
			if(mult%2) {
				centralStrip+= mult/2;
			}else{	
      				float x = gRandom->Uniform(2);
				centralStrip+=(x<1)? (mult/2)-1 : (mult/2);
			}
			
			sprintf(meId,"ClusterSize_vs_CentralStrip%s",detUnitLabel);
			meMap[meId]->Fill(centralStrip,mult);
			
			for(int index=0; index<mult; ++index){
				sprintf(meId,"ClusterSize_vs_Strip%s",detUnitLabel);
				meMap[meId]->Fill(firstStrip+index,mult);
			}
			
			sprintf(meId,"ClusterSize_vs_LowerStrip%s",detUnitLabel);
			meMap[meId]->Fill(firstStrip,mult);

			sprintf(meId,"ClusterSize_vs_HigherStrip%s",detUnitLabel);
			meMap[meId]->Fill(firstStrip+mult-1,mult);
			
			sprintf(meId,"RecHitXPosition_%s",detUnitLabel);
			meMap[meId]->Fill(xposition);
 
 
			sprintf(meId,"RecHitDX_%s",detUnitLabel);
			meMap[meId]->Fill(error.xx());
 

			sprintf(meId,"RecHitX_vs_dx_%s",detUnitLabel);
			meMap[meId]->Fill(xposition,error.xx());

			//sprintf(meId,"RecHitYPosition_%s",detUnitLabel);
			//meMap[meId]->Fill(yposition);

			//sprintf(meId,"RecHitDY_%s",detUnitLabel);
			//meMap[meId]->Fill(error.yy());
 
			//sprintf(meId,"RecHitDXDY_%s",detUnitLabel);
			//meMap[meId]->Fill(error.xy());

			//sprintf(meId,"RecHitY_vs_dY_%s",detUnitLabel);
			//meMap[meId]->Fill(yposition,error.yy());
			
			numberOfHits++;
	
	}/// loop on RPCRecHits
	
		sprintf(meId,"NumberOfClusters_%s",detUnitLabel);
		meMap[meId]->Fill(numbOfClusters);
		
		if(numberOfHits>5) numberOfHits=16;
		sprintf(meId,"RecHitCounter_%s",detUnitLabel);
		meMap[meId]->Fill(numberOfHits);
	}
	
 
 }/// loop on RPC Det Unit

 std::map<uint32_t, bool >::iterator mapItrCheck;
 for (mapItrCheck = foundHitsInChamber.begin(); mapItrCheck != foundHitsInChamber.end(); ++ mapItrCheck) {
 	if(mapItrCheck->second=false){
 		uint32_t idCheck=mapItrCheck->first;
		std::map<std::string, MonitorElement*> meMapCheck=meCollection[idCheck];		
		RPCDetId detIdCheck(idCheck); 
		char detUnitLabelCheck[128];
		sprintf(detUnitLabelCheck ,"%d",detIdCheck());
		sprintf(meId,"RecHitCounter_%s",detUnitLabelCheck);
		meMapCheck[meId]->Fill(0);		
	}
 }

  

  if((!(counter%saveRootFileEventsInterval))&&(saveRootFile) ) {
    dbe->save(RootFileName);
  }
  

}
 
 

