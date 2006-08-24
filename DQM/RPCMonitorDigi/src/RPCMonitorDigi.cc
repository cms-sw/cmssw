/** \file
 *
 *  implementation of RPCMonitorDigi class
 *
 *  $Date: 2006/07/11 16:11:38 $
 *  $Revision: 1.11 $
 *
 * \author Ilaria Segoni
 */

#include <map>
#include <string>

#include <DQM/RPCMonitorDigi/interface/RPCMonitorDigi.h>

///Data Format
#include <DataFormats/RPCDigi/interface/RPCDigi.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>

///RPCRecHits
#include <DataFormats/RPCRecHit/interface/RPCRecHitCollection.h>
#include <Geometry/Surface/interface/LocalError.h>
#include <Geometry/Vector/interface/LocalPoint.h>


///Log messages
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>


RPCMonitorDigi::RPCMonitorDigi( const edm::ParameterSet& pset ):counter(0){

  nameInLog = pset.getUntrackedParameter<std::string>("moduleLogName", "RPC_DQM");

  saveRootFile  = pset.getUntrackedParameter<bool>("DigiDQMSaveRootFile", false); 
  saveRootFileEventsInterval  = pset.getUntrackedParameter<int>("DigiEventsInterval", 10000); 
  RootFileName  = pset.getUntrackedParameter<std::string>("RootFileNameDigi", "RPCMonitor.root"); 
  
  /// get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  dbe->showDirStructure();


}


RPCMonitorDigi::~RPCMonitorDigi(){
}

void RPCMonitorDigi::endJob(void)
{
  if(saveRootFile) dbe->save(RootFileName);
}


void RPCMonitorDigi::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup ){
 edm::LogInfo (nameInLog) <<"Beginning analyzing event " << counter;

 char detUnitLabel[128];
 char layerLabel[128];
 char meId [128];

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

 
 sprintf(detUnitLabel ,"%d",detId());
 sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detId.layer(),detId.subsector(),detId.roll());
 
 std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
 if (meItr == meCollection.end() || (meCollection.size()==0)) {
 	meCollection[id]=bookDetUnitME(detId);
 }
 std::map<std::string, MonitorElement*> meMap=meCollection[id];
 	
 
 int numberOfDigi= 0;
	
	RPCDigiCollection::const_iterator digiItr; 
	for (digiItr = ((*collectionItr ).second).first;
		digiItr!=((*collectionItr).second).second; ++digiItr){
		
		int strip= (*digiItr).strip();
		int bx=(*digiItr).bx();
		//(*digiItr).print();
	        ++numberOfDigi;

		sprintf(meId,"Occupancy_%s",detUnitLabel);
		meMap[meId]->Fill(strip);

		sprintf(meId,"BXN_%s",detUnitLabel);
		meMap[meId]->Fill(bx);
	
	}/// loop on Digi

	sprintf(meId,"NumberOfDigi_%s",detUnitLabel);
	meMap[meId]->Fill(numberOfDigi);

	typedef std::pair<RPCRecHitCollection::const_iterator, RPCRecHitCollection::const_iterator> rangeRecHits;
	rangeRecHits recHitCollection =  rpcHits->get(detId);
	
	
	if(recHitCollection.first==recHitCollection.second){
		sprintf(meId,"MissingHits_%s",detUnitLabel);
		meMap[meId]->Fill((int)(counter), 1.0);
	
	}else{
		sprintf(meId,"MissingHits_%s",detUnitLabel);
		meMap[meId]->Fill((int)(counter), 0.0);
		
		RPCRecHitCollection::const_iterator it;
		int numberOfHits=0;
	
		int numbOfClusters(0);
		for (it = recHitCollection.first; it != recHitCollection.second ; it++){
 
			numbOfClusters++; 

			RPCDetId detIdRecHits=it->rpcId();
			uint32_t idRecHits=detIdRecHits(); 
			LocalError error=it->localPositionError();//plot of errors/roll => should be gaussian	
			LocalPoint point=it->localPosition();	  //plot of coordinates/roll =>should be flat
			int mult=it->clusterSize();		  //cluster size plot => should be within 3-4	
			int firstStrip=it->firstClusterStrip();    //plot first Strip => should be flat
			float xposition=point.x();
			float yposition=point.y();
	
			sprintf(meId,"ClusterSize_%s",detUnitLabel);
			if(mult<=10) meMap[meId]->Fill(mult);
			if(mult>10)  meMap[meId]->Fill(11);
			
			sprintf(meId,"RecHitXPosition_%s",detUnitLabel);
			meMap[meId]->Fill(xposition);
 
			sprintf(meId,"RecHitYPosition_%s",detUnitLabel);
			meMap[meId]->Fill(yposition);
 
			sprintf(meId,"RecHitDX_%s",detUnitLabel);
			meMap[meId]->Fill(error.xx());
 
			sprintf(meId,"RecHitDY_%s",detUnitLabel);
			meMap[meId]->Fill(error.yy());
 
			sprintf(meId,"RecHitDXDY_%s",detUnitLabel);
			meMap[meId]->Fill(error.xy());

			sprintf(meId,"RecHitX_vs_dx_%s",detUnitLabel);
			meMap[meId]->Fill(xposition,error.xx());

			sprintf(meId,"RecHitY_vs_dY_%s",detUnitLabel);
			meMap[meId]->Fill(yposition,error.yy());
			numberOfHits++;
	
	}/// loop on RPCRecHits
	
		sprintf(meId,"NumberOfClusters_%s",detUnitLabel);
		meMap[meId]->Fill(numbOfClusters);
	}
	
 
 }/// loop on RPC Det Unit


  

  if((!(counter%saveRootFileEventsInterval))&&(saveRootFile) ) {
    dbe->save(RootFileName);
  }
  
  
  counter++;
  //dbe->showDirStructure();
  //usleep(10000000);

}
 
 

