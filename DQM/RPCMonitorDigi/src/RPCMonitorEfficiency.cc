/** \file
 *
 *  implementation of RPCMonitorEfficiency class
 *
 *  $Date: 2006/07/15 09:29:50 $
 *  Revision: 1.2 $
 *
 * \author  Camilo Carrillo
 */

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

#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <cmath>


#include <DQM/RPCMonitorDigi/interface/RPCMonitorEfficiency.h>
///Log messages
#include <FWCore/ServiceRegistry/interface/Service.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>


/* Collaborating Class Header */
#include "FWCore/Framework/interface/MakerMacros.h" //
#include "FWCore/Framework/interface/Frameworkfwd.h"//
#include "FWCore/Framework/interface/Event.h"//
#include "FWCore/ParameterSet/interface/ParameterSet.h"//
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"//
#include "DataFormats/MuonDetId/interface/DTChamberId.h"//
#include "DataFormats/MuonDetId/interface/RPCDetId.h"//
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"

/// for DQM
#include <Geometry/DTGeometry/interface/DTGeometry.h>//
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>//
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>//
#include <Geometry/CommonDetUnit/interface/GeomDet.h>//
#include <Geometry/Records/interface/MuonGeometryRecord.h>//
#include <FWCore/Framework/interface/ESHandle.h>//
#include <FWCore/Framework/interface/eventSetupGetImplementation.h>//

#include <TrackingTools/DetLayers/interface/DetLayer.h>//
#include <RecoMuon/Records/interface/MuonRecoGeometryRecord.h>//
#include <RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h>//


class DTStationIndex{
public: 
  DTStationIndex():_region(0),_wheel(0),_sector(0),_station(0){}
  DTStationIndex(int region, int wheel, int sector, int station) : 
    _region(region),
    _wheel(wheel),
    _sector(sector),
    _station(station){}
  ~DTStationIndex(){}
  int region() const {return _region;}
  int wheel() const {return _wheel;}
  int sector() const {return _sector;}
  int station() const {return _station;}
  bool operator<(const DTStationIndex& dtind) const{
    if(dtind.region()!=this->region())
      return dtind.region()<this->region();
    else if(dtind.wheel()!=this->wheel())
      return dtind.wheel()<this->wheel();
    else if(dtind.sector()!=this->sector())
      return dtind.sector()<this->sector();
    else if(dtind.station()!=this->station())
      return dtind.station()<this->station();
    return false;
  }
private:
  int _region;
  int _wheel;
  int _sector;
  int _station; 
};


//**********************************************************************************************************



RPCMonitorEfficiency::RPCMonitorEfficiency( const edm::ParameterSet& pset ):counter(0){
  theRecHits4DLabel = pset.getParameter<std::string>("recHits4DLabel");
  digiLabel=pset.getParameter<std::string>("digiLabel");
  HistoOutFile= pset.getParameter<std::string>("HistoOutFile");

  EffSaveRootFile  = pset.getUntrackedParameter<bool>("EffSaveRootFile", false); 
  EffSaveRootFileEventsInterval  = pset.getUntrackedParameter<int>("EffEventsInterval", 10000); 
  EffRootFileName  = pset.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiency.root"); 

  /// get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  dbe->showDirStructure();
}

void RPCMonitorEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ){

  typedef    std::pair<const GeomDet*,TrajectoryStateOnSurface> DetWithState;

  char layerLabel[128];
  char meId [128];

  edm::ESHandle<DTGeometry> dtGeo;
  iSetup.get<MuonGeometryRecord>().get(dtGeo);
  
  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  // Take the whole det layer geometry
  edm::ESHandle<MuonDetLayerGeometry> detLayerGeometry;
  iSetup.get<MuonRecoGeometryRecord>().get(detLayerGeometry);
  

  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  iEvent.getByLabel(theRecHits4DLabel, all4DSegments);
  
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByLabel(digiLabel, rpcDigis);


  std::map<DTStationIndex,std::set<RPCDetId> > rollstore;

  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();
       it<rpcGeo->dets().end();it++){
    RPCRoll* ir = dynamic_cast<RPCRoll*>(*it);
    RPCDetId rpcId = ir->id();
    int region=rpcId.region();
    int wheel=rpcId.ring();
    int sector=rpcId.sector();
    int station=rpcId.station();

    DTStationIndex ind(region,wheel,sector,station);
    
    std::set<RPCDetId> myrolls;
    if (rollstore.find(ind)!=rollstore.end()){
      myrolls=rollstore[ind];
    }
    myrolls.insert(rpcId);
    rollstore[ind]=myrolls;
    myrolls.insert(rpcId);
    rollstore[ind]=myrolls;
  }
  

  DTRecSegment4DCollection::const_iterator segment;  


  //This is the for over the DT Segments!*************************************************************

  if(all4DSegments->size()>0){

    std::map<DTChamberId,int> scounter;
    for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ if(segment->dimension()==4){
      scounter[segment->chamberId()]++;
    }}
    for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
      if(segment->dimension()==4 && scounter[segment->chamberId()] == 1 ){
      
      DTChamberId DTId = segment->chamberId();
      int dtWheel = DTId.wheel();
      int dtStation = DTId.station();
      int dtSector = DTId.sector();

      const GeomDet* gdet=dtGeo->idToDet(segment->geographicalId());
      const BoundPlane & DTSurface = gdet->surface();

      
      std::set<RPCDetId> rls=rollstore[DTStationIndex(0,dtWheel,dtSector,dtStation)];
      
      LocalPoint localPoint= segment->localPosition();
      LocalVector localDirection=segment->localDirection();

      float Xo=localPoint.x();
      float Yo=localPoint.y();
      
      float dx=localDirection.x();
      float dy=localDirection.y();
      float dz=localDirection.z();
      for (std::set<RPCDetId>::iterator iteraRoll = rls.begin();iteraRoll != rls.end(); iteraRoll++){//Barre rolls del DT
	
	const RPCRoll* eroll=dynamic_cast<const RPCRoll*>
	  (rpcGeo->idToDetUnit(*iteraRoll));
	const BoundPlane & RPCSurface = eroll->surface();
	
	

	GlobalPoint CenterPointRollGlobal=RPCSurface.toGlobal(LocalPoint(0,0,0));
	
	LocalPoint AgainCenterRoll = DTSurface.toLocal(CenterPointRollGlobal);
	
	float D=AgainCenterRoll.z();
	
	float X=Xo+dx*D/dz;
	float Y=Yo+dy*D/dz;
	float Z=D;

	const RectangularStripTopology* top_ =  
	  dynamic_cast<const RectangularStripTopology*>(&(eroll->topology()));
	LocalPoint xmin = top_->localPosition(0.);
	LocalPoint xmax = top_->localPosition((float)eroll->nstrips());
	float rsize = fabs( xmax.x()-xmin.x() )/2.;
	float stripl = top_->stripLength();
	

	GlobalPoint GlobalPointExtrapolated = DTSurface.toGlobal(LocalPoint(X,Y,Z));


	LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);

	if ( fabs(PointExtrapolatedRPCFrame.z()) < 0.01  &&
	     fabs(PointExtrapolatedRPCFrame.x()) < rsize &&
	     fabs(PointExtrapolatedRPCFrame.y()) < stripl/2.){ 
	  
	  const float strip=eroll->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.));

	  //------------------------------- HISTO PREDICTED STRIP --------------------------------

	  RPCDetId  detId = eroll->id();
	  uint32_t id = detId.rawId();

	  //	  uint32_t id=detId(); 
 
	  char detUnitLabel[128];
	  sprintf(detUnitLabel ,"%d",id);
	  sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detId.layer(),detId.subsector(),detId.roll());
	  
	  std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
	  if (meItr == meCollection.end() || (meCollection.size()==0)) {
	    meCollection[id] = bookDetUnitMEEff(detId);
	  }
	  std::map<std::string, MonitorElement*> meMap=meCollection[id];

	  sprintf(meId,"Occupancy_%s",detUnitLabel);
	  meMap[meId]->Fill(strip);

	  //______________________________________________________________________________________

	  float stripDetected = 0;
	  RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(eroll->id());
	
	  for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;//print the digis in the event
	       digiIt!=rpcRangeDigi.second;
	       ++digiIt){
	    
	    
	    if (digiIt->strip() < 1 || digiIt->strip() > eroll->nstrips() ){
	      std::cout <<" XXXXXXXXXXXXX Problemt with "<<eroll->id()<<std::endl;
	    }
	    
	    if(digiIt->strip() == (int)(strip)){
	      stripDetected = strip;
	      break;
	    }
	  }
	  sprintf(meId,"DetOccupancy_%s",detUnitLabel);
	  if(stripDetected != 0){
	    meMap[meId]->Fill(stripDetected);
	  }
	}
      }
    } 
    }
  }
}

void RPCMonitorEfficiency::endJob(void)
{
   if(EffSaveRootFile) dbe->save(EffRootFileName);
}

RPCMonitorEfficiency::~RPCMonitorEfficiency(){}







