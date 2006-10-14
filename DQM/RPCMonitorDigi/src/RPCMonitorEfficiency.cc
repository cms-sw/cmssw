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
  //  HistoOutFile= pset.getParameter<std::string>("HistoOutFile");

  EffSaveRootFile  = pset.getUntrackedParameter<bool>("EffSaveRootFile", false); 
  EffSaveRootFileEventsInterval  = pset.getUntrackedParameter<int>("EffEventsInterval", 10000); 
  EffRootFileName  = pset.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiency.root"); 

  /// get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  dbe->showDirStructure();

  _idList.clear();
}

// void RPCMonitorEfficiency::beginJob(const edm::EventSetup &)
// {
//   theFile = new TFile(HistoOutFile.c_str(),"RECREATE");
//   std::cout<<"Begin Job"<<std::endl;
//   hPositionX = new TH1F("Histo","Histo",100,0,100);
// }


void RPCMonitorEfficiency::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup ){
  typedef    std::pair<const GeomDet*,TrajectoryStateOnSurface> DetWithState;

  counter++;
  bool goodStatistic = false;
  if(counter%1000 == 0) goodStatistic = true;
  std::cout<<"COUNTER = "<< counter<<"  "<<counter%500<<"  "<<"BOOL = "<<goodStatistic<<" "<<"------------"<<std::endl;

  char layerLabel[128];
  //char meId [128];
  char meIdRPC [128];
  char meIdDT [128];
  char effIdRPC [128];
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

  bool DTevent = false;
  uint32_t id = 0;
  char detUnitLabel[128];
  std::map<std::string, MonitorElement*> meMap;

  DTRecSegment4DCollection::const_iterator segment;  


  if(all4DSegments->size()>0){
    std::cout<<"\n\n\n Number of Segment in this event = "<<all4DSegments->size()<<std::endl;



    std::map<DTChamberId,int> scounter;
    for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ if(segment->dimension()==4){
      scounter[segment->chamberId()]++;//Counting how many segments per DTChamberID, labeling each chamber with an int
    }}

    //This is the for over the DT Segments!*************************************************************
    
    for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ //loop over all the 4D Segments
      if(segment->dimension()==4 && scounter[segment->chamberId()] == 1 ){//check if the dimension of the segment is 4 and that there is only one segmentperChamber
      
	DTChamberId DTId = segment->chamberId();
	int dtWheel = DTId.wheel();
	int dtStation = DTId.station();
	int dtSector = DTId.sector();

	const GeomDet* gdet=dtGeo->idToDet(segment->geographicalId());
	const BoundPlane & DTSurface = gdet->surface();
	
	std::cout<<"\n\n-----------------++++++++++++++++++++++++++++++++++++Segment in Chamber id: Wh "<<dtWheel<<" St "<<dtStation<<" Se "<<dtSector<<std::endl;
	std::cout<<"Number of segments for DT "<<segment->chamberId()<<" = "<<scounter[segment->chamberId()]<<std::endl; //Should print 1
      
	std::set<RPCDetId> rls=rollstore[DTStationIndex(0,dtWheel,dtSector,dtStation)];
      
	std::cout <<"\n----------------------------------- Inside the Rolls asociated for this Segment"<<std::endl;

	LocalPoint localPoint= segment->localPosition();
	LocalVector localDirection=segment->localDirection();
	std::cout<<"DT Segment Point in DTLocal "<<localPoint<<std::endl;
	std::cout<<"DT Segment Direction in DTLocal "<<localDirection<<std::endl;
	std::cout<<"DT Segment Dimension "<<segment->dimension()<<std::endl; //should be 4
	float Xo=localPoint.x();
	float Yo=localPoint.y();
	
	float dx=localDirection.x();
	float dy=localDirection.y();
	float dz=localDirection.z();

	for (std::set<RPCDetId>::iterator iteraRoll = rls.begin();iteraRoll != rls.end(); iteraRoll++){//Loop over all the rolls
	  
	  const RPCRoll* eroll=dynamic_cast<const RPCRoll*> (rpcGeo->idToDetUnit(*iteraRoll));
	  const BoundPlane & RPCSurface = eroll->surface(); //To get the roll's surface
	  
	  std::cout<<"\n\n~~~~~~~!!!~~~~~"<<eroll->id()<<std::endl;
	
	  
	  GlobalPoint CenterPointRollGlobal=RPCSurface.toGlobal(LocalPoint(0,0,0));
	  //std::cout<<"Center (0,0,0) Roll in Global"<<CenterPointRollGlobal<<std::endl;
	  
	  LocalPoint CenterRollinDTFrame = DTSurface.toLocal(CenterPointRollGlobal);
	  //std::cout<<"Center (0,0,0) Roll In DTLocal"<<CenterRollinDTFrame<<std::endl;
	  
	  float D=CenterRollinDTFrame.z();
	  //std::cout<<"D="<<D<<"cm"<<std::endl;
	  
	  
	  float X=Xo+dx*D/dz;
	  float Y=Yo+dy*D/dz;
	  float Z=D;
	  
	  const RectangularStripTopology* top_=dynamic_cast<const RectangularStripTopology*>(&(eroll->topology()));
	  LocalPoint xmin = top_->localPosition(0.);
	  LocalPoint xmax = top_->localPosition((float)eroll->nstrips());
	  float rsize = fabs( xmax.x()-xmin.x() )*0.5;
	  float stripl = top_->stripLength();
	  	  
	  //std::cout<<"X Predicted in DTLocal= "<<X<<"cm"<<std::endl;
	  //std::cout<<"Y Predicted in DTLocal= "<<Y<<"cm"<<std::endl;
	  //std::cout<<"Z Predicted in DTLocal= "<<Z<<"cm"<<std::endl;
	
	  GlobalPoint GlobalPointExtrapolated = DTSurface.toGlobal(LocalPoint(X,Y,Z));
	  std::cout<<"Point ExtraPolated in Global"<<GlobalPointExtrapolated<< std::endl;
	  
	  LocalPoint PointExtrapolatedRPCFrame = RPCSurface.toLocal(GlobalPointExtrapolated);
	  std::cout<<"Point Extrapolated in RPCLocal"<<PointExtrapolatedRPCFrame<< std::endl;
	  
	  if ( fabs(PointExtrapolatedRPCFrame.z()) < 0.01  && fabs(PointExtrapolatedRPCFrame.x()) < rsize && fabs(PointExtrapolatedRPCFrame.y()) < stripl*0.5){ 
	    //conditions to find the right roll to extrapolate
	    
	    const float stripPredicted=eroll->strip(LocalPoint(PointExtrapolatedRPCFrame.x(),PointExtrapolatedRPCFrame.y(),0.)); //getting the number of the strip
	    
	    std::cout<<"\nTHE ROLL THAT CONTAINS THE CANDIDATE STRIP IS "<<eroll->id()<<" "<<"(from data) STRIP---> "<<stripPredicted<< std::endl;
	    
	    //------------------------------- HISTOGRAM STRIP PREDICTED FROM DT  -------------------
	    
	    RPCDetId  detId = eroll->id();
	    id = detId.rawId();
	    std::cout << "Found Chamber RB "<<detId.station()<<std::endl;
	    
	    _idList.push_back(id);   
	    sprintf(detUnitLabel ,"%d",id);
	    sprintf(layerLabel ,"layer%d_subsector%d_roll%d",detId.layer(),detId.subsector(),detId.roll());
	    //	    meCollection[id] = bookDetUnitMEEff(detId);
	    std::map<uint32_t, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(id);
	    if (meItr == meCollection.end() || (meCollection.size()==0)){
	       meCollection[id] = bookDetUnitMEEff(detId);
	       std::cout << "Create new histograms  for "<<layerLabel<<std::endl;
	    }
	    
	    //	    std::map<std::string, MonitorElement*> meMap=meCollection[id];
	    meMap = meCollection[id];
	    sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
	    meMap[meIdDT]->Fill(stripPredicted);

// 	    bool goodStatistic = false;
// 	    std::cout<<"Counter = "<<counter<<"  "<<"ENTRIES HISTO = "<<((int)(meMap[meIdDT]->getEntries())/100)%100<<std::endl;

// 	    if(((int)(meMap[meIdDT]->getEntries())/100)%100 == 0){
// 	      goodStatistic = true;
// 	    }

	    //------------------------------- HISTOGRAM STRIP DETECTED WITH RPC --------------------
    
	    bool anycoincidence=false;
	    int stripDetected = 0;
	    RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(eroll->id());

	    for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){//loop over the digis in the event
	      std::cout<<" digi "<<*digiIt<<std::endl;//print the digis in the event
	      stripDetected=digiIt->strip();
	      if(fabs((float)(stripDetected) - stripPredicted)<5.){//compare the strip Detected with the predicted
		std::cout <<"************!!!!!!WE HAVE A COINCEDENCE!!!!!!******* Predicted "<<stripPredicted<<" Detected "<<stripDetected<<std::endl;
		anycoincidence=true;
	      }
	    }
	    
	    if(anycoincidence==false){
	      std::cout <<"THIS PREDICTION DOESN'T HAVE ANY CORRESPONDENCE WITH THE DATA"<<std::endl;
	    }else {
	      sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
	      meMap[meIdRPC]->Fill(stripPredicted);//We can not divide two diferents things


	  }
	}
      } 
    }
  }
 }
}

void RPCMonitorEfficiency::endJob(void)
{

  std::vector<uint32_t>::iterator meIt;
  for(meIt = _idList.begin(); meIt != _idList.end(); ++meIt){

    char detUnitLabel[128];
    char meIdRPC [128];
    char meIdDT [128];
    char effIdRPC [128];
    sprintf(detUnitLabel ,"%d",*meIt);
    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
    sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
    sprintf(effIdRPC,"EfficienyFromDTExtrapolation_%s",detUnitLabel);
    
    std::map<std::string, MonitorElement*> meMap=meCollection[*meIt];

    for(unsigned int i = 1; i <= 100; ++i ){
      
      if(meMap[meIdDT]->getBinContent(i) != 0){
	float eff = meMap[meIdRPC]->getBinContent(i)/meMap[meIdDT]->getBinContent(i);
	float erreff = sqrt(eff*(1-eff)/meMap[meIdDT]->getBinContent(i));
	meMap[effIdRPC]->setBinContent(i,eff);
	meMap[effIdRPC]->setBinError(i,erreff);
	
      }
    }
  }

  if(EffSaveRootFile) dbe->save(EffRootFileName);
//    theFile->Write();
//    theFile->Close();
   std::cout<<"End Job"<<std::endl;
}

RPCMonitorEfficiency::~RPCMonitorEfficiency(){}

/*Para agregar despues para chequar que el strip que estamos precidienco tiene sentido!
	    //Division> to be placed in the appropriate place
	    //sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
	    //sprintf(meIdDT,"ExpectedOccupancyFromDT_%s",detUnitLabel);
	    //sprintf(meId,"EfficienyFromDTExtrapolation_%s",detUnitLabel);
	    //meMap[meId]->Divide(meMap[meIdRPC], meMap[meIdDT]);
	  //else {std::cout <<"...... nothing to do in this roll!"<<std::endl;}
	    //std::cout<<"id:  "<<eroll->id()<<" Number of strips "<<eroll->nstrips()<<std::endl;
	    

*/




