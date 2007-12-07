/**********************************************
 *                                            *
 *           Giuseppe Roselli                 *
 *         INFN, Sezione di Bari              *
 *      Via Amendola 173, 70126 Bari          *
 *         Phone: +390805443218               *
 *      giuseppe.roselli@ba.infn.it           *
 *                                            *
 *                                            *
 **********************************************/


#include "DQM/RPCMonitorDigi/interface/RPCEfficiencyFromTrack.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include <DataFormats/RPCRecHit/interface/RPCRecHit.h>
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>
#include <Geometry/CSCGeometry/interface/CSCGeometry.h>
#include <DataFormats/CSCRecHit/interface/CSCSegmentCollection.h>
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include <DataFormats/GeometrySurface/interface/LocalError.h>
#include <DataFormats/GeometryVector/interface/LocalPoint.h>
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include <memory>
#include <cmath>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TAxis.h"

//
// class decleration
//
using namespace edm;
using namespace reco;
using namespace std;



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


class CSCStationIndex{
public:
  CSCStationIndex():_region(0),_station(0),_ring(0),_chamber(0){}
  CSCStationIndex(int region, int station, int ring, int chamber):
    _region(region),
    _station(station),
    _ring(ring),
    _chamber(chamber){}
  ~CSCStationIndex(){}
  int region() const {return _region;}
  int station() const {return _station;}
  int ring() const {return _ring;}
  int chamber() const {return _chamber;}
  bool operator<(const CSCStationIndex& cscind) const{
    if(cscind.region()!=this->region())
      return cscind.region()<this->region();
    else if(cscind.station()!=this->station())
      return cscind.station()<this->station();
    else if(cscind.ring()!=this->ring())
      return cscind.ring()<this->ring();
    else if(cscind.chamber()!=this->chamber())
      return cscind.chamber()<this->chamber();
    return false;
  }

private:
  int _region;
  int _station;
  int _ring;  
  int _chamber;
};






RPCEfficiencyFromTrack::RPCEfficiencyFromTrack(const edm::ParameterSet& iConfig){
  std::map<RPCDetId, int> buff;
  counter.clear();
  counter.reserve(3);
  counter.push_back(buff);
  counter.push_back(buff);
  counter.push_back(buff);
  totalcounter.clear();
  totalcounter.reserve(3);
  totalcounter[0]=0;
  totalcounter[1]=0;
  totalcounter[2]=0;
  
  MeasureEndCap = iConfig.getParameter<bool>("ReadEndCap");
  MeasureBarrel = iConfig.getParameter<bool>("ReadBarrel");
  maxRes = iConfig.getParameter<double>("EfficiencyCut");
  EffSaveRootFile  = iConfig.getUntrackedParameter<bool>("EffSaveRootFile", true); 
  EffSaveRootFileEventsInterval  = iConfig.getUntrackedParameter<int>("EffEventsInterval", 1000); 
  EffRootFileName  = iConfig.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiencyFromTrack.root"); 
  TjInput  = iConfig.getUntrackedParameter<std::string>("trajectoryInput");
  RPCDataLabel = iConfig.getUntrackedParameter<std::string>("rpcRecHitLabel");
  thePropagatorName = iConfig.getParameter<std::string>("PropagatorName");
  thePropagator = 0;

  GlobalRootLabel= iConfig.getUntrackedParameter<std::string>("GlobalRootFileName","GlobalEfficiencyFromTrack.root");
  fOutputFile  = new TFile(GlobalRootLabel.c_str(), "RECREATE" );
  hRecPt = new TH1F("RecPt","ReconstructedPt",100,0.5,100.5);
  hGlobalRes = new TH1F("GlobalResiduals","GlobalRPCResiduals",50,-15.,15.);
  hGlobalPull = new TH1F("GlobalPull","GlobalRPCPull",50,-15.,15.);
  histoMean = new TH1F("MeanEfficincy","MeanEfficiency_vs_Ch",100,20.5,120.5);
  // get hold of back-end interface
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
  edm::Service<MonitorDaemon> daemon;
  daemon.operator->();

  _idList.clear(); 

}


RPCEfficiencyFromTrack::~RPCEfficiencyFromTrack()
{
  fOutputFile->Write();
  fOutputFile->Close();
}


void RPCEfficiencyFromTrack::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  char layerLabel[128];
  char meIdRPC [128];
  char meIdTrack [128];
  std::map<RPCDetId, int> buff;


  Handle<Trajectories> trajectories;
  iEvent.getByLabel(TjInput,trajectories);

  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  edm::ESHandle<DTGeometry> dtGeo;
  iSetup.get<MuonGeometryRecord>().get(dtGeo);

  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByLabel(RPCDataLabel,rpcHits);

  ESHandle<MagneticField> theMGField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMGField);
  
  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  Handle<reco::TrackCollection> staTracks;
  iEvent.getByLabel(TjInput, staTracks);

  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  iEvent.getByLabel("dt4DSegments", all4DSegments);

  edm::Handle<CSCSegmentCollection> allCSCSegments;
  iEvent.getByLabel("cscSegments", allCSCSegments);
  
  edm::ESHandle<CSCGeometry> cscGeo;
  iSetup.get<MuonGeometryRecord>().get(cscGeo);

  reco::TrackCollection::const_iterator staTrack;

  ESHandle<Propagator> prop;
  iSetup.get<TrackingComponentsRecord>().get(thePropagatorName, prop);
  thePropagator = prop->clone();
  thePropagator->setPropagationDirection(anyDirection);


  std::map<DTStationIndex,std::set<RPCDetId> > rollstoreDT;
  std::map<CSCStationIndex,std::set<RPCDetId> > rollstoreCSC;    
  
  for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
    if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
      RPCChamber* ch = dynamic_cast< RPCChamber* >( *it ); 
      std::vector< const RPCRoll*> roles = (ch->rolls());
      for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	RPCDetId rpcId = (*r)->id();
	int region=rpcId.region();
	
	if(region==0 && MeasureBarrel==true){
	  int wheel=rpcId.ring();
	  int sector=rpcId.sector();
	  int station=rpcId.station();
	  DTStationIndex ind(region,wheel,sector,station);
	  std::set<RPCDetId> myrolls;
	  if (rollstoreDT.find(ind)!=rollstoreDT.end()) myrolls=rollstoreDT[ind];
	  myrolls.insert(rpcId);
	  rollstoreDT[ind]=myrolls;
	}
	else if( MeasureEndCap==true && region!=0){
	  int region=rpcId.region();
          int station=rpcId.station();
          int ring=rpcId.ring();
          int cscring=ring;
          int cscstation=station;
	  RPCGeomServ rpcsrv(rpcId);
	  int rpcsegment = rpcsrv.segment();
	  int cscchamber = rpcsegment;
          if((station==2||station==3)&&ring==3){
            cscring = 2;
          }
	  if((station==4)&&(ring==2||ring==3)){
            cscstation=3;
            cscring=2;
          }
          CSCStationIndex ind(region,cscstation,cscring,cscchamber);
          std::set<RPCDetId> myrolls;
	  if (rollstoreCSC.find(ind)!=rollstoreCSC.end()){
            myrolls=rollstoreCSC[ind];
          }
          
          myrolls.insert(rpcId);
          rollstoreCSC[ind]=myrolls;
        }
      }
    }
  }
  
  std::vector<RPCDetId> RollFinderFromDT;
  RollFinderFromDT.clear();

  std::vector<RPCDetId> RollFinderFromCSC;
  RollFinderFromCSC.clear();

  if(all4DSegments->size()>0){
    std::map<DTChamberId,int> scounter;
    DTRecSegment4DCollection::const_iterator segment;  
    for (segment = all4DSegments->begin();segment!=all4DSegments->end(); ++segment){
      scounter[segment->chamberId()]++;
    }    
    for (segment = all4DSegments->begin(); segment != all4DSegments->end(); ++segment){ 
      DTChamberId DTId = segment->chamberId();
      if(scounter[DTId] == 1){	
	int dtWheel = DTId.wheel();
	int dtStation = DTId.station();
	int dtSector = DTId.sector();
	std::set<RPCDetId> rollsForThisDT = rollstoreDT[DTStationIndex(0,dtWheel,dtSector,dtStation)];
	for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisDT.begin();iteraRoll != rollsForThisDT.end(); iteraRoll++){
	  RollFinderFromDT.push_back(*iteraRoll);
	}
      }
    }
  }

  if(allCSCSegments->size()>0){
    std::map<CSCDetId,int> CSCSegmentsCounter;
    CSCSegmentCollection::const_iterator segment;
    for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
      CSCSegmentsCounter[segment->cscDetId()]++;
    }    
    for (segment = allCSCSegments->begin();segment!=allCSCSegments->end(); ++segment){
      if(segment->dimension()==4){
	CSCDetId CSCId = segment->cscDetId();
	if(CSCSegmentsCounter[CSCId]==1){
	  int cscEndCap = CSCId.endcap();
	  int cscStation = CSCId.station();
	  int cscRing = CSCId.ring();
	  int rpcRegion = 1; if(cscEndCap==2) rpcRegion= -1;
	  int rpcRing = cscRing;
	  if(rpcRing==4)rpcRing =1;
	  int rpcStation = cscStation;
	  int rpcSegment = 0;
	  if(cscStation!=1&&cscRing==1){
	    rpcSegment = CSCId.chamber();
	  }
	  else{
	    rpcSegment = (CSCId.chamber()==1) ? 36 : CSCId.chamber()-1;
	  }
	  std::set<RPCDetId> rollsForThisCSC = rollstoreCSC[CSCStationIndex(rpcRegion,rpcStation,rpcRing,rpcSegment)];
	  for (std::set<RPCDetId>::iterator iteraRoll = rollsForThisCSC.begin();iteraRoll != rollsForThisCSC.end(); iteraRoll++){
	    RollFinderFromCSC.push_back(*iteraRoll);
	  }
	}
      }
    }
  }

  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 
    TrajectoryStateOnSurface tState = track.innermostMeasurementState();
    FreeTrajectoryState* fState = tState.freeState();    
    std::cout << "Traj state"<<fState->position().x()<<"\t"<<fState->position().y()<<"\t"<<fState->position().z() << std::endl;

    for (std::vector<RPCDetId>::iterator iteraRoll = RollFinderFromDT.begin();iteraRoll != RollFinderFromDT.end(); iteraRoll++){	
      const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
      const BoundPlane& rpcPlane = rollasociated->surface();
 
      //Propagator
      TrajectoryStateOnSurface tsosAtRPC = thePropagator->propagate(*fState,rpcPlane);	  
      
      const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&(rollasociated->topology()));
      LocalPoint xmin = top_->localPosition(0.);
      LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
      float rsize = fabs( xmax.x()-xmin.x() )*0.5;
      float stripl = top_->stripLength();
      if(tsosAtRPC.isValid()
	 && fabs(tsosAtRPC.localPosition().z()) < 0.01 
	 && fabs(tsosAtRPC.localPosition().x()) < rsize 
	 && fabs(tsosAtRPC.localPosition().y()) < stripl*0.5){
	
	RPCDetId rollId = rollasociated->id();
	RPCGeomServ RPCname(rollId);
	std::string nameRoll = RPCname.name();
	_idList.push_back(nameRoll);
	std::cout<<"RPC -- Candidate"<<nameRoll<<std::endl;
	char detUnitLabel[128];
	sprintf(detUnitLabel ,"%s",nameRoll.c_str());
	sprintf(layerLabel ,"%s",nameRoll.c_str());
	std::map<std::string, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(nameRoll);
	if (meItr == meCollection.end()){
	  meCollection[nameRoll] = bookDetUnitTrackEff(rollId);
	}
	std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];

	totalcounter[0]++;
	buff=counter[0];
	buff[rollId]++;
	counter[0]=buff;

	int stripPredicted = (int)(rollasociated->strip(tsosAtRPC.localPosition()));
	double xextrap = tsosAtRPC.localPosition().x();
	
	sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
	meMap[meIdTrack]->Fill(stripPredicted);
	

	RPCRecHitCollection::range rpcRecHitRange = rpcHits->get(rollasociated->id());
	RPCRecHitCollection::const_iterator recIt;

	bool anycoincidence=false;
	std::vector<double> ResVec;
	ResVec.clear();
	std::vector<double> RecErr;
	RecErr.clear();
	std::vector<double> extrVec;
	extrVec.clear();
	std::vector<double> posVec;
	posVec.clear();
	std::vector<double> stripD;
	stripD.clear();
	std::vector<double> stripPr;
	stripPr.clear();
	
	double res=0.;
	
	for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
	  LocalPoint rhitlocal = (*recIt).localPosition();
	  double rhitpos = rhitlocal.x();  
	  double stripDetected = (int)(rollasociated->strip(rhitlocal));
	  LocalError RecError = (*recIt).localPositionError();
	  double sigmaRec = RecError.xx();
	  res = (double)(xextrap - rhitpos);
	  
	  ResVec.push_back(res);		
	  RecErr.push_back(sigmaRec);
	  extrVec.push_back(xextrap);	
	  posVec.push_back(rhitpos);	
	  stripD.push_back(stripDetected);	
	  stripPr.push_back(stripPredicted);
	}

	if(ResVec.size()>0){
	  res = ResVec[0];
	  int rpos=0;
	  
	  for(unsigned int rs=0;rs<ResVec.size();rs++){
	    if(fabs(ResVec[rs]) < fabs(res)){
	      res = ResVec[rs];
	      rpos=rs;
	    }
	  }
	  std::cout<<"**********************************************"<<std::endl;
	  std::cout<<"\t                                   "<<std::endl;
	  std::cout<<"Point Extrapolated                   "<<extrVec[rpos]<<std::endl;
	  std::cout<<"Real Point                           "<<posVec[rpos]<<std::endl;
	  std::cout<<"**********************************************"<<std::endl;
	  std::cout<<"Strip Extrapolated "<<stripPr[rpos]<<" Strip Detected "<<stripD[rpos]<<std::endl;
	  std::cout<<"**********************************************"<<std::endl;
	  
	  sprintf(meIdRPC,"Residuals_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(res);
	  sprintf(meIdRPC,"Residuals_VS_RecPt_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(tsosAtRPC.globalMomentum().perp(),res);

	  hGlobalRes->Fill(res);
	  hGlobalPull->Fill(res/RecErr[rpos]);
	  hRecPt->Fill(tsosAtRPC.globalMomentum().perp());


	  if(fabs(res)<maxRes){
	    anycoincidence=true;
	    std::cout<<"Good Match "<<"\t"<<"Residuals = "<<res<<"\t"<<nameRoll<<std::endl;
	  }
	  else{
	    anycoincidence=false;
	    std::cout<<"No Match "<<"\t"<<"Residuals = "<<res<<"\t"<<nameRoll<<std::endl;
	  }
	  	  
	  if(anycoincidence==true){
	    totalcounter[1]++;
	    buff=counter[1];
	    buff[rollId]++;
	    counter[1]=buff;
	    
	    sprintf(meIdRPC,"RealDetectedOccupancy_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(stripD[rpos]);
	    
	    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(stripPr[rpos]);
	    
	  }else{
	    totalcounter[2]++;
	    buff=counter[2];
	    buff[rollId]++;
	    counter[2]=buff;
	  }
	}
      }
    }
  }
}  



void RPCEfficiencyFromTrack::beginJob(const edm::EventSetup&)
{
  std::cout<<"----------------Beginning------------------"<<std::endl;
}

void RPCEfficiencyFromTrack::endJob(){

  std::map<RPCDetId, int> pred = counter[0];
  std::map<RPCDetId, int> obse = counter[1];
  std::map<RPCDetId, int> reje = counter[2];
  std::map<RPCDetId, int>::iterator irpc;
  
  for (irpc=pred.begin(); irpc!=pred.end();irpc++){
    RPCDetId id=irpc->first;
    int p=pred[id]; 
    int o=obse[id]; 
    int r=reje[id]; 
    //assert(p==o+r);
   
    if(p!=0){
      float ef = float(o)/float(p); 
      float er = sqrt(ef*(1.-ef)/float(p));
      std::cout <<"\n "<<id<<"\t Predicted "<<p<<"\t Observed "<<o<<"\t Eff = "<<ef*100.<<" % +/- "<<er*100.<<" %";
      histoMean->Fill(ef*100.);
      if(ef<0.8){
	std::cout<<"\t \t Warning!";
      } 
    }
    else{
      std::cout<<"No predictions in this file predicted=0"<<std::endl;
    }
  }
  if(totalcounter[0]!=0){
    float tote = float(totalcounter[1])/float(totalcounter[0]);
    float totr = sqrt(tote*(1.-tote)/float(totalcounter[0]));
    std::cout <<"\n\n \t \t TOTAL EFFICIENCY \t Predicted "<<totalcounter[0]<<"\t Observed "<<totalcounter[1]<<"\t Eff = "<<tote*100.<<"\t +/- \t"<<totr*100.<<"%"<<std::endl;
    std::cout <<totalcounter[1]<<" "<<totalcounter[0]<<" flagcode"<<std::endl;

  }
  else{
    std::cout<<"No predictions in this file = 0!!!"<<std::endl;
  }

  std::vector<std::string>::iterator meIt;
  int id=0;
  for(meIt = _idList.begin(); meIt != _idList.end(); ++meIt){
    id++;

    char detUnitLabel[128];
    char meIdRPC [128];
    char meIdTrack [128];
    char meIdRes [128];
    char effIdRPC [128];
    sprintf(detUnitLabel ,"%s",(*meIt).c_str());
    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
    sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
    sprintf(meIdRes,"Residuals_%s",detUnitLabel);
    sprintf(effIdRPC,"EfficienyFromTrackExtrapolation_%s",detUnitLabel);
    std::map<std::string, MonitorElement*> meMap=meCollection[*meIt];

    for(unsigned int i=1;i<=100;++i){
      if(meMap[meIdTrack]->getBinContent(i) != 0){
	float eff = meMap[meIdRPC]->getBinContent(i)/meMap[meIdTrack]->getBinContent(i);
	float erreff = sqrt(eff*(1-eff)/meMap[meIdTrack]->getBinContent(i));
	meMap[effIdRPC]->setBinContent(i,eff*100.);
	meMap[effIdRPC]->setBinError(i,erreff*100.);
      }
    }
  }  
  if(EffSaveRootFile) dbe->save(EffRootFileName);
}

DEFINE_FWK_MODULE(RPCEfficiencyFromTrack);
