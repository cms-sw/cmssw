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

#include "DataFormats/RPCDigi/interface/RPCDigi.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
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

#include <DataFormats/MuonDetId/interface/DTChamberId.h>
#include <Geometry/DTGeometry/interface/DTGeometry.h>
#include <DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h>

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

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include <memory>
#include <cmath>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TAxis.h"



using namespace edm;
using namespace reco;
using namespace std;


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
  gmtSource_=iConfig.getParameter< InputTag >("gmtSource");
  EffSaveRootFile  = iConfig.getUntrackedParameter<bool>("EffSaveRootFile", true); 
  EffSaveRootFileEventsInterval  = iConfig.getUntrackedParameter<int>("EffEventsInterval", 1000); 
  DTTrigValue = iConfig.getUntrackedParameter<int>("dtTrigger",-10); 

  EffRootFileName  = iConfig.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiencyFromTrack.root"); 
  TjInput  = iConfig.getUntrackedParameter<std::string>("trajectoryInput");
  RPCDataLabel = iConfig.getUntrackedParameter<std::string>("rpcRecHitLabel");
  digiLabel = iConfig.getUntrackedParameter<std::string>("rpcDigiLabel");
  thePropagatorName = iConfig.getParameter<std::string>("PropagatorName");
  thePropagator = 0;

  GlobalRootLabel= iConfig.getUntrackedParameter<std::string>("GlobalRootFileName","GlobalEfficiencyFromTrack.root");
  fOutputFile  = new TFile(GlobalRootLabel.c_str(), "RECREATE" );

  hGlobalRes = new TH1F("GlobalResiduals","GlobalRPCResiduals",50,-15.,15.);
  hGlobalPull = new TH1F("GlobalPull","GlobalRPCPull",50,-15.,15.);
  histoMean = new TH1F("MeanEfficincy","MeanEfficiency_vs_Ch",60,20.5,120.5);


  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();
  _idList.clear(); 
  Run=0;
  effres = new ofstream("EfficiencyResults.dat");
}


RPCEfficiencyFromTrack::~RPCEfficiencyFromTrack(){
  effres->close();
  delete effres;

  fOutputFile->WriteTObject(hGlobalRes);
  fOutputFile->WriteTObject(hGlobalPull);
  fOutputFile->WriteTObject(histoMean);


  fOutputFile->Close();
}



void RPCEfficiencyFromTrack::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  char layerLabel[128];
  char meIdRPC [128];
  char meIdTrack [128];
  std::map<RPCDetId, int> buff;

  edm::ESHandle<RPCGeometry> rpcGeo;
  iSetup.get<MuonGeometryRecord>().get(rpcGeo);

  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByLabel(RPCDataLabel,rpcHits);

  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByLabel(digiLabel, rpcDigis);

  ESHandle<MagneticField> theMGField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMGField);
  
  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);


  int nDTTF = 0;
  if(DTTrigValue!=-10){
    edm::Handle<std::vector<L1MuRegionalCand> > rpcBarrel;
    edm::Handle<std::vector<L1MuRegionalCand> > rpcForward;
    iEvent.getByLabel ("gtDigis","RPCb",rpcBarrel);
    iEvent.getByLabel ("gtDigis","RPCf",rpcForward);
    edm::Handle<L1MuGMTReadoutCollection> pCollection;
    try {
      iEvent.getByLabel(gmtSource_,pCollection);
    }
    catch (...) {
      return;
    }

    L1MuGMTReadoutCollection const* gmtrc = pCollection.product();
    vector<L1MuGMTReadoutRecord> gmt_records = gmtrc->getRecords();
    vector<L1MuGMTReadoutRecord>::const_iterator RRItr;

    for(RRItr = gmt_records.begin(); RRItr != gmt_records.end(); RRItr++ ){    
      vector<L1MuRegionalCand> DTTFCands  = RRItr->getDTBXCands();
      vector<L1MuGMTExtendedCand>::const_iterator GMTItr;
      int BxInEvent = RRItr->getBxInEvent();
    
      if(BxInEvent!=0) continue;
      vector<L1MuRegionalCand>::const_iterator DTTFItr;
      for( DTTFItr = DTTFCands.begin(); DTTFItr != DTTFCands.end(); ++DTTFItr ) {
	if(!DTTFItr->empty()) nDTTF++;
      }
    }
  }

  Handle<reco::TrackCollection> staTracks;
  iEvent.getByLabel(TjInput, staTracks);

  reco::TrackCollection::const_iterator staTrack;

  ESHandle<Propagator> prop;
  iSetup.get<TrackingComponentsRecord>().get(thePropagatorName, prop);
  thePropagator = prop->clone();
  thePropagator->setPropagationDirection(anyDirection);
 

  std::map<RPCDetId,TrajectoryStateOnSurface> RPCstate;


  if(staTracks->size()!=0 && nDTTF>DTTrigValue){
    for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
      reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry);
 
      RPCstate.clear();
 
      if(track.numberOfValidHits()>20){
	for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
	  if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
	    RPCChamber* ch = dynamic_cast< RPCChamber* >( *it );
	    std::vector< const RPCRoll*> rolhit = (ch->rolls());

	    for(std::vector<const RPCRoll*>::const_iterator itRoll = rolhit.begin();itRoll != rolhit.end(); ++itRoll){
	      RPCDetId rollId=(*itRoll)->id();
	      const BoundPlane *rpcPlane =  &((*itRoll)->surface());

	      //Barrel
	      if(MeasureBarrel==true && rollId.region()==0){		
		const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&((*itRoll)->topology()));
		LocalPoint xmin = top_->localPosition(0.);
		LocalPoint xmax = top_->localPosition((float)(*itRoll)->nstrips());
		float rsize = fabs( xmax.x()-xmin.x() )*0.5;
		float stripl = top_->stripLength();
		
		TrajectoryStateClosestToPoint tcp = track.impactPointTSCP();
		const FreeTrajectoryState &fTS=tcp.theState();
		const FreeTrajectoryState *FreeState = &fTS;
		TrajectoryStateOnSurface tsosAtRPC = thePropagator->propagate(*FreeState,*rpcPlane);
	      
		if(tsosAtRPC.isValid()
		   && fabs(tsosAtRPC.localPosition().z()) < 0.01 
		   && fabs(tsosAtRPC.localPosition().x()) < rsize 
		   && fabs(tsosAtRPC.localPosition().y()) < stripl*0.5
		   && tsosAtRPC.localError().positionError().xx()<1.
		   && tsosAtRPC.localError().positionError().yy()<1.){
		  RPCstate[rollId]=tsosAtRPC;
		}	      
	      }

	      //EndCap
	      if(MeasureEndCap==true && rollId.region()!=0){	      
		const TrapezoidalStripTopology* top_= dynamic_cast<const TrapezoidalStripTopology*> (&((*itRoll)->topology()));
		LocalPoint xmin = top_->localPosition(0.);
		LocalPoint xmax = top_->localPosition((float)(*itRoll)->nstrips());
		float rsize = fabs( xmax.x()-xmin.x() )*0.5;
		float stripl = top_->stripLength();

		TrajectoryStateClosestToPoint tcp = track.impactPointTSCP();
		const FreeTrajectoryState &fTS=tcp.theState();
		const FreeTrajectoryState *FreeState = &fTS;
		TrajectoryStateOnSurface tsosAtRPC = thePropagator->propagate(*FreeState,*rpcPlane);
	      
		if(tsosAtRPC.isValid()
		   && fabs(tsosAtRPC.localPosition().z()) < 0.01 
		   && fabs(tsosAtRPC.localPosition().x()) < rsize 
		   && fabs(tsosAtRPC.localPosition().y()) < stripl*0.5
		   && tsosAtRPC.localError().positionError().xx()<1.
		   && tsosAtRPC.localError().positionError().yy()<1.){
		  RPCstate[rollId]=tsosAtRPC;
		}	      
	      }
	    }
	  }
	}
      }

      //Efficiency      
      std::map<RPCDetId,TrajectoryStateOnSurface>::iterator irpc;
      for (irpc=RPCstate.begin(); irpc!=RPCstate.end();irpc++){
	RPCDetId rollId=irpc->first;
	const RPCRoll* rollasociated = rpcGeo->roll(rollId);
	TrajectoryStateOnSurface tsosAtRoll = RPCstate[rollId];
	
	const float stripPredicted =rollasociated->strip(LocalPoint(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y(),0.));
	const float xextrap = tsosAtRoll.localPosition().x();
	
	totalcounter[0]++;
	buff=counter[0];
	buff[rollId]++;
	counter[0]=buff;


	RPCGeomServ RPCname(rollId);
	std::string nameRoll = RPCname.name();
	
	_idList.push_back(nameRoll);
	char detUnitLabel[128];
	sprintf(detUnitLabel ,"%s",nameRoll.c_str());
	sprintf(layerLabel ,"%s",nameRoll.c_str());
	std::map<std::string, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(nameRoll);
	if (meItr == meCollection.end()){
	  meCollection[nameRoll] = bookDetUnitTrackEff(rollId,iSetup);
	}
	std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
	
	sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
	meMap[meIdTrack]->Fill(stripPredicted);

	Run=iEvent.id().run();
	aTime=iEvent.time().value();	 
	
	RPCRecHitCollection::range rpcRecHitRange = rpcHits->get(rollasociated->id());
	RPCRecHitCollection::const_iterator recIt;	  
	
	float res=0.;
	
	for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
	  LocalPoint rhitlocal = (*recIt).localPosition();
	  double rhitpos = rhitlocal.x();  
	  
	  LocalError RecError = (*recIt).localPositionError();
	  double sigmaRec = RecError.xx();
	  res = (double)(xextrap - rhitpos);
	  
	  sprintf(meIdRPC,"ClusterSize_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill((*recIt).clusterSize());
	  
	  sprintf(meIdRPC,"BunchX_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill((*recIt).BunchX());	 
	  
	  sprintf(meIdRPC,"Residuals_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(res);
	  
	  hGlobalRes->Fill(res);
	  hGlobalPull->Fill(res/sigmaRec);
	}
	
	int stripDetected=0;
	bool anycoincidence=false;
	
	RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rollasociated->id());	
	for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
	  stripDetected=digiIt->strip();
	  res = (float)(stripDetected) - stripPredicted;
	  if(fabs(res)<maxRes){
	    anycoincidence=true;
	    break;
	  }
	}
	
	if(anycoincidence==true){
	  
	  std::cout<<"Good Match "<<"\t"<<"Residuals = "<<res<<"\t"<<nameRoll<<std::endl;
	  
	  totalcounter[1]++;
	  buff=counter[1];
	  buff[rollId]++;
	  counter[1]=buff;
	  
	  sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(stripPredicted);
	  
	}else if(anycoincidence==false){
	  if(res==0){
	    std::cout<<"No Fired "<<nameRoll<<std::endl;
	  }
	  if(res!=0){
	    std::cout<<"No Match "<<"\t"<<"Residuals = "<<res<<"\t"<<nameRoll<<std::endl;
	  }

	  totalcounter[2]++;
	  buff=counter[2];
	  buff[rollId]++;
	  counter[2]=buff;
	}
      }
    }
  }
}




void RPCEfficiencyFromTrack::beginJob(const edm::EventSetup&){

}

void RPCEfficiencyFromTrack::endJob(){ 
  std::map<RPCDetId, int> pred = counter[0];
  std::map<RPCDetId, int> obse = counter[1];
  std::map<RPCDetId, int> reje = counter[2];
  std::map<RPCDetId, int>::iterator irpc;
  int f=0;
  for (irpc=pred.begin(); irpc!=pred.end();irpc++){
    RPCDetId id=irpc->first;
    RPCGeomServ RPCname(id);
    std::string nameRoll = RPCname.name();
    std::string wheel;
    std::string rpc;
    std::string partition;

    int p=pred[id]; 
    int o=obse[id]; 
    int r=reje[id]; 
    assert(p==o+r);
   
    if(p!=0){
      float ef = float(o)/float(p); 
      float er = sqrt(ef*(1.-ef)/float(p));
      if(ef>0.){
	*effres << nameRoll <<"\t Eff = "<<ef*100.<<" % +/- "<<er*100.<<" %"<<"\t Run= "<<Run<<"\t"<<ctime(&aTime)<<" ";
	histoMean->Fill(ef*100.);
      }
    }
    else{
      *effres <<"No predictions in this file predicted=0"<<std::endl;
    }
    f++;
  }
  if(totalcounter[0]!=0){
    float tote = float(totalcounter[1])/float(totalcounter[0]);
    float totr = sqrt(tote*(1.-tote)/float(totalcounter[0]));
    std::cout<<"Total Eff = "<<tote<<" +/- "<<totr<<std::endl;
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
    char effIdRPC [128];

    sprintf(detUnitLabel ,"%s",(*meIt).c_str());
    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
    sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
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
