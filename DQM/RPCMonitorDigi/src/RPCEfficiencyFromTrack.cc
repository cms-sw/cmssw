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
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>

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
  ringSelection = iConfig.getParameter<int>("Ring");
  selectwheel = iConfig.getParameter<bool>("SelectWheel");
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
  Run=0;
  effres = new ofstream("EfficiencyResults.dat");
}


RPCEfficiencyFromTrack::~RPCEfficiencyFromTrack()
{
  effres->close();
  delete effres;

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

  edm::Handle<RPCRecHitCollection> rpcHits;
  iEvent.getByLabel(RPCDataLabel,rpcHits);

  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByLabel("muonRPCDigis", rpcDigis);

  ESHandle<MagneticField> theMGField;
  iSetup.get<IdealMagneticFieldRecord>().get(theMGField);
  
  ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
  iSetup.get<GlobalTrackingGeometryRecord>().get(theTrackingGeometry);

  Handle<reco::TrackCollection> staTracks;
  iEvent.getByLabel(TjInput, staTracks);

  reco::TrackCollection::const_iterator staTrack;

  ESHandle<Propagator> prop;
  iSetup.get<TrackingComponentsRecord>().get(thePropagatorName, prop);
  thePropagator = prop->clone();
  thePropagator->setPropagationDirection(anyDirection);

  std::vector<RPCDetId> rollRec;
  rollRec.clear();


  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 
    rollRec.clear();
    for (TrackingGeometry::DetContainer::const_iterator it=rpcGeo->dets().begin();it<rpcGeo->dets().end();it++){
      if( dynamic_cast< RPCChamber* >( *it ) != 0 ){
	RPCChamber* ch = dynamic_cast< RPCChamber* >( *it );
	int reg=0;
	int sec=0;
	int whe=0;
	std::vector< const RPCRoll*> roles = (ch->rolls());
	for(std::vector<const RPCRoll*>::const_iterator r = roles.begin();r != roles.end(); ++r){
	  RPCDetId rpcId = (*r)->id();
	  reg=rpcId.region();
	  whe=rpcId.ring();
	  sec=rpcId.sector();
	}

	//Barrel
	if(track.innermostMeasurementState().isValid() && MeasureBarrel==true && reg==0 && whe==0 && (sec==10 || sec==11)){
	  std::vector< const RPCRoll*> rolhit = (ch->rolls());
	  for(std::vector<const RPCRoll*>::const_iterator itRoll = rolhit.begin();itRoll != rolhit.end(); ++itRoll){
	    RPCDetId rollId=(*itRoll)->id();
	    RPCRecHitCollection::range rpcRecHitRange = rpcHits->get(rollId);
	    RPCRecHitCollection::const_iterator recIt;
	    int recFound=0;
	    for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
	      recFound++;
	    }
	    if(recFound>0)rollRec.push_back(rollId);
	  }
	  if(rollRec.size()==0){
	    for(std::vector<const RPCRoll*>::const_iterator itRoll = rolhit.begin();itRoll != rolhit.end(); ++itRoll){
	      RPCDetId rollId=(*itRoll)->id();
	      const BoundPlane* rpcPlane1 = &((*itRoll)->surface());
	      TrajectoryStateClosestToPoint tcp1=track.impactPointTSCP();
	      const FreeTrajectoryState& fS1=tcp1.theState();
	      const FreeTrajectoryState* fState1 = &fS1; 
	      TrajectoryStateOnSurface tsosAtRPC = thePropagator->propagate(*fState1,*rpcPlane1);
	      const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&((*itRoll)->topology()));
	      LocalPoint xmin = top_->localPosition(0.);
	      LocalPoint xmax = top_->localPosition((float)(*itRoll)->nstrips());
	      float rsize = fabs( xmax.x()-xmin.x() )*0.5;
	      float stripl = top_->stripLength();
	      if(tsosAtRPC.isValid()
		 && fabs(tsosAtRPC.localPosition().z()) < 0.01 
		 && fabs(tsosAtRPC.localPosition().x()) < rsize 
		 && fabs(tsosAtRPC.localPosition().y()) < stripl*0.5){
		rollRec.push_back(rollId);
	      }
	    }
	  }
	}
	
	
	//EndCap
	if(track.innermostMeasurementState().isValid() && reg!=0 && MeasureEndCap==true){
	  std::vector< const RPCRoll*> rolhit = (ch->rolls());
	  for(std::vector<const RPCRoll*>::const_iterator itRoll = rolhit.begin();itRoll != rolhit.end(); ++itRoll){
	    RPCDetId rollId=(*itRoll)->id();
	    RPCRecHitCollection::range rpcRecHitRange = rpcHits->get(rollId);
	    RPCRecHitCollection::const_iterator recIt;
	    int recFound=0;
	    for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
	      recFound++;
	    }
	    if(recFound>0)rollRec.push_back(rollId);
	  }
	  if(rollRec.size()==0){
	    for(std::vector<const RPCRoll*>::const_iterator itRoll = rolhit.begin();itRoll != rolhit.end(); ++itRoll){
	      RPCDetId rollId=(*itRoll)->id();
	      const BoundPlane* rpcPlane1 = &((*itRoll)->surface());
	      TrajectoryStateClosestToPoint tcp1=track.impactPointTSCP();
	      const FreeTrajectoryState& fS1=tcp1.theState();
	      const FreeTrajectoryState* fState1 = &fS1; 
	      TrajectoryStateOnSurface tsosAtRPC = thePropagator->propagate(*fState1,*rpcPlane1);
	      const TrapezoidalStripTopology* top_= dynamic_cast<const TrapezoidalStripTopology*> (&((*itRoll)->topology()));
	      LocalPoint xmin = top_->localPosition(0.);
	      LocalPoint xmax = top_->localPosition((float)(*itRoll)->nstrips());
	      float rsize = fabs( xmax.x()-xmin.x() )*0.5;
	      float stripl = top_->stripLength();
	      if(tsosAtRPC.isValid()
		 && fabs(tsosAtRPC.localPosition().z()) < 0.01 
		 && fabs(tsosAtRPC.localPosition().x()) < rsize 
		 && fabs(tsosAtRPC.localPosition().y()) < stripl*0.5){		  
		rollRec.push_back(rollId);
	      }
	    }
	  }
	}
      }
    }
    
    //Efficiency

    for (std::vector<RPCDetId>::iterator iteraRoll = rollRec.begin();iteraRoll != rollRec.end(); iteraRoll++){
      const RPCRoll* rollasociated = rpcGeo->roll(*iteraRoll);
      RPCDetId rollId = rollasociated->id();
      float rsize=0.;
      float stripl=0.;

      if(rollId.region()==0){
	const RectangularStripTopology* top_= dynamic_cast<const RectangularStripTopology*> (&(rollasociated->topology()));
	LocalPoint xmin = top_->localPosition(0.);
	LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
	rsize = fabs( xmax.x()-xmin.x() )*0.5;
	stripl = top_->stripLength();
      }      
      if(rollId.region()!=0){
	const TrapezoidalStripTopology* top_= dynamic_cast<const TrapezoidalStripTopology*> (&(rollasociated->topology()));
	LocalPoint xmin = top_->localPosition(0.);
	LocalPoint xmax = top_->localPosition((float)rollasociated->nstrips());
	rsize = fabs( xmax.x()-xmin.x() )*0.5;
	stripl = top_->stripLength();
      }

      const BoundPlane& rpcPlane1 = rollasociated->surface();
      TrajectoryStateClosestToPoint tcp1=track.impactPointTSCP();
      const FreeTrajectoryState& fS1=tcp1.theState();
      const FreeTrajectoryState* fState1 = &fS1;       

      TrajectoryStateOnSurface tsosAtRoll = thePropagator->propagate(*fState1,rpcPlane1);

      if(tsosAtRoll.isValid() 
	 && fabs(tsosAtRoll.localPosition().z()) < 0.01 
	 && fabs(tsosAtRoll.localPosition().x()) < rsize 
	 && fabs(tsosAtRoll.localPosition().y()) < stripl*0.5 ){

	RPCGeomServ RPCname(rollId);
	std::string nameRoll = RPCname.name();
	
	if(rollId.region()==0){
	  int first = nameRoll.find("W");
	  int second = nameRoll.substr(first,nameRoll.npos).find("/");
	  std::string wheel=nameRoll.substr(first,second);		
	  first = nameRoll.find("/");
	  second = nameRoll.substr(first,nameRoll.npos).rfind("/");
	  std::string rpc=nameRoll.substr(first+1,second-1);		
	  first = nameRoll.rfind("/");
	  std::string partition=nameRoll.substr(first+1);
	    nameRoll=wheel+"_"+rpc+"_"+partition;
	}
	
	_idList.push_back(nameRoll);
	char detUnitLabel[128];
	sprintf(detUnitLabel ,"%s",nameRoll.c_str());
	sprintf(layerLabel ,"%s",nameRoll.c_str());
	std::map<std::string, std::map<std::string,MonitorElement*> >::iterator meItr = meCollection.find(nameRoll);
	if (meItr == meCollection.end()){
	  meCollection[nameRoll] = bookDetUnitTrackEff(rollId);
	}
	std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
	Run=iEvent.id().run();
	aTime=iEvent.time().value();
	totalcounter[0]++;
	buff=counter[0];
	buff[rollId]++;
	counter[0]=buff;
	
	const float stripPredicted =rollasociated->strip(LocalPoint(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y(),0.));
	double xextrap = tsosAtRoll.localPosition().x();
	
	sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
	meMap[meIdTrack]->Fill(stripPredicted);

	sprintf(meIdRPC,"PredictedImpactPoint_%s",detUnitLabel);
	meMap[meIdRPC]->Fill(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y());
	
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
	
	float res=0.;


	for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
	  LocalPoint rhitlocal = (*recIt).localPosition();
	  double rhitpos = rhitlocal.x();  
	  int stripDetected = (int)(rollasociated->strip(rhitlocal));
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
	
	int rpos=0;
	if(ResVec.size()==1){
	  res = ResVec[0];
	  for(unsigned int rs=0;rs<ResVec.size();rs++){
	    if(fabs(ResVec[rs]) < fabs(res)){
	      res = ResVec[rs];
	      rpos=rs;
	    }
	  }	  

	  double xtsosErr = tsosAtRoll.localError().positionError().xx();
	  double rpcPull = res/sqrt(RecErr[rpos]*RecErr[rpos]+xtsosErr*xtsosErr);

	  sprintf(meIdRPC,"Residuals_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(res);
	  sprintf(meIdRPC,"Residuals_VS_RecPt_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(tsosAtRoll.globalMomentum().perp(),res);
	  
	  hGlobalRes->Fill(res);
	  hGlobalPull->Fill(res/RecErr[rpos]);
	  hRecPt->Fill(tsosAtRoll.globalMomentum().perp());

	  sprintf(meIdRPC,"LocalPull_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(rpcPull);
	}


	int stripDetected=0;
	RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rollasociated->id());	
	for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
	  stripDetected=digiIt->strip();
	  res = (float)(stripDetected) - stripPredicted;
	  if(fabs(res)<maxRes){
	    anycoincidence=true;
	    std::cout<<"Good Match "<<"\t"<<"Residuals = "<<res<<"\t"<<nameRoll<<std::endl;
	    break;
	  }
	  else{
	    anycoincidence=false;
	    std::cout<<"No Match "<<"\t"<<"Residuals = "<<res<<"\t"<<nameRoll<<std::endl;
	    continue;
	  }
	}
	

	if(anycoincidence==true){
	  totalcounter[1]++;
	  buff=counter[1];
	  buff[rollId]++;
	  counter[1]=buff;

	  sprintf(meIdRPC,"2DExtrapolation_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y());

	  sprintf(meIdRPC,"RealDetectedOccupancy_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(stripDetected);
	  
	  sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(stripPredicted);
	    
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



void RPCEfficiencyFromTrack::beginJob(const edm::EventSetup&)
{
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

    if(id.region()==0){
      int first = nameRoll.find("W");
      int second = nameRoll.substr(first,nameRoll.npos).find("/");
      wheel=nameRoll.substr(first,second);	
      first = nameRoll.find("/");
      second = nameRoll.substr(first,nameRoll.npos).rfind("/");
      rpc=nameRoll.substr(first+1,second-1);
      first = nameRoll.rfind("/");
      partition=nameRoll.substr(first+1);
      nameRoll=wheel+"_"+rpc+"_"+partition;
    }

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
    char meIdImpact[128];
    char meIdRPC_2D[128];
    char effIdRPC_2D[128];

    sprintf(detUnitLabel ,"%s",(*meIt).c_str());
    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
    sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
    sprintf(meIdRes,"Residuals_%s",detUnitLabel);
    sprintf(effIdRPC,"EfficienyFromTrackExtrapolation_%s",detUnitLabel);
    sprintf(meIdRPC_2D,"2DExtrapolation_%s",detUnitLabel);
    sprintf(meIdImpact,"PredictedImpactPoint_%s",detUnitLabel);
    sprintf(effIdRPC_2D,"EfficienyFromTrack2DExtrapolation_%s",detUnitLabel);
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
