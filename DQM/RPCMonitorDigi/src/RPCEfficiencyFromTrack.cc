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

Double_t linearFX(Double_t *x, Double_t *par){
  Double_t y=0.;
  y = par[0]*(*x) + par[1];  
  return y;
}

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

  cosmic = iConfig.getParameter<bool>("AreCosmic");
  EffSaveRootFile  = iConfig.getUntrackedParameter<bool>("EffSaveRootFile", true); 
  EffSaveRootFileEventsInterval  = iConfig.getUntrackedParameter<int>("EffEventsInterval", 1000); 
  EffRootFileName  = iConfig.getUntrackedParameter<std::string>("EffRootFileName", "RPCEfficiencyFromTrack.root"); 
  TjInput  = iConfig.getUntrackedParameter<std::string>("trajectoryInput");
  RPCDataLabel = iConfig.getUntrackedParameter<std::string>("rpcRecHitLabel");
  wh  = iConfig.getUntrackedParameter<int>("wheel",1);
  thePropagatorName = iConfig.getParameter<std::string>("PropagatorName");
  thePropagator = 0;

  GlobalRootLabel= iConfig.getUntrackedParameter<std::string>("GlobalRootFileName","GlobalEfficiencyFromTrack.root");
  fOutputFile  = new TFile(GlobalRootLabel.c_str(), "RECREATE" );
  hRecPt = new TH1F("RecPt","ReconstructedPt",100,0.5,100.5);
  hGlobalRes = new TH1F("GlobalResiduals","GlobalRPCResiduals",50,-15.,15.);
  hGlobalPull = new TH1F("GlobalPull","GlobalRPCPull",50,-15.,15.);
  histoMean = new TH1F("MeanEfficincy","MeanEfficiency_vs_Ch",60,20.5,120.5);
  ExtrapError = new TH2F("ExtrapError","Extrapolation Error Distribution",201,0.,100.,201,0.,100.);

  EffGlob1 = new TH1F("GlobEfficiencySec1","Eff. vs. roll",20,0.5,20.5);
  EffGlob2 = new TH1F("GlobEfficiencySec2","Eff. vs. roll",20,0.5,20.5);
  EffGlob3 = new TH1F("GlobEfficiencySec3","Eff. vs. roll",20,0.5,20.5);
  EffGlob4 = new TH1F("GlobEfficiencySec4","Eff. vs. roll",20,0.5,20.5);
  EffGlob5 = new TH1F("GlobEfficiencySec5","Eff. vs. roll",20,0.5,20.5);
  EffGlob6 = new TH1F("GlobEfficiencySec6","Eff. vs. roll",20,0.5,20.5);
  EffGlob7 = new TH1F("GlobEfficiencySec7","Eff. vs. roll",20,0.5,20.5);
  EffGlob8 = new TH1F("GlobEfficiencySec8","Eff. vs. roll",20,0.5,20.5);
  EffGlob9 = new TH1F("GlobEfficiencySec9","Eff. vs. roll",20,0.5,20.5);
  EffGlob10 = new TH1F("GlobEfficiencySec10","Eff. vs. roll",20,0.5,20.5);
  EffGlob11 = new TH1F("GlobEfficiencySec11","Eff. vs. roll",20,0.5,20.5);
  EffGlob12 = new TH1F("GlobEfficiencySec12","Eff. vs. roll",20,0.5,20.5);

  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();
  _idList.clear(); 
  Run=0;
  effres = new ofstream("EfficiencyResults.dat");
}


RPCEfficiencyFromTrack::~RPCEfficiencyFromTrack(){
  effres->close();
  delete effres;
  fOutputFile->WriteTObject(EffGlob1);
  fOutputFile->WriteTObject(EffGlob2);
  fOutputFile->WriteTObject(EffGlob3);
  fOutputFile->WriteTObject(EffGlob4);
  fOutputFile->WriteTObject(EffGlob5);
  fOutputFile->WriteTObject(EffGlob6);
  fOutputFile->WriteTObject(EffGlob7);
  fOutputFile->WriteTObject(EffGlob8);
  fOutputFile->WriteTObject(EffGlob9);
  fOutputFile->WriteTObject(EffGlob10);
  fOutputFile->WriteTObject(EffGlob11);
  fOutputFile->WriteTObject(EffGlob12);

  fOutputFile->WriteTObject(hRecPt);
  fOutputFile->WriteTObject(hGlobalRes);
  fOutputFile->WriteTObject(hGlobalPull);
  fOutputFile->WriteTObject(histoMean);
  fOutputFile->WriteTObject(ExtrapError);
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
  int lay=0;

  std::vector<float> globalX,globalY,globalZ;    
  globalX.clear(); globalY.clear(); globalZ.clear();

  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
    reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 
    rollRec.clear();
    globalX.clear(); globalY.clear(); globalZ.clear();
    lay=0;
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
	  break;
	}

	//Barrel
	if(track.innermostMeasurementState().isValid() && MeasureBarrel==true && reg==0){
 	  std::vector< const RPCRoll*> rolhit = (ch->rolls());
	  for(std::vector<const RPCRoll*>::const_iterator itRoll = rolhit.begin();itRoll != rolhit.end(); ++itRoll){
	    RPCDetId rollId=(*itRoll)->id();
	    
	    RPCRecHitCollection::range rpcRecHitRange = rpcHits->get(rollId);
	    RPCRecHitCollection::const_iterator recIt;

	    const RPCRoll* rollasociated = dynamic_cast<const RPCRoll*>(rpcGeo->roll(rollId));
	    const BoundSurface& bSurface = rollasociated->surface();
	    
	    int recFound=0;
	    
	    for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
	      LocalPoint rhitlocal = (*recIt).localPosition();
	      const GlobalPoint rhitglob = bSurface.toGlobal(rhitlocal);	      
	      recFound++;
	    }
	    if(recFound==1){
	      for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
		LocalPoint rhitlocal = (*recIt).localPosition();
		const GlobalPoint rhitglob = bSurface.toGlobal(rhitlocal);
		if((*recIt).clusterSize()<3.){
		  globalX.push_back(rhitglob.x());
		  globalY.push_back(rhitglob.y());
		  globalZ.push_back(rhitglob.z());
		}
	      }
	      rollRec.push_back(rollId);	  
	    }
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

	    const RPCRoll* rollasociated = dynamic_cast<const RPCRoll*>(rpcGeo->roll(rollId));
	    const BoundSurface& bSurface = rollasociated->surface();
	    
	    int recFound=0;
	    for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
	      LocalPoint rhitlocal = (*recIt).localPosition();
	      const GlobalPoint rhitglob = bSurface.toGlobal(rhitlocal);
	      recFound++;
	    }
	    if(recFound==1){
	      for (recIt = rpcRecHitRange.first; recIt!=rpcRecHitRange.second; ++recIt){
		LocalPoint rhitlocal = (*recIt).localPosition();
		const GlobalPoint rhitglob = bSurface.toGlobal(rhitlocal);
		if((*recIt).clusterSize()<3.){
		  globalX.push_back(rhitglob.x());
		  globalY.push_back(rhitglob.y());
		  globalZ.push_back(rhitglob.z());
		}
	      }
	      rollRec.push_back(rollId);
	    }
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
    double chi2=0.;
    if(cosmic==true){
      char folder[128];
      sprintf(folder,"HistoXYFit_%d",static_cast<int>(iEvent.id().event()));
      TH1F* histoXYFit = new TH1F(folder,folder,1401,-700,700);
      
      sprintf(folder,"HistoYZFit_%d",static_cast<int>(iEvent.id().event()));
      TH1F* histoYZFit = new TH1F(folder,folder,1401,-700,700);
      
      for(unsigned int i = 0; i < globalX.size(); ++i){
	histoXYFit->Fill(globalX[i],globalY[i]);
      }
      for(unsigned int z = 0; z < globalX.size(); ++z){
	histoYZFit->Fill(globalY[z],globalZ[z]);
      }
      
      TF1 *func = new TF1("linearFX",linearFX,-700,700,2);
      func->SetParameters(0.,0.);
      func->SetParNames("angCoef","interc");
      
      histoXYFit->Fit("linearFX","r");
      chi2=func->GetChisquare();
    }

    std::cout<<"Normalized Chi2 track --> "<<track.normalizedChi2()<<std::endl;

    if((chi2<1. && chi2>0.) || (cosmic==false)){
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
	
	const BoundPlane& rpcPlane = rollasociated->surface();
	FreeTrajectoryState fState1 =track.initialFreeState();
	
	TrajectoryStateOnSurface tsosAtRoll = thePropagator->propagate(fState1,rpcPlane);
	
	if(tsosAtRoll.isValid() 
	   && fabs(tsosAtRoll.localPosition().z()) < 0.01 
	   && fabs(tsosAtRoll.localPosition().x()) < rsize 
	   && fabs(tsosAtRoll.localPosition().y()) < stripl*0.5){
	  
	  RPCGeomServ RPCname(rollId);
	  std::string nameRoll = RPCname.name();

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
	  
	  const float stripPredicted =rollasociated->strip(LocalPoint(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y(),0.));
	  double xextrap = tsosAtRoll.localPosition().x();
	
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

	    if(fabs(tsosAtRoll.localError().positionError().xx())<1. && fabs(tsosAtRoll.localError().positionError().yy())<1.){
	      double rpcPull = res/sigmaRec;
	      
	      sprintf(meIdRPC,"Residuals_%s",detUnitLabel);
	      meMap[meIdRPC]->Fill(res);
	      
	      sprintf(meIdRPC,"Residuals_VS_RecPt_%s",detUnitLabel);
	      meMap[meIdRPC]->Fill(tsosAtRoll.globalMomentum().perp(),res);
	      
	      sprintf(meIdRPC,"Residuals_VS_CLsize_%s",detUnitLabel);
	      meMap[meIdRPC]->Fill((*recIt).clusterSize(),res);

	      sprintf(meIdRPC,"LocalPull_%s",detUnitLabel);
	      meMap[meIdRPC]->Fill(rpcPull);

	      hGlobalRes->Fill(res);
	      hGlobalPull->Fill(res/sigmaRec);
	      hRecPt->Fill(tsosAtRoll.globalMomentum().perp());	     
	    }
	  }
	  	  
	  int stripDetected=0;
	  bool anycoincidence=false;

	  RPCDigiCollection::Range rpcRangeDigi=rpcDigis->get(rollasociated->id());	
	  for (RPCDigiCollection::const_iterator digiIt = rpcRangeDigi.first;digiIt!=rpcRangeDigi.second;++digiIt){
	    stripDetected=digiIt->strip();
	    res = (float)(stripDetected) - stripPredicted;
	    if(fabs(res)<maxRes){
	      anycoincidence=true;
	    }
	  }
	  	  
	  if(anycoincidence==true){
	    totalcounter[0]++;
	    buff=counter[0];
	    buff[rollId]++;
	    counter[0]=buff;

	    std::cout<<"Good Match "<<"\t"<<"Residuals = "<<res<<"\t"<<nameRoll<<std::endl;

	    totalcounter[1]++;
	    buff=counter[1];
	    buff[rollId]++;
	    counter[1]=buff;

	    ExtrapError->Fill(tsosAtRoll.localError().positionError().xx(),tsosAtRoll.localError().positionError().yy());

	    sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
	    meMap[meIdTrack]->Fill(stripPredicted);

	    sprintf(meIdRPC,"PredictedImpactPoint_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y());

	    sprintf(meIdRPC,"2DExtrapolation_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y());
	    
	    sprintf(meIdRPC,"RealDetectedOccupancy_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(stripDetected);
	    
	    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(stripPredicted);
	    
	  }else if(anycoincidence==false && fabs(tsosAtRoll.localError().positionError().xx())<1. && fabs(tsosAtRoll.localError().positionError().yy())<1.){
	    totalcounter[0]++;
	    buff=counter[0];
	    buff[rollId]++;
	    counter[0]=buff;

	    std::cout<<"No Match "<<"\t"<<"Residuals = "<<res<<"\t"<<nameRoll<<std::endl;

	    sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
	    meMap[meIdTrack]->Fill(stripPredicted);

	    sprintf(meIdRPC,"PredictedImpactPoint_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y());

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
}

void RPCEfficiencyFromTrack::endJob(){
  int index1=0,index2=0,index3=0,index4=0,index5=0,index6=0,index7=0,index8=0,index9=0,index10=0,index11=0,index12=0;

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
	if(id.sector()==1 && id.ring()==wh){
	  index1++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob1->SetBinContent(index1,ef*100.);
	  EffGlob1->SetBinError(index1,er*100.);
	  
	  EffGlob1->GetXaxis()->SetBinLabel(index1,camera);
	  EffGlob1->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==2 && id.ring()==wh){
	  index2++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob2->SetBinContent(index2,ef*100.);
	  EffGlob2->SetBinError(index2,er*100.);
	  
	  EffGlob2->GetXaxis()->SetBinLabel(index2,camera);
	  EffGlob2->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==3 && id.ring()==wh){
	  index3++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob3->SetBinContent(index3,ef*100.);
	  EffGlob3->SetBinError(index3,er*100.);
	  
	  EffGlob3->GetXaxis()->SetBinLabel(index3,camera);
	  EffGlob3->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==4 && id.ring()==wh){
	  index4++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob4->SetBinContent(index4,ef*100.);
	  EffGlob4->SetBinError(index4,er*100.);
	  
	  EffGlob4->GetXaxis()->SetBinLabel(index4,camera);
	  EffGlob4->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==5 && id.ring()==wh){
	  index5++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob5->SetBinContent(index5,ef*100.);
	  EffGlob5->SetBinError(index5,er*100.);
	  
	  EffGlob5->GetXaxis()->SetBinLabel(index5,camera);
	  EffGlob5->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==6 && id.ring()==wh){
	  index6++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob6->SetBinContent(index6,ef*100.);
	  EffGlob6->SetBinError(index6,er*100.);
	  
	  EffGlob6->GetXaxis()->SetBinLabel(index6,camera);
	  EffGlob6->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==7 && id.ring()==wh){
	  index7++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob7->SetBinContent(index7,ef*100.);
	  EffGlob7->SetBinError(index7,er*100.);
	  
	  EffGlob7->GetXaxis()->SetBinLabel(index7,camera);
	  EffGlob7->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==8 && id.ring()==wh){
	  index8++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob8->SetBinContent(index8,ef*100.);
	  EffGlob8->SetBinError(index8,er*100.);
	  
	  EffGlob8->GetXaxis()->SetBinLabel(index8,camera);
	  EffGlob8->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==9 && id.ring()==wh){
	  index9++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob9->SetBinContent(index9,ef*100.);
	  EffGlob9->SetBinError(index9,er*100.);
	  
	  EffGlob9->GetXaxis()->SetBinLabel(index9,camera);
	  EffGlob9->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==10 && id.ring()==wh){
	  index10++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob10->SetBinContent(index10,ef*100.);
	  EffGlob10->SetBinError(index10,er*100.);
	  
	  EffGlob10->GetXaxis()->SetBinLabel(index10,camera);
	  EffGlob10->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==11 && id.ring()==wh){
	  index11++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob11->SetBinContent(index11,ef*100.);
	  EffGlob11->SetBinError(index11,er*100.);
	  
	  EffGlob11->GetXaxis()->SetBinLabel(index11,camera);
	  EffGlob11->GetXaxis()->LabelsOption("v");
	}
	if(id.sector()==12 && id.ring()==wh){
	  index12++;
	  char cam[128];	
	  sprintf(cam,"%s",nameRoll.c_str());
	  TString camera = (TString)cam;
	  
	  EffGlob12->SetBinContent(index12,ef*100.);
	  EffGlob12->SetBinError(index12,er*100.);
	  
	  EffGlob12->GetXaxis()->SetBinLabel(index12,camera);
	  EffGlob12->GetXaxis()->LabelsOption("v");
	}

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
