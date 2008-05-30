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

  ExtrapErrorG = new TH2F("ExtrapErrorEffGood","Extrapolation Error Distribution for Efficiency cases",201,0.,100.,201,0.,100.);
  ExtrapErrorN = new TH2F("ExtrapErrorNoEff","Extrapolation Error Distribution for No Efficiency cases",201,0.,100.,201,0.,100.);

  chisquareEff= new TH1F("chi2_Eff","Chi2 Distribution for Efficiency cases",200,0.,50.);
  chisquareNoEff= new TH1F("chi2_NoEff","Chi2 Distribution for No Efficiency cases",200,0.,50.);

  EXPGlob1 = new TH1F("ExpW-2","Exp. vs. roll",210,0.5,210.5);
  EXPGlob2 = new TH1F("ExpW-1","Exp. vs. roll",210,0.5,210.5);
  EXPGlob3 = new TH1F("ExpW+0","Exp. vs. roll",210,0.5,210.5);
  EXPGlob4 = new TH1F("ExpW+1","Exp. vs. roll",210,0.5,210.5);
  EXPGlob5 = new TH1F("ExpW+2","Exp. vs. roll",210,0.5,210.5);

  RPCGlob1 = new TH1F("RpcW-2","Real. vs. roll",210,0.5,210.5);
  RPCGlob2 = new TH1F("RpcW-1","Real. vs. roll",210,0.5,210.5);
  RPCGlob3 = new TH1F("RpcW+0","Real. vs. roll",210,0.5,210.5);
  RPCGlob4 = new TH1F("RpcW+1","Real. vs. roll",210,0.5,210.5);
  RPCGlob5 = new TH1F("RpcW+2","Real. vs. roll",210,0.5,210.5);

  ChiEff = new TH1F("Eff_vs_Chi2","Eff. vs. Chi2",200,0.,50.);
  // get hold of back-end interface
  dbe = edm::Service<DQMStore>().operator->();
  _idList.clear(); 
  Run=0;
  effres = new ofstream("EfficiencyResults.dat");
}


RPCEfficiencyFromTrack::~RPCEfficiencyFromTrack(){
  effres->close();
  delete effres;

  fOutputFile->WriteTObject(EXPGlob1);
  fOutputFile->WriteTObject(EXPGlob2);
  fOutputFile->WriteTObject(EXPGlob3);
  fOutputFile->WriteTObject(EXPGlob4);
  fOutputFile->WriteTObject(EXPGlob5);

  fOutputFile->WriteTObject(RPCGlob1);
  fOutputFile->WriteTObject(RPCGlob2);
  fOutputFile->WriteTObject(RPCGlob3);
  fOutputFile->WriteTObject(RPCGlob4);
  fOutputFile->WriteTObject(RPCGlob5);

  fOutputFile->WriteTObject(hRecPt);
  fOutputFile->WriteTObject(hGlobalRes);
  fOutputFile->WriteTObject(hGlobalPull);
  fOutputFile->WriteTObject(histoMean);
  fOutputFile->WriteTObject(ExtrapErrorG);
  fOutputFile->WriteTObject(ExtrapErrorN);
  fOutputFile->WriteTObject(chisquareEff);
  fOutputFile->WriteTObject(chisquareNoEff);
  fOutputFile->WriteTObject(ChiEff);

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


  edm::ESHandle<DTGeometry> dtGeo;
  iSetup.get<MuonGeometryRecord>().get(dtGeo);
    
  edm::Handle<DTRecSegment4DCollection> all4DSegments;
  iEvent.getByLabel("dt4DSegments", all4DSegments);

  Handle<reco::TrackCollection> staTracks;
  iEvent.getByLabel(TjInput, staTracks);

  reco::TrackCollection::const_iterator staTrack;

  ESHandle<Propagator> prop;
  iSetup.get<TrackingComponentsRecord>().get(thePropagatorName, prop);
  thePropagator = prop->clone();
  thePropagator->setPropagationDirection(anyDirection);
  
  std::vector<RPCDetId> rollRec;
  rollRec.clear();
 
  std::vector<float> globalX,globalY,globalZ;    
  globalX.clear(); globalY.clear(); globalZ.clear();

  if(staTracks->size()<=2){
    for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack){
      reco::TransientTrack track(*staTrack,&*theMGField,theTrackingGeometry); 
      rollRec.clear();
      globalX.clear(); globalY.clear(); globalZ.clear();

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
		 && fabs(tsosAtRPC.localPosition().y()) < stripl*0.5 && tsosAtRPC.localError().positionError().xx()<1.){
		
		rollRec.push_back(rollId);
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
		  break;
		}		
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
		 && fabs(tsosAtRPC.localPosition().y()) < stripl*0.5 && tsosAtRPC.localError().positionError().xx()<1.){
		
		rollRec.push_back(rollId);
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
		  break;
		}		
	      }	      
	    }
	  }
	}
      }
      
      std::cout<<"Size roll   "<<rollRec.size()<<std::endl;
      
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
      
	const BoundPlane *rpcPlane = &(rollasociated->surface());            
	TrajectoryStateClosestToPoint tcp=track.impactPointTSCP();
	const FreeTrajectoryState &fTS=tcp.theState();
	const FreeTrajectoryState *FreeState = &fTS;
	TrajectoryStateOnSurface tsosAtRoll = thePropagator->propagate(*FreeState,*rpcPlane);
      
      
	if(tsosAtRoll.isValid()
	   && fabs(tsosAtRoll.localPosition().z()) < 0.01 
	   && fabs(tsosAtRoll.localPosition().x()) < rsize
	   && fabs(tsosAtRoll.localPosition().y()) < stripl*0.5){
	
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
	    meCollection[nameRoll] = bookDetUnitTrackEff(rollId);
	  }
	  std::map<std::string, MonitorElement*> meMap=meCollection[nameRoll];
	
	  sprintf(meIdTrack,"ExpectedOccupancyFromTrack_%s",detUnitLabel);
	  meMap[meIdTrack]->Fill(stripPredicted);
	
	  sprintf(meIdRPC,"PredictedImpactPoint_%s",detUnitLabel);
	  meMap[meIdRPC]->Fill(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y());
	
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
	  
	    chisquareEff->Fill(track.normalizedChi2());
	    ExtrapErrorG->Fill(tsosAtRoll.localError().positionError().xx(),tsosAtRoll.localError().positionError().yy());
	  
	    sprintf(meIdRPC,"2DExtrapolation_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(tsosAtRoll.localPosition().x(),tsosAtRoll.localPosition().y());
	  
	    sprintf(meIdRPC,"RealDetectedOccupancy_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(stripDetected);
	  
	    sprintf(meIdRPC,"RPCDataOccupancy_%s",detUnitLabel);
	    meMap[meIdRPC]->Fill(stripPredicted);
	  
	  }else if(anycoincidence==false){
	  
	    std::cout<<"No Match "<<"\t"<<"Residuals = "<<res<<"\t"<<nameRoll<<std::endl;
	  
	    totalcounter[2]++;
	    buff=counter[2];
	    buff[rollId]++;
	    counter[2]=buff;
	  }
	  if(anycoincidence==false && res!=0){
	    chisquareNoEff->Fill(track.normalizedChi2());
	    ExtrapErrorN->Fill(tsosAtRoll.localError().positionError().xx(),tsosAtRoll.localError().positionError().yy());
	  }
	}
      }
    }
  }
}




void RPCEfficiencyFromTrack::beginJob(const edm::EventSetup&){

}

void RPCEfficiencyFromTrack::endJob(){
  for(int a=0;a<200;a++){
    if(chisquareNoEff->GetBinContent(a)+chisquareEff->GetBinContent(a)!=0){
      double valNoEff=chisquareNoEff->GetBinContent(a);
      double valEff=chisquareEff->GetBinContent(a);
      double eff=valEff/(valEff+valNoEff);
      float erreff = sqrt(eff*(1-eff)/(valEff+valNoEff));
      ChiEff->SetBinContent(a,eff*100.);
      ChiEff->SetBinError(a,erreff*100.);
    }
  }
  int index1=0,index2=0,index3=0,index4=0,index5=0;

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

      if(id.region()==0 && id.ring()==-2){
	index1++;
	char cam[128];	
	sprintf(cam,"%s",nameRoll.c_str());
	TString camera = (TString)cam;
	  
	RPCGlob1->SetBinContent(index1,float(o));	  
	RPCGlob1->GetXaxis()->SetBinLabel(index1,camera);
	RPCGlob1->GetXaxis()->LabelsOption("v");

	EXPGlob1->SetBinContent(index1,float(p));	  
	EXPGlob1->GetXaxis()->SetBinLabel(index1,camera);
	EXPGlob1->GetXaxis()->LabelsOption("v");
      }
      if(id.region()==0 && id.ring()==-1){
	index2++;
	char cam[128];	
	sprintf(cam,"%s",nameRoll.c_str());
	TString camera = (TString)cam;
	  
	RPCGlob2->SetBinContent(index2,float(o));	  
	RPCGlob2->GetXaxis()->SetBinLabel(index2,camera);
	RPCGlob2->GetXaxis()->LabelsOption("v");

	EXPGlob2->SetBinContent(index2,float(p));	  
	EXPGlob2->GetXaxis()->SetBinLabel(index2,camera);
	EXPGlob2->GetXaxis()->LabelsOption("v");
      }
      if(id.region()==0 && id.ring()==0){
	index3++;
	char cam[128];	
	sprintf(cam,"%s",nameRoll.c_str());
	TString camera = (TString)cam;
	  
	RPCGlob3->SetBinContent(index3,float(o));	  
	RPCGlob3->GetXaxis()->SetBinLabel(index3,camera);
	RPCGlob3->GetXaxis()->LabelsOption("v");

	EXPGlob3->SetBinContent(index3,float(p));	  
	EXPGlob3->GetXaxis()->SetBinLabel(index3,camera);
	EXPGlob3->GetXaxis()->LabelsOption("v");
      }
      if(id.region()==0 && id.ring()==1){
	index4++;
	char cam[128];	
	sprintf(cam,"%s",nameRoll.c_str());
	TString camera = (TString)cam;
	  
	RPCGlob4->SetBinContent(index4,float(o));	  
	RPCGlob4->GetXaxis()->SetBinLabel(index4,camera);
	RPCGlob4->GetXaxis()->LabelsOption("v");

	EXPGlob4->SetBinContent(index4,float(p));	  
	EXPGlob4->GetXaxis()->SetBinLabel(index4,camera);
	EXPGlob4->GetXaxis()->LabelsOption("v");
      }
      if(id.region()==0 && id.ring()==2){
	index5++;
	char cam[128];	
	sprintf(cam,"%s",nameRoll.c_str());
	TString camera = (TString)cam;
	  
	RPCGlob5->SetBinContent(index5,float(o));	  
	RPCGlob5->GetXaxis()->SetBinLabel(index5,camera);
	RPCGlob5->GetXaxis()->LabelsOption("v");

	EXPGlob5->SetBinContent(index5,float(p));	  
	EXPGlob5->GetXaxis()->SetBinLabel(index5,camera);
	EXPGlob5->GetXaxis()->LabelsOption("v");
      }
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
