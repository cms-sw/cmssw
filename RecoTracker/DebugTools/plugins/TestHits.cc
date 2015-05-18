#include "RecoTracker/DebugTools/interface/TestHits.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrackerRecHit2D/interface/TrackingRecHitLessFromGlobalPosition.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include <TDirectory.h>
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"

#include "TrackingTools/TrackFitters/interface/TrajectoryFitter.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

typedef TrajectoryStateOnSurface TSOS;
typedef TransientTrackingRecHit::ConstRecHitPointer CTTRHp;
using namespace std;
using namespace edm;

TestHits::TestHits(const edm::ParameterSet& iConfig):
  trackerHitAssociatorConfig_(consumesCollector()) {
  LogTrace("TestHits") << iConfig<< std::endl;
  propagatorName = iConfig.getParameter<std::string>("Propagator");   
  builderName = iConfig.getParameter<std::string>("TTRHBuilder");   
  srcName = iConfig.getParameter<std::string>("src");   
  fname = iConfig.getParameter<std::string>("Fitter");
  mineta = iConfig.getParameter<double>("mineta");
  maxeta = iConfig.getParameter<double>("maxeta");
}

TestHits::~TestHits(){}

void TestHits::beginRun(edm::Run & run, const edm::EventSetup& iSetup)
{
 
  iSetup.get<TrackerDigiGeometryRecord>().get(theG);
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);  
  iSetup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);
  iSetup.get<TransientRecHitRecord>().get(builderName,theBuilder);
  iSetup.get<TrajectoryFitter::Record>().get(fname, fit);
 
  file = new TFile("testhits.root","recreate");
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      title.str("");
      title << "PullGP_X_" << i+1 << "-" << j+1 << "_ts";
      hPullGP_X_ts[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGP_Y_" << i+1 << "-" << j+1 << "_ts";
      hPullGP_Y_ts[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGP_Z_" << i+1 << "-" << j+1 << "_ts";
      hPullGP_Z_ts[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "Chi2Increment_" << i+1 << "-" << j+1;
      hChi2Increment[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,0,100);

      title.str("");
      title << "PullGM_X_" << i+1 << "-" << j+1 << "_ts";
      hPullGM_X_ts[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGM_Y_" << i+1 << "-" << j+1 << "_ts";
      hPullGM_Y_ts[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGM_Z_" << i+1 << "-" << j+1 << "_ts";
      hPullGM_Z_ts[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

      title.str("");
      title << "PullGP_X_" << i+1 << "-" << j+1 << "_tr";
      hPullGP_X_tr[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGP_Y_" << i+1 << "-" << j+1 << "_tr";
      hPullGP_Y_tr[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGP_Z_" << i+1 << "-" << j+1 << "_tr";
      hPullGP_Z_tr[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

      title.str("");
      title << "PullGP_X_" << i+1 << "-" << j+1 << "_rs";
      hPullGP_X_rs[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGP_Y_" << i+1 << "-" << j+1 << "_rs";
      hPullGP_Y_rs[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGP_Z_" << i+1 << "-" << j+1 << "_rs";
      hPullGP_Z_rs[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

      if ( ((i==2||i==4)&&(j==0||j==1)) || (i==3||i==5) ){
	//mono
	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGP_X_ts_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGP_Y_ts_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGP_Z_ts_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

	title.str("");
	title << "PullGM_X_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGM_X_ts_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGM_Y_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGM_Y_ts_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGM_Z_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGM_Z_ts_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_tr_mono";
	hPullGP_X_tr_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_tr_mono";
	hPullGP_Y_tr_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_tr_mono";
	hPullGP_Z_tr_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_rs_mono";
	hPullGP_X_rs_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_rs_mono";
	hPullGP_Y_rs_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_rs_mono";
	hPullGP_Z_rs_mono[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

	//stereo
	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGP_X_ts_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGP_Y_ts_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGP_Z_ts_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

	title.str("");
	title << "PullGM_X_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGM_X_ts_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGM_Y_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGM_Y_ts_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGM_Z_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGM_Z_ts_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_tr_stereo";
	hPullGP_X_tr_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_tr_stereo";
	hPullGP_Y_tr_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_tr_stereo";
	hPullGP_Z_tr_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);

	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_rs_stereo";
	hPullGP_X_rs_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_rs_stereo";
	hPullGP_Y_rs_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_rs_stereo";
	hPullGP_Z_rs_stereo[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      }
    }
  hTotChi2Increment = new TH1F("TotChi2Increment","TotChi2Increment",1000,0,100);
  hProcess_vs_Chi2  = new TH2F("Process_vs_Chi2","Process_vs_Chi2",1000,0,100,17,-0.5,16.5);  
  hClsize_vs_Chi2  = new TH2F("Clsize_vs_Chi2","Clsize_vs_Chi2",1000,0,100,17,-0.5,16.5);
}


void TestHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  iSetup.get<TrackerTopologyRcd>().get(tTopo);


  LogTrace("TestHits") << "\nnew event";

  iEvent.getByLabel(srcName,theTCCollection ); 
  TrackerHitAssociator hitAssociator(iEvent, trackerHitAssociatorConfig_);

  TrajectoryStateCombiner combiner;

  for (TrackCandidateCollection::const_iterator i=theTCCollection->begin(); i!=theTCCollection->end();i++){

    LogTrace("TestHits") << "\n*****************new candidate*****************" << std::endl;
      
    const TrackCandidate * theTC = &(*i);
    PTrajectoryStateOnDet state = theTC->trajectoryStateOnDet();
    const TrackCandidate::range& recHitVec=theTC->recHits();

    //convert PTrajectoryStateOnDet to TrajectoryStateOnSurface
    
    
    DetId  detId(state.detId());
    TrajectoryStateOnSurface theTSOS=
      trajectoryStateTransform::transientState(state, &(theG->idToDet(detId)->surface()),theMF.product());

    if (theTSOS.globalMomentum().eta()>maxeta || theTSOS.globalMomentum().eta()<mineta) continue;
    
    //convert the TrackingRecHit vector to a TransientTrackingRecHit vector
    TransientTrackingRecHit::RecHitContainer hits;
    
    for (edm::OwnVector<TrackingRecHit>::const_iterator i=recHitVec.first;
	 i!=recHitVec.second; i++){
      hits.push_back(theBuilder->build(&(*i) ));
    }

    //call the fitter
    std::vector<Trajectory> result = fit->fit(theTC->seed(), hits, theTSOS);
    if (result.size()==0) continue;
    std::vector<TrajectoryMeasurement> vtm = result[0].measurements();
    double tchi2 = 0;

    TSOS lastState = theTSOS;
    for (std::vector<TrajectoryMeasurement>::iterator tm=vtm.begin(); tm!=vtm.end();tm++){

      TransientTrackingRecHit::ConstRecHitPointer rhit = tm->recHit();
      if ((rhit)->isValid()==0&&rhit->det()!=0) continue;
      LogTrace("TestHits") << "*****************new hit*****************" ;

      int subdetId = rhit->det()->geographicalId().subdetId();
      DetId id = rhit->det()->geographicalId();
      int layerId  = tTopo->layer(id);
      LogTrace("TestHits") << "subdetId=" << subdetId << " layerId=" << layerId ;

      double delta = 99999;
      LocalPoint rhitLPv = rhit->localPosition();

      std::vector<PSimHit> assSimHits = hitAssociator.associateHit(*(rhit->hit()));
      if (assSimHits.size()==0) continue;
      PSimHit shit;
      for(std::vector<PSimHit>::const_iterator m=assSimHits.begin(); m<assSimHits.end(); m++){
	if ((m->localPosition()-rhitLPv).mag()<delta) {
	  shit=*m;
	  delta = (m->localPosition()-rhitLPv).mag();
	}
      }

      TSOS currentState = tm->forwardPredictedState();
      if (tm->backwardPredictedState().isValid()) 
	currentState = combiner(tm->backwardPredictedState(), tm->forwardPredictedState());
      TSOS updatedState = tm->updatedState();
      tchi2+=tm->estimate();

      //plot chi2 increment
      double chi2increment = tm->estimate();
      LogTrace("TestHits") << "tm->estimate()=" << tm->estimate();
      title.str("");
      title << "Chi2Increment_" << subdetId << "-" << layerId;
      hChi2Increment[title.str()]->Fill( chi2increment );
      hTotChi2Increment->Fill( chi2increment );
      hProcess_vs_Chi2->Fill( chi2increment, shit.processType() );
      if (dynamic_cast<const SiPixelRecHit*>(rhit->hit()))	
	hClsize_vs_Chi2->Fill( chi2increment, ((const SiPixelRecHit*)(rhit->hit()))->cluster()->size() );
      if (dynamic_cast<const SiStripRecHit2D*>(rhit->hit()))	
	hClsize_vs_Chi2->Fill( chi2increment, ((const SiStripRecHit2D*)(rhit->hit()))->cluster()->amplitudes().size() );
     
      //test hits
      const Surface * surf = &( (rhit)->det()->surface() );
      LocalVector shitLMom;
      LocalPoint shitLPos;
      if (dynamic_cast<const SiStripMatchedRecHit2D*>((rhit)->hit())) {
	double rechitmatchedx = rhit->localPosition().x();
	double rechitmatchedy = rhit->localPosition().y();
	double mindist = 999999;
	double distx, disty;
	std::pair<LocalPoint,LocalVector> closestPair;
	const StripGeomDetUnit* stripDet =(StripGeomDetUnit*) ((const GluedGeomDet *)(rhit)->det())->stereoDet();
	const BoundPlane& plane = (rhit)->det()->surface();
	for(std::vector<PSimHit>::const_iterator m=assSimHits.begin(); m<assSimHits.end(); m++) {
	  //project simhit;
	  std::pair<LocalPoint,LocalVector> hitPair = projectHit((*m),stripDet,plane);
	  distx = fabs(rechitmatchedx - hitPair.first.x());
	  disty = fabs(rechitmatchedy - hitPair.first.y());
	  double dist = distx*distx+disty*disty;
	  if(sqrt(dist)<mindist){
	    mindist = dist;
	    closestPair = hitPair;
	  }
	}
	shitLPos = closestPair.first;
	shitLMom = closestPair.second;
      } else {
	shitLPos = shit.localPosition();	
	shitLMom = shit.momentumAtEntry();
      }
      GlobalVector shitGMom = surf->toGlobal(shitLMom);
      GlobalPoint shitGPos = surf->toGlobal(shitLPos);

      GlobalVector tsosGMom = currentState.globalMomentum();
      GlobalError  tsosGMEr(currentState.cartesianError().matrix().Sub<AlgebraicSymMatrix33>(3,3));
      GlobalPoint  tsosGPos = currentState.globalPosition();
      GlobalError  tsosGPEr = currentState.cartesianError().position();

      GlobalPoint rhitGPos = (rhit)->globalPosition();
      GlobalError rhitGPEr = (rhit)->globalPositionError();

      double pullGPX_rs = (rhitGPos.x()-shitGPos.x())/sqrt(rhitGPEr.cxx());
      double pullGPY_rs = (rhitGPos.y()-shitGPos.y())/sqrt(rhitGPEr.cyy());
      double pullGPZ_rs = (rhitGPos.z()-shitGPos.z())/sqrt(rhitGPEr.czz());
      //double pullGPX_rs = (rhitGPos.x()-shitGPos.x());
      //double pullGPY_rs = (rhitGPos.y()-shitGPos.y());
      //double pullGPZ_rs = (rhitGPos.z()-shitGPos.z());
      
      LogTrace("TestHits") << "rs" << std::endl;
      LogVerbatim("TestHits") << "assSimHits.size()=" << assSimHits.size() ;
      LogVerbatim("TestHits") << "tsos globalPos   =" << tsosGPos ;
      LogVerbatim("TestHits") << "sim hit globalPos=" << shitGPos ;
      LogVerbatim("TestHits") << "rec hit globalPos=" << rhitGPos ;
      LogVerbatim("TestHits") << "geographicalId   =" << rhit->det()->geographicalId().rawId() ;
      LogVerbatim("TestHits") << "surface position =" << surf->position() ;

      title.str("");
      title << "PullGP_X_" << subdetId << "-" << layerId << "_rs";
      hPullGP_X_rs[title.str()]->Fill( pullGPX_rs );
      title.str("");
      title << "PullGP_Y_" << subdetId << "-" << layerId << "_rs";
      hPullGP_Y_rs[title.str()]->Fill( pullGPY_rs );
      title.str("");
      title << "PullGP_Z_" << subdetId << "-" << layerId << "_rs";
      hPullGP_Z_rs[title.str()]->Fill( pullGPZ_rs );

      double pullGPX_tr = (tsosGPos.x()-rhitGPos.x())/sqrt(tsosGPEr.cxx()+rhitGPEr.cxx());
      double pullGPY_tr = (tsosGPos.y()-rhitGPos.y())/sqrt(tsosGPEr.cyy()+rhitGPEr.cyy());
      double pullGPZ_tr = (tsosGPos.z()-rhitGPos.z())/sqrt(tsosGPEr.czz()+rhitGPEr.czz());
      //double pullGPX_tr = (tsosGPos.x()-rhitGPos.x());
      //double pullGPY_tr = (tsosGPos.y()-rhitGPos.y());
      //double pullGPZ_tr = (tsosGPos.z()-rhitGPos.z());

      LogTrace("TestHits") << "tr" << std::endl;

      title.str("");
      title << "PullGP_X_" << subdetId << "-" << layerId << "_tr";
      hPullGP_X_tr[title.str()]->Fill( pullGPX_tr );
      title.str("");
      title << "PullGP_Y_" << subdetId << "-" << layerId << "_tr";
      hPullGP_Y_tr[title.str()]->Fill( pullGPY_tr );
      title.str("");
      title << "PullGP_Z_" << subdetId << "-" << layerId << "_tr";
      hPullGP_Z_tr[title.str()]->Fill( pullGPZ_tr );

      double pullGPX_ts = (tsosGPos.x()-shitGPos.x())/sqrt(tsosGPEr.cxx());
      double pullGPY_ts = (tsosGPos.y()-shitGPos.y())/sqrt(tsosGPEr.cyy());
      double pullGPZ_ts = (tsosGPos.z()-shitGPos.z())/sqrt(tsosGPEr.czz());
      //double pullGPX_ts = (tsosGPos.x()-shitGPos.x());
      //double pullGPY_ts = (tsosGPos.y()-shitGPos.y());
      //double pullGPZ_ts = (tsosGPos.z()-shitGPos.z());

      LogTrace("TestHits") << "ts1" << std::endl;

      title.str("");
      title << "PullGP_X_" << subdetId << "-" << layerId << "_ts";
      hPullGP_X_ts[title.str()]->Fill( pullGPX_ts );
      title.str("");
      title << "PullGP_Y_" << subdetId << "-" << layerId << "_ts";
      hPullGP_Y_ts[title.str()]->Fill( pullGPY_ts );
      title.str("");
      title << "PullGP_Z_" << subdetId << "-" << layerId << "_ts";
      hPullGP_Z_ts[title.str()]->Fill( pullGPZ_ts );

      double pullGMX_ts = (tsosGMom.x()-shitGMom.x())/sqrt(tsosGMEr.cxx());
      double pullGMY_ts = (tsosGMom.y()-shitGMom.y())/sqrt(tsosGMEr.cyy());
      double pullGMZ_ts = (tsosGMom.z()-shitGMom.z())/sqrt(tsosGMEr.czz());
      //double pullGMX_ts = (tsosGMom.x()-shitGMom.x());
      //double pullGMY_ts = (tsosGMom.y()-shitGMom.y());
      //double pullGMZ_ts = (tsosGMom.z()-shitGMom.z());

      LogTrace("TestHits") << "ts2" << std::endl;

      title.str("");
      title << "PullGM_X_" << subdetId << "-" << layerId << "_ts";
      hPullGM_X_ts[title.str()]->Fill( pullGMX_ts );
      title.str("");
      title << "PullGM_Y_" << subdetId << "-" << layerId << "_ts";
      hPullGM_Y_ts[title.str()]->Fill( pullGMY_ts );
      title.str("");
      title << "PullGM_Z_" << subdetId << "-" << layerId << "_ts";
      hPullGM_Z_ts[title.str()]->Fill( pullGMZ_ts );

      if (dynamic_cast<const SiStripMatchedRecHit2D*>((rhit)->hit())) {
	//mono
	LogTrace("TestHits") << "MONO HIT" << std::endl;
        auto m = dynamic_cast<const SiStripMatchedRecHit2D*>((rhit)->hit())->monoHit();
	CTTRHp tMonoHit = theBuilder->build(&m);
	if (tMonoHit==0) continue;
	vector<PSimHit> assMonoSimHits = hitAssociator.associateHit(*tMonoHit->hit());
	if (assMonoSimHits.size()==0) continue;
	const PSimHit sMonoHit = *(assSimHits.begin());
	const Surface * monoSurf = &( tMonoHit->det()->surface() );
	if (monoSurf==0) continue;
	TSOS monoState = thePropagator->propagate(lastState,*monoSurf);
	if (monoState.isValid()==0) continue;

	LocalVector monoShitLMom = sMonoHit.momentumAtEntry();
	GlobalVector monoShitGMom = monoSurf->toGlobal(monoShitLMom);
	LocalPoint monoShitLPos = sMonoHit.localPosition();
	GlobalPoint monoShitGPos = monoSurf->toGlobal(monoShitLPos);

	GlobalVector monoTsosGMom = monoState.globalMomentum();
	GlobalError  monoTsosGMEr(monoState.cartesianError().matrix().Sub<AlgebraicSymMatrix33>(3,3));
	GlobalPoint  monoTsosGPos = monoState.globalPosition();
	GlobalError  monoTsosGPEr = monoState.cartesianError().position();

	GlobalPoint monoRhitGPos = tMonoHit->globalPosition();
	GlobalError monoRhitGPEr = tMonoHit->globalPositionError();

	double pullGPX_rs_mono = (monoRhitGPos.x()-monoShitGPos.x())/sqrt(monoRhitGPEr.cxx());
	double pullGPY_rs_mono = (monoRhitGPos.y()-monoShitGPos.y())/sqrt(monoRhitGPEr.cyy());
	double pullGPZ_rs_mono = (monoRhitGPos.z()-monoShitGPos.z())/sqrt(monoRhitGPEr.czz());
	//double pullGPX_rs_mono = (monoRhitGPos.x()-monoShitGPos.x());
	//double pullGPY_rs_mono = (monoRhitGPos.y()-monoShitGPos.y());
	//double pullGPZ_rs_mono = (monoRhitGPos.z()-monoShitGPos.z());

	title.str("");
	title << "PullGP_X_" << subdetId << "-" << layerId << "_rs_mono";
	hPullGP_X_rs_mono[title.str()]->Fill( pullGPX_rs_mono );
	title.str("");
	title << "PullGP_Y_" << subdetId << "-" << layerId << "_rs_mono";
	hPullGP_Y_rs_mono[title.str()]->Fill( pullGPY_rs_mono );
	title.str("");
	title << "PullGP_Z_" << subdetId << "-" << layerId << "_rs_mono";
	hPullGP_Z_rs_mono[title.str()]->Fill( pullGPZ_rs_mono );

	double pullGPX_tr_mono = (monoTsosGPos.x()-monoRhitGPos.x())/sqrt(monoTsosGPEr.cxx()+monoRhitGPEr.cxx());
	double pullGPY_tr_mono = (monoTsosGPos.y()-monoRhitGPos.y())/sqrt(monoTsosGPEr.cyy()+monoRhitGPEr.cyy());
	double pullGPZ_tr_mono = (monoTsosGPos.z()-monoRhitGPos.z())/sqrt(monoTsosGPEr.czz()+monoRhitGPEr.czz());
	//double pullGPX_tr_mono = (monoTsosGPos.x()-monoRhitGPos.x());
	//double pullGPY_tr_mono = (monoTsosGPos.y()-monoRhitGPos.y());
	//double pullGPZ_tr_mono = (monoTsosGPos.z()-monoRhitGPos.z());

	title.str("");
	title << "PullGP_X_" << subdetId << "-" << layerId << "_tr_mono";
	hPullGP_X_tr_mono[title.str()]->Fill( pullGPX_tr_mono );
	title.str("");
	title << "PullGP_Y_" << subdetId << "-" << layerId << "_tr_mono";
	hPullGP_Y_tr_mono[title.str()]->Fill( pullGPY_tr_mono );
	title.str("");
	title << "PullGP_Z_" << subdetId << "-" << layerId << "_tr_mono";
	hPullGP_Z_tr_mono[title.str()]->Fill( pullGPZ_tr_mono );

	double pullGPX_ts_mono = (monoTsosGPos.x()-monoShitGPos.x())/sqrt(monoTsosGPEr.cxx());
	double pullGPY_ts_mono = (monoTsosGPos.y()-monoShitGPos.y())/sqrt(monoTsosGPEr.cyy());
	double pullGPZ_ts_mono = (monoTsosGPos.z()-monoShitGPos.z())/sqrt(monoTsosGPEr.czz());
	//double pullGPX_ts_mono = (monoTsosGPos.x()-monoShitGPos.x());
	//double pullGPY_ts_mono = (monoTsosGPos.y()-monoShitGPos.y());
	//double pullGPZ_ts_mono = (monoTsosGPos.z()-monoShitGPos.z());

	title.str("");
	title << "PullGP_X_" << subdetId << "-" << layerId << "_ts_mono";
	hPullGP_X_ts_mono[title.str()]->Fill( pullGPX_ts_mono );
	title.str("");
	title << "PullGP_Y_" << subdetId << "-" << layerId << "_ts_mono";
	hPullGP_Y_ts_mono[title.str()]->Fill( pullGPY_ts_mono );
	title.str("");
	title << "PullGP_Z_" << subdetId << "-" << layerId << "_ts_mono";
	hPullGP_Z_ts_mono[title.str()]->Fill( pullGPZ_ts_mono );

	double pullGMX_ts_mono = (monoTsosGMom.x()-monoShitGMom.x())/sqrt(monoTsosGMEr.cxx());
	double pullGMY_ts_mono = (monoTsosGMom.y()-monoShitGMom.y())/sqrt(monoTsosGMEr.cyy());
	double pullGMZ_ts_mono = (monoTsosGMom.z()-monoShitGMom.z())/sqrt(monoTsosGMEr.czz());
	//double pullGMX_ts_mono = (monoTsosGMom.x()-monoShitGMom.x());
	//double pullGMY_ts_mono = (monoTsosGMom.y()-monoShitGMom.y());
	//double pullGMZ_ts_mono = (monoTsosGMom.z()-monoShitGMom.z());

	title.str("");
	title << "PullGM_X_" << subdetId << "-" << layerId << "_ts_mono";
	hPullGM_X_ts_mono[title.str()]->Fill( pullGMX_ts_mono );
	title.str("");
	title << "PullGM_Y_" << subdetId << "-" << layerId << "_ts_mono";
	hPullGM_Y_ts_mono[title.str()]->Fill( pullGMY_ts_mono );
	title.str("");
	title << "PullGM_Z_" << subdetId << "-" << layerId << "_ts_mono";
	hPullGM_Z_ts_mono[title.str()]->Fill( pullGMZ_ts_mono );

	//stereo
	LogTrace("TestHits") << "STEREO HIT" << std::endl;
        auto s = dynamic_cast<const SiStripMatchedRecHit2D*>((rhit)->hit())->stereoHit();
	CTTRHp tStereoHit = 
	  theBuilder->build(&s);
	if (tStereoHit==0) continue;
	vector<PSimHit> assStereoSimHits = hitAssociator.associateHit(*tStereoHit->hit());
	if (assStereoSimHits.size()==0) continue;
	const PSimHit sStereoHit = *(assSimHits.begin());
	const Surface * stereoSurf = &( tStereoHit->det()->surface() );
	if (stereoSurf==0) continue;
	TSOS stereoState = thePropagator->propagate(lastState,*stereoSurf);
	if (stereoState.isValid()==0) continue;

	LocalVector stereoShitLMom = sStereoHit.momentumAtEntry();
	GlobalVector stereoShitGMom = stereoSurf->toGlobal(stereoShitLMom);
	LocalPoint stereoShitLPos = sStereoHit.localPosition();
	GlobalPoint stereoShitGPos = stereoSurf->toGlobal(stereoShitLPos);

	GlobalVector stereoTsosGMom = stereoState.globalMomentum();
	GlobalError  stereoTsosGMEr(stereoState.cartesianError().matrix().Sub<AlgebraicSymMatrix33>(3,3));
	GlobalPoint  stereoTsosGPos = stereoState.globalPosition();
	GlobalError  stereoTsosGPEr = stereoState.cartesianError().position();

	GlobalPoint stereoRhitGPos = tStereoHit->globalPosition();
	GlobalError stereoRhitGPEr = tStereoHit->globalPositionError();

	double pullGPX_rs_stereo = (stereoRhitGPos.x()-stereoShitGPos.x())/sqrt(stereoRhitGPEr.cxx());
	double pullGPY_rs_stereo = (stereoRhitGPos.y()-stereoShitGPos.y())/sqrt(stereoRhitGPEr.cyy());
	double pullGPZ_rs_stereo = (stereoRhitGPos.z()-stereoShitGPos.z())/sqrt(stereoRhitGPEr.czz());
	//double pullGPX_rs_stereo = (stereoRhitGPos.x()-stereoShitGPos.x());
	//double pullGPY_rs_stereo = (stereoRhitGPos.y()-stereoShitGPos.y());
	//double pullGPZ_rs_stereo = (stereoRhitGPos.z()-stereoShitGPos.z());

	title.str("");
	title << "PullGP_X_" << subdetId << "-" << layerId << "_rs_stereo";
	hPullGP_X_rs_stereo[title.str()]->Fill( pullGPX_rs_stereo );
	title.str("");
	title << "PullGP_Y_" << subdetId << "-" << layerId << "_rs_stereo";
	hPullGP_Y_rs_stereo[title.str()]->Fill( pullGPY_rs_stereo );
	title.str("");
	title << "PullGP_Z_" << subdetId << "-" << layerId << "_rs_stereo";
	hPullGP_Z_rs_stereo[title.str()]->Fill( pullGPZ_rs_stereo );

	double pullGPX_tr_stereo = (stereoTsosGPos.x()-stereoRhitGPos.x())/sqrt(stereoTsosGPEr.cxx()+stereoRhitGPEr.cxx());
	double pullGPY_tr_stereo = (stereoTsosGPos.y()-stereoRhitGPos.y())/sqrt(stereoTsosGPEr.cyy()+stereoRhitGPEr.cyy());
	double pullGPZ_tr_stereo = (stereoTsosGPos.z()-stereoRhitGPos.z())/sqrt(stereoTsosGPEr.czz()+stereoRhitGPEr.czz());
	//double pullGPX_tr_stereo = (stereoTsosGPos.x()-stereoRhitGPos.x());
	//double pullGPY_tr_stereo = (stereoTsosGPos.y()-stereoRhitGPos.y());
	//double pullGPZ_tr_stereo = (stereoTsosGPos.z()-stereoRhitGPos.z());

	title.str("");
	title << "PullGP_X_" << subdetId << "-" << layerId << "_tr_stereo";
	hPullGP_X_tr_stereo[title.str()]->Fill( pullGPX_tr_stereo );
	title.str("");
	title << "PullGP_Y_" << subdetId << "-" << layerId << "_tr_stereo";
	hPullGP_Y_tr_stereo[title.str()]->Fill( pullGPY_tr_stereo );
	title.str("");
	title << "PullGP_Z_" << subdetId << "-" << layerId << "_tr_stereo";
	hPullGP_Z_tr_stereo[title.str()]->Fill( pullGPZ_tr_stereo );

	double pullGPX_ts_stereo = (stereoTsosGPos.x()-stereoShitGPos.x())/sqrt(stereoTsosGPEr.cxx());
	double pullGPY_ts_stereo = (stereoTsosGPos.y()-stereoShitGPos.y())/sqrt(stereoTsosGPEr.cyy());
	double pullGPZ_ts_stereo = (stereoTsosGPos.z()-stereoShitGPos.z())/sqrt(stereoTsosGPEr.czz());
	//double pullGPX_ts_stereo = (stereoTsosGPos.x()-stereoShitGPos.x());
	//double pullGPY_ts_stereo = (stereoTsosGPos.y()-stereoShitGPos.y());
	//double pullGPZ_ts_stereo = (stereoTsosGPos.z()-stereoShitGPos.z());

	title.str("");
	title << "PullGP_X_" << subdetId << "-" << layerId << "_ts_stereo";
	hPullGP_X_ts_stereo[title.str()]->Fill( pullGPX_ts_stereo );
	title.str("");
	title << "PullGP_Y_" << subdetId << "-" << layerId << "_ts_stereo";
	hPullGP_Y_ts_stereo[title.str()]->Fill( pullGPY_ts_stereo );
	title.str("");
	title << "PullGP_Z_" << subdetId << "-" << layerId << "_ts_stereo";
	hPullGP_Z_ts_stereo[title.str()]->Fill( pullGPZ_ts_stereo );

	double pullGMX_ts_stereo = (stereoTsosGMom.x()-stereoShitGMom.x())/sqrt(stereoTsosGMEr.cxx());
	double pullGMY_ts_stereo = (stereoTsosGMom.y()-stereoShitGMom.y())/sqrt(stereoTsosGMEr.cyy());
	double pullGMZ_ts_stereo = (stereoTsosGMom.z()-stereoShitGMom.z())/sqrt(stereoTsosGMEr.czz());
	//double pullGMX_ts_stereo = (stereoTsosGMom.x()-stereoShitGMom.x());
	//double pullGMY_ts_stereo = (stereoTsosGMom.y()-stereoShitGMom.y());
	//double pullGMZ_ts_stereo = (stereoTsosGMom.z()-stereoShitGMom.z());

	title.str("");
	title << "PullGM_X_" << subdetId << "-" << layerId << "_ts_stereo";
	hPullGM_X_ts_stereo[title.str()]->Fill( pullGMX_ts_stereo );
	title.str("");
	title << "PullGM_Y_" << subdetId << "-" << layerId << "_ts_stereo";
	hPullGM_Y_ts_stereo[title.str()]->Fill( pullGMY_ts_stereo );
	title.str("");
	title << "PullGM_Z_" << subdetId << "-" << layerId << "_ts_stereo";
	hPullGM_Z_ts_stereo[title.str()]->Fill( pullGMZ_ts_stereo );
      }    
      lastState = updatedState;
    }
    LogTrace("TestHits") << "traj chi2="  << tchi2 ;
    LogTrace("TestHits") << "track chi2=" << result[0].chiSquared() ;
  }
  LogTrace("TestHits") << "end of event" << std::endl;
}
//     TSOS lastState = theTSOS;
//     for (std::vector<TrajectoryMeasurement>::iterator tm=vtm.begin(); tm!=vtm.end();tm++){

//       TransientTrackingRecHit::ConstRecHitPointer rhit = tm->recHit();
//       if ((rhit)->isValid()==0&&rhit->det()!=0) continue;
    
// //     //test hits
// //     TSOS lastState, currentState, updatedState;
// //     updatedState=theTSOS;
// //     for (TransientTrackingRecHit::RecHitContainer::iterator rhit=hits.begin()+1;rhit!=hits.end();++rhit){

//       lastState=updatedState;

//       LogTrace("TestHits") << "new hit" << std::endl;

//       if ((*rhit)->isValid()==0) continue;

//       int subdetId = (*rhit)->det()->geographicalId().subdetId();
//       int layerId  = 0;
//       DetId id = (*rhit)->det()->geographicalId();
//       if (id.subdetId()==3) layerId = ((tTopo->tibLayer(id);
//       if (id.subdetId()==5) layerId = ((tTopo->tobLayer(id);
//       if (id.subdetId()==1) layerId = ((tTopo->pxbLayer(id);
//       if (id.subdetId()==4) layerId = ((tTopo->tidWheel(id);
//       if (id.subdetId()==6) layerId = ((tTopo->tecWheel(id);
//       if (id.subdetId()==2) layerId = ((tTopo->pxfDisk(id);
//       const Surface * surf = &( (*rhit)->det()->surface() );
//       currentState=thePropagator->propagate(lastState,*surf);	
//       if (currentState.isValid()==0) continue;
//       updatedState=theUpdator->update(currentState,**rhit);

//       double delta = 99999;
//       LocalPoint rhitLP = rhit->localPosition();

//       std::vector<PSimHit> assSimHits = hitAssociator.associateHit(*(*rhit)->hit());
//       if (assSimHits.size()==0) continue;
//       PSimHit shit;
//       for(std::vector<PSimHit>::const_iterator m=assSimHits.begin(); m<assSimHits.end(); m++){
// 	if ((*m-rhitLP).mag()<delta) {
// 	  shit=*m;
// 	  delta = (*m-rhitLP).mag();
// 	}
//       }

//       LocalVector shitLMom = shit.momentumAtEntry();
//       GlobalVector shitGMom = surf->toGlobal(shitLMom);
//       LocalPoint shitLPos = shit.localPosition();
//       GlobalPoint shitGPos = surf->toGlobal(shitLPos);

//       GlobalVector tsosGMom = currentState.globalMomentum();
//       GlobalError  tsosGMEr(currentState.cartesianError().matrix().Sub<AlgebraicSymMatrix33>(3,3));
//       GlobalPoint  tsosGPos = currentState.globalPosition();
//       GlobalError  tsosGPEr = currentState.cartesianError().position();

//       GlobalPoint rhitGPos = (*rhit)->globalPosition();
//       GlobalError rhitGPEr = (*rhit)->globalPositionError();

//       double pullGPX_rs = (rhitGPos.x()-shitGPos.x())/sqrt(rhitGPEr.cxx());
//       double pullGPY_rs = (rhitGPos.y()-shitGPos.y())/sqrt(rhitGPEr.cyy());
//       double pullGPZ_rs = (rhitGPos.z()-shitGPos.z())/sqrt(rhitGPEr.czz());
// //       double pullGPX_rs = (rhitGPos.x()-shitGPos.x())/*/sqrt(rhitGPEr.cxx())*/;
// //       double pullGPY_rs = (rhitGPos.y()-shitGPos.y())/*/sqrt(rhitGPEr.cyy())*/;
// //       double pullGPZ_rs = (rhitGPos.z()-shitGPos.z())/*/sqrt(rhitGPEr.czz())*/;

//       //plot chi2 increment
//       MeasurementExtractor me(currentState);
//       double chi2increment = computeChi2Increment(me,*rhit);
//       LogTrace("TestHits") << "chi2increment=" << chi2increment << std::endl;
//       title.str("");
//       title << "Chi2Increment_" << subdetId << "-" << layerId;
//       hChi2Increment[title.str()]->Fill( chi2increment );
//       hTotChi2Increment->Fill( chi2increment );
//       hChi2_vs_Process->Fill( chi2increment, shit.processType() );
//       if (dynamic_cast<const SiPixelRecHit*>((*rhit)->hit()))	
// 	hChi2_vs_clsize->Fill( chi2increment, ((const SiPixelRecHit*)(*rhit)->hit())->cluster()->size() );
//       if (dynamic_cast<const SiStripRecHit2D*>((*rhit)->hit()))	
// 	hChi2_vs_clsize->Fill( chi2increment, ((const SiStripRecHit2D*)(*rhit)->hit())->cluster()->amplitudes().size() );
      
//       LogTrace("TestHits") << "rs" << std::endl;

//       title.str("");
//       title << "PullGP_X_" << subdetId << "-" << layerId << "_rs";
//       hPullGP_X_rs[title.str()]->Fill( pullGPX_rs );
//       title.str("");
//       title << "PullGP_Y_" << subdetId << "-" << layerId << "_rs";
//       hPullGP_Y_rs[title.str()]->Fill( pullGPY_rs );
//       title.str("");
//       title << "PullGP_Z_" << subdetId << "-" << layerId << "_rs";
//       hPullGP_Z_rs[title.str()]->Fill( pullGPZ_rs );

//       double pullGPX_tr = (tsosGPos.x()-rhitGPos.x())/sqrt(tsosGPEr.cxx()+rhitGPEr.cxx());
//       double pullGPY_tr = (tsosGPos.y()-rhitGPos.y())/sqrt(tsosGPEr.cyy()+rhitGPEr.cyy());
//       double pullGPZ_tr = (tsosGPos.z()-rhitGPos.z())/sqrt(tsosGPEr.czz()+rhitGPEr.czz());
// //       double pullGPX_tr = (tsosGPos.x()-rhitGPos.x())/*/sqrt(tsosGPEr.cxx()+rhitGPEr.cxx())*/;
// //       double pullGPY_tr = (tsosGPos.y()-rhitGPos.y())/*/sqrt(tsosGPEr.cyy()+rhitGPEr.cyy())*/;
// //       double pullGPZ_tr = (tsosGPos.z()-rhitGPos.z())/*/sqrt(tsosGPEr.czz()+rhitGPEr.czz())*/;

//       LogTrace("TestHits") << "tr" << std::endl;

//       title.str("");
//       title << "PullGP_X_" << subdetId << "-" << layerId << "_tr";
//       hPullGP_X_tr[title.str()]->Fill( pullGPX_tr );
//       title.str("");
//       title << "PullGP_Y_" << subdetId << "-" << layerId << "_tr";
//       hPullGP_Y_tr[title.str()]->Fill( pullGPY_tr );
//       title.str("");
//       title << "PullGP_Z_" << subdetId << "-" << layerId << "_tr";
//       hPullGP_Z_tr[title.str()]->Fill( pullGPZ_tr );

//       double pullGPX_ts = (tsosGPos.x()-shitGPos.x())/sqrt(tsosGPEr.cxx());
//       double pullGPY_ts = (tsosGPos.y()-shitGPos.y())/sqrt(tsosGPEr.cyy());
//       double pullGPZ_ts = (tsosGPos.z()-shitGPos.z())/sqrt(tsosGPEr.czz());
// //       double pullGPX_ts = (tsosGPos.x()-shitGPos.x())/*/sqrt(tsosGPEr.cxx())*/;
// //       double pullGPY_ts = (tsosGPos.y()-shitGPos.y())/*/sqrt(tsosGPEr.cyy())*/;
// //       double pullGPZ_ts = (tsosGPos.z()-shitGPos.z())/*/sqrt(tsosGPEr.czz())*/;

//       LogTrace("TestHits") << "ts1" << std::endl;

//       title.str("");
//       title << "PullGP_X_" << subdetId << "-" << layerId << "_ts";
//       hPullGP_X_ts[title.str()]->Fill( pullGPX_ts );
//       title.str("");
//       title << "PullGP_Y_" << subdetId << "-" << layerId << "_ts";
//       hPullGP_Y_ts[title.str()]->Fill( pullGPY_ts );
//       title.str("");
//       title << "PullGP_Z_" << subdetId << "-" << layerId << "_ts";
//       hPullGP_Z_ts[title.str()]->Fill( pullGPZ_ts );

//       double pullGMX_ts = (tsosGMom.x()-shitGMom.x())/sqrt(tsosGMEr.cxx());
//       double pullGMY_ts = (tsosGMom.y()-shitGMom.y())/sqrt(tsosGMEr.cyy());
//       double pullGMZ_ts = (tsosGMom.z()-shitGMom.z())/sqrt(tsosGMEr.czz());
// //       double pullGMX_ts = (tsosGMom.x()-shitGMom.x())/*/sqrt(tsosGMEr.cxx())*/;
// //       double pullGMY_ts = (tsosGMom.y()-shitGMom.y())/*/sqrt(tsosGMEr.cyy())*/;
// //       double pullGMZ_ts = (tsosGMom.z()-shitGMom.z())/*/sqrt(tsosGMEr.czz())*/;

//       LogTrace("TestHits") << "ts2" << std::endl;

//       title.str("");
//       title << "PullGM_X_" << subdetId << "-" << layerId << "_ts";
//       hPullGM_X_ts[title.str()]->Fill( pullGMX_ts );
//       title.str("");
//       title << "PullGM_Y_" << subdetId << "-" << layerId << "_ts";
//       hPullGM_Y_ts[title.str()]->Fill( pullGMY_ts );
//       title.str("");
//       title << "PullGM_Z_" << subdetId << "-" << layerId << "_ts";
//       hPullGM_Z_ts[title.str()]->Fill( pullGMZ_ts );

//       if (dynamic_cast<const SiStripMatchedRecHit2D*>((*rhit)->hit())) {
// 	//mono
// 	LogTrace("TestHits") << "MONO HIT" << std::endl;
// 	CTTRHp tMonoHit = 
// 	  theBuilder->build(dynamic_cast<const SiStripMatchedRecHit2D*>((*rhit)->hit())->monoHit());
// 	if (tMonoHit==0) continue;
// 	vector<PSimHit> assMonoSimHits = hitAssociator.associateHit(*tMonoHit->hit());
// 	if (assMonoSimHits.size()==0) continue;
// 	const PSimHit sMonoHit = *(assSimHits.begin());
// 	const Surface * monoSurf = &( tMonoHit->det()->surface() );
// 	if (monoSurf==0) continue;
// 	TSOS monoState = thePropagator->propagate(lastState,*monoSurf);
// 	if (monoState.isValid()==0) continue;

// 	LocalVector monoShitLMom = sMonoHit.momentumAtEntry();
// 	GlobalVector monoShitGMom = monoSurf->toGlobal(monoShitLMom);
// 	LocalPoint monoShitLPos = sMonoHit.localPosition();
// 	GlobalPoint monoShitGPos = monoSurf->toGlobal(monoShitLPos);

// 	GlobalVector monoTsosGMom = monoState.globalMomentum();
// 	GlobalError  monoTsosGMEr(monoState.cartesianError().matrix().Sub<AlgebraicSymMatrix33>(3,3));
// 	GlobalPoint  monoTsosGPos = monoState.globalPosition();
// 	GlobalError  monoTsosGPEr = monoState.cartesianError().position();

// 	GlobalPoint monoRhitGPos = tMonoHit->globalPosition();
// 	GlobalError monoRhitGPEr = tMonoHit->globalPositionError();

// 	double pullGPX_rs_mono = (monoRhitGPos.x()-monoShitGPos.x())/sqrt(monoRhitGPEr.cxx());
// 	double pullGPY_rs_mono = (monoRhitGPos.y()-monoShitGPos.y())/sqrt(monoRhitGPEr.cyy());
// 	double pullGPZ_rs_mono = (monoRhitGPos.z()-monoShitGPos.z())/sqrt(monoRhitGPEr.czz());
// // 	double pullGPX_rs_mono = (monoRhitGPos.x()-monoShitGPos.x())/*/sqrt(monoRhitGPEr.cxx())*/;
// // 	double pullGPY_rs_mono = (monoRhitGPos.y()-monoShitGPos.y())/*/sqrt(monoRhitGPEr.cyy())*/;
// // 	double pullGPZ_rs_mono = (monoRhitGPos.z()-monoShitGPos.z())/*/sqrt(monoRhitGPEr.czz())*/;

// 	title.str("");
// 	title << "PullGP_X_" << subdetId << "-" << layerId << "_rs_mono";
// 	hPullGP_X_rs_mono[title.str()]->Fill( pullGPX_rs_mono );
// 	title.str("");
// 	title << "PullGP_Y_" << subdetId << "-" << layerId << "_rs_mono";
// 	hPullGP_Y_rs_mono[title.str()]->Fill( pullGPY_rs_mono );
// 	title.str("");
// 	title << "PullGP_Z_" << subdetId << "-" << layerId << "_rs_mono";
// 	hPullGP_Z_rs_mono[title.str()]->Fill( pullGPZ_rs_mono );

// 	double pullGPX_tr_mono = (monoTsosGPos.x()-monoRhitGPos.x())/sqrt(monoTsosGPEr.cxx()+monoRhitGPEr.cxx());
// 	double pullGPY_tr_mono = (monoTsosGPos.y()-monoRhitGPos.y())/sqrt(monoTsosGPEr.cyy()+monoRhitGPEr.cyy());
// 	double pullGPZ_tr_mono = (monoTsosGPos.z()-monoRhitGPos.z())/sqrt(monoTsosGPEr.czz()+monoRhitGPEr.czz());
// // 	double pullGPX_tr_mono = (monoTsosGPos.x()-monoRhitGPos.x())/*/sqrt(monoTsosGPEr.cxx()+monoRhitGPEr.cxx())*/;
// // 	double pullGPY_tr_mono = (monoTsosGPos.y()-monoRhitGPos.y())/*/sqrt(monoTsosGPEr.cyy()+monoRhitGPEr.cyy())*/;
// // 	double pullGPZ_tr_mono = (monoTsosGPos.z()-monoRhitGPos.z())/*/sqrt(monoTsosGPEr.czz()+monoRhitGPEr.czz())*/;

// 	title.str("");
// 	title << "PullGP_X_" << subdetId << "-" << layerId << "_tr_mono";
// 	hPullGP_X_tr_mono[title.str()]->Fill( pullGPX_tr_mono );
// 	title.str("");
// 	title << "PullGP_Y_" << subdetId << "-" << layerId << "_tr_mono";
// 	hPullGP_Y_tr_mono[title.str()]->Fill( pullGPY_tr_mono );
// 	title.str("");
// 	title << "PullGP_Z_" << subdetId << "-" << layerId << "_tr_mono";
// 	hPullGP_Z_tr_mono[title.str()]->Fill( pullGPZ_tr_mono );

// 	double pullGPX_ts_mono = (monoTsosGPos.x()-monoShitGPos.x())/sqrt(monoTsosGPEr.cxx());
// 	double pullGPY_ts_mono = (monoTsosGPos.y()-monoShitGPos.y())/sqrt(monoTsosGPEr.cyy());
// 	double pullGPZ_ts_mono = (monoTsosGPos.z()-monoShitGPos.z())/sqrt(monoTsosGPEr.czz());
// // 	double pullGPX_ts_mono = (monoTsosGPos.x()-monoShitGPos.x())/*/sqrt(monoTsosGPEr.cxx())*/;
// // 	double pullGPY_ts_mono = (monoTsosGPos.y()-monoShitGPos.y())/*/sqrt(monoTsosGPEr.cyy())*/;
// // 	double pullGPZ_ts_mono = (monoTsosGPos.z()-monoShitGPos.z())/*/sqrt(monoTsosGPEr.czz())*/;

// 	title.str("");
// 	title << "PullGP_X_" << subdetId << "-" << layerId << "_ts_mono";
// 	hPullGP_X_ts_mono[title.str()]->Fill( pullGPX_ts_mono );
// 	title.str("");
// 	title << "PullGP_Y_" << subdetId << "-" << layerId << "_ts_mono";
// 	hPullGP_Y_ts_mono[title.str()]->Fill( pullGPY_ts_mono );
// 	title.str("");
// 	title << "PullGP_Z_" << subdetId << "-" << layerId << "_ts_mono";
// 	hPullGP_Z_ts_mono[title.str()]->Fill( pullGPZ_ts_mono );

// 	double pullGMX_ts_mono = (monoTsosGMom.x()-monoShitGMom.x())/sqrt(monoTsosGMEr.cxx());
// 	double pullGMY_ts_mono = (monoTsosGMom.y()-monoShitGMom.y())/sqrt(monoTsosGMEr.cyy());
// 	double pullGMZ_ts_mono = (monoTsosGMom.z()-monoShitGMom.z())/sqrt(monoTsosGMEr.czz());
// // 	double pullGMX_ts_mono = (monoTsosGMom.x()-monoShitGMom.x())/*/sqrt(monoTsosGMEr.cxx())*/;
// // 	double pullGMY_ts_mono = (monoTsosGMom.y()-monoShitGMom.y())/*/sqrt(monoTsosGMEr.cyy())*/;
// // 	double pullGMZ_ts_mono = (monoTsosGMom.z()-monoShitGMom.z())/*/sqrt(monoTsosGMEr.czz())*/;

// 	title.str("");
// 	title << "PullGM_X_" << subdetId << "-" << layerId << "_ts_mono";
// 	hPullGM_X_ts_mono[title.str()]->Fill( pullGMX_ts_mono );
// 	title.str("");
// 	title << "PullGM_Y_" << subdetId << "-" << layerId << "_ts_mono";
// 	hPullGM_Y_ts_mono[title.str()]->Fill( pullGMY_ts_mono );
// 	title.str("");
// 	title << "PullGM_Z_" << subdetId << "-" << layerId << "_ts_mono";
// 	hPullGM_Z_ts_mono[title.str()]->Fill( pullGMZ_ts_mono );

// 	//stereo
// 	LogTrace("TestHits") << "STEREO HIT" << std::endl;
// 	CTTRHp tStereoHit = 
// 	  theBuilder->build(dynamic_cast<const SiStripMatchedRecHit2D*>((*rhit)->hit())->stereoHit());
// 	if (tStereoHit==0) continue;
// 	vector<PSimHit> assStereoSimHits = hitAssociator.associateHit(*tStereoHit->hit());
// 	if (assStereoSimHits.size()==0) continue;
// 	const PSimHit sStereoHit = *(assSimHits.begin());
// 	const Surface * stereoSurf = &( tStereoHit->det()->surface() );
// 	if (stereoSurf==0) continue;
// 	TSOS stereoState = thePropagator->propagate(lastState,*stereoSurf);
// 	if (stereoState.isValid()==0) continue;

// 	LocalVector stereoShitLMom = sStereoHit.momentumAtEntry();
// 	GlobalVector stereoShitGMom = stereoSurf->toGlobal(stereoShitLMom);
// 	LocalPoint stereoShitLPos = sStereoHit.localPosition();
// 	GlobalPoint stereoShitGPos = stereoSurf->toGlobal(stereoShitLPos);

// 	GlobalVector stereoTsosGMom = stereoState.globalMomentum();
// 	GlobalError  stereoTsosGMEr(stereoState.cartesianError().matrix().Sub<AlgebraicSymMatrix33>(3,3));
// 	GlobalPoint  stereoTsosGPos = stereoState.globalPosition();
// 	GlobalError  stereoTsosGPEr = stereoState.cartesianError().position();

// 	GlobalPoint stereoRhitGPos = tStereoHit->globalPosition();
// 	GlobalError stereoRhitGPEr = tStereoHit->globalPositionError();

// 	double pullGPX_rs_stereo = (stereoRhitGPos.x()-stereoShitGPos.x())/sqrt(stereoRhitGPEr.cxx());
// 	double pullGPY_rs_stereo = (stereoRhitGPos.y()-stereoShitGPos.y())/sqrt(stereoRhitGPEr.cyy());
// 	double pullGPZ_rs_stereo = (stereoRhitGPos.z()-stereoShitGPos.z())/sqrt(stereoRhitGPEr.czz());
// // 	double pullGPX_rs_stereo = (stereoRhitGPos.x()-stereoShitGPos.x())/*/sqrt(stereoRhitGPEr.cxx())*/;
// // 	double pullGPY_rs_stereo = (stereoRhitGPos.y()-stereoShitGPos.y())/*/sqrt(stereoRhitGPEr.cyy())*/;
// // 	double pullGPZ_rs_stereo = (stereoRhitGPos.z()-stereoShitGPos.z())/*/sqrt(stereoRhitGPEr.czz())*/;

// 	title.str("");
// 	title << "PullGP_X_" << subdetId << "-" << layerId << "_rs_stereo";
// 	hPullGP_X_rs_stereo[title.str()]->Fill( pullGPX_rs_stereo );
// 	title.str("");
// 	title << "PullGP_Y_" << subdetId << "-" << layerId << "_rs_stereo";
// 	hPullGP_Y_rs_stereo[title.str()]->Fill( pullGPY_rs_stereo );
// 	title.str("");
// 	title << "PullGP_Z_" << subdetId << "-" << layerId << "_rs_stereo";
// 	hPullGP_Z_rs_stereo[title.str()]->Fill( pullGPZ_rs_stereo );

// 	double pullGPX_tr_stereo = (stereoTsosGPos.x()-stereoRhitGPos.x())/sqrt(stereoTsosGPEr.cxx()+stereoRhitGPEr.cxx());
// 	double pullGPY_tr_stereo = (stereoTsosGPos.y()-stereoRhitGPos.y())/sqrt(stereoTsosGPEr.cyy()+stereoRhitGPEr.cyy());
// 	double pullGPZ_tr_stereo = (stereoTsosGPos.z()-stereoRhitGPos.z())/sqrt(stereoTsosGPEr.czz()+stereoRhitGPEr.czz());
// // 	double pullGPX_tr_stereo = (stereoTsosGPos.x()-stereoRhitGPos.x())/*/sqrt(stereoTsosGPEr.cxx()+stereoRhitGPEr.cxx())*/;
// // 	double pullGPY_tr_stereo = (stereoTsosGPos.y()-stereoRhitGPos.y())/*/sqrt(stereoTsosGPEr.cyy()+stereoRhitGPEr.cyy())*/;
// // 	double pullGPZ_tr_stereo = (stereoTsosGPos.z()-stereoRhitGPos.z())/*/sqrt(stereoTsosGPEr.czz()+stereoRhitGPEr.czz())*/;

// 	title.str("");
// 	title << "PullGP_X_" << subdetId << "-" << layerId << "_tr_stereo";
// 	hPullGP_X_tr_stereo[title.str()]->Fill( pullGPX_tr_stereo );
// 	title.str("");
// 	title << "PullGP_Y_" << subdetId << "-" << layerId << "_tr_stereo";
// 	hPullGP_Y_tr_stereo[title.str()]->Fill( pullGPY_tr_stereo );
// 	title.str("");
// 	title << "PullGP_Z_" << subdetId << "-" << layerId << "_tr_stereo";
// 	hPullGP_Z_tr_stereo[title.str()]->Fill( pullGPZ_tr_stereo );

// 	double pullGPX_ts_stereo = (stereoTsosGPos.x()-stereoShitGPos.x())/sqrt(stereoTsosGPEr.cxx());
// 	double pullGPY_ts_stereo = (stereoTsosGPos.y()-stereoShitGPos.y())/sqrt(stereoTsosGPEr.cyy());
// 	double pullGPZ_ts_stereo = (stereoTsosGPos.z()-stereoShitGPos.z())/sqrt(stereoTsosGPEr.czz());
// // 	double pullGPX_ts_stereo = (stereoTsosGPos.x()-stereoShitGPos.x())/*/sqrt(stereoTsosGPEr.cxx())*/;
// // 	double pullGPY_ts_stereo = (stereoTsosGPos.y()-stereoShitGPos.y())/*/sqrt(stereoTsosGPEr.cyy())*/;
// // 	double pullGPZ_ts_stereo = (stereoTsosGPos.z()-stereoShitGPos.z())/*/sqrt(stereoTsosGPEr.czz())*/;

// 	title.str("");
// 	title << "PullGP_X_" << subdetId << "-" << layerId << "_ts_stereo";
// 	hPullGP_X_ts_stereo[title.str()]->Fill( pullGPX_ts_stereo );
// 	title.str("");
// 	title << "PullGP_Y_" << subdetId << "-" << layerId << "_ts_stereo";
// 	hPullGP_Y_ts_stereo[title.str()]->Fill( pullGPY_ts_stereo );
// 	title.str("");
// 	title << "PullGP_Z_" << subdetId << "-" << layerId << "_ts_stereo";
// 	hPullGP_Z_ts_stereo[title.str()]->Fill( pullGPZ_ts_stereo );

// 	double pullGMX_ts_stereo = (stereoTsosGMom.x()-stereoShitGMom.x())/sqrt(stereoTsosGMEr.cxx());
// 	double pullGMY_ts_stereo = (stereoTsosGMom.y()-stereoShitGMom.y())/sqrt(stereoTsosGMEr.cyy());
// 	double pullGMZ_ts_stereo = (stereoTsosGMom.z()-stereoShitGMom.z())/sqrt(stereoTsosGMEr.czz());
// // 	double pullGMX_ts_stereo = (stereoTsosGMom.x()-stereoShitGMom.x())/*/sqrt(stereoTsosGMEr.cxx())*/;
// // 	double pullGMY_ts_stereo = (stereoTsosGMom.y()-stereoShitGMom.y())/*/sqrt(stereoTsosGMEr.cyy())*/;
// // 	double pullGMZ_ts_stereo = (stereoTsosGMom.z()-stereoShitGMom.z())/*/sqrt(stereoTsosGMEr.czz())*/;

// 	title.str("");
// 	title << "PullGM_X_" << subdetId << "-" << layerId << "_ts_stereo";
// 	hPullGM_X_ts_stereo[title.str()]->Fill( pullGMX_ts_stereo );
// 	title.str("");
// 	title << "PullGM_Y_" << subdetId << "-" << layerId << "_ts_stereo";
// 	hPullGM_Y_ts_stereo[title.str()]->Fill( pullGMY_ts_stereo );
// 	title.str("");
// 	title << "PullGM_Z_" << subdetId << "-" << layerId << "_ts_stereo";
// 	hPullGM_Z_ts_stereo[title.str()]->Fill( pullGMZ_ts_stereo );
//       }
//     }
//   }
//   LogTrace("TestHits") << "end of event" << std::endl;
// }

void TestHits::endJob() {
  //file->Write();
  TDirectory * chi2i = file->mkdir("Chi2_Increment");

  TDirectory * gp_ts = file->mkdir("GP_TSOS-SimHit");
  TDirectory * gm_ts = file->mkdir("GM_TSOS-SimHit");
  TDirectory * gp_tr = file->mkdir("GP_TSOS-RecHit");
  TDirectory * gp_rs = file->mkdir("GP_RecHit-SimHit");

  TDirectory * gp_tsx = gp_ts->mkdir("X");
  TDirectory * gp_tsy = gp_ts->mkdir("Y");
  TDirectory * gp_tsz = gp_ts->mkdir("Z");
  TDirectory * gm_tsx = gm_ts->mkdir("X");
  TDirectory * gm_tsy = gm_ts->mkdir("Y");
  TDirectory * gm_tsz = gm_ts->mkdir("Z");
  TDirectory * gp_trx = gp_tr->mkdir("X");
  TDirectory * gp_try = gp_tr->mkdir("Y");
  TDirectory * gp_trz = gp_tr->mkdir("Z");
  TDirectory * gp_rsx = gp_rs->mkdir("X");
  TDirectory * gp_rsy = gp_rs->mkdir("Y");
  TDirectory * gp_rsz = gp_rs->mkdir("Z");

  TDirectory * gp_tsx_mono = gp_ts->mkdir("MONOX");
  TDirectory * gp_tsy_mono = gp_ts->mkdir("MONOY");
  TDirectory * gp_tsz_mono = gp_ts->mkdir("MONOZ");
  TDirectory * gm_tsx_mono = gm_ts->mkdir("MONOX");
  TDirectory * gm_tsy_mono = gm_ts->mkdir("MONOY");
  TDirectory * gm_tsz_mono = gm_ts->mkdir("MONOZ");
  TDirectory * gp_trx_mono = gp_tr->mkdir("MONOX");
  TDirectory * gp_try_mono = gp_tr->mkdir("MONOY");
  TDirectory * gp_trz_mono = gp_tr->mkdir("MONOZ");
  TDirectory * gp_rsx_mono = gp_rs->mkdir("MONOX");
  TDirectory * gp_rsy_mono = gp_rs->mkdir("MONOY");
  TDirectory * gp_rsz_mono = gp_rs->mkdir("MONOZ");

  TDirectory * gp_tsx_stereo = gp_ts->mkdir("STEREOX");
  TDirectory * gp_tsy_stereo = gp_ts->mkdir("STEREOY");
  TDirectory * gp_tsz_stereo = gp_ts->mkdir("STEREOZ");
  TDirectory * gm_tsx_stereo = gm_ts->mkdir("STEREOX");
  TDirectory * gm_tsy_stereo = gm_ts->mkdir("STEREOY");
  TDirectory * gm_tsz_stereo = gm_ts->mkdir("STEREOZ");
  TDirectory * gp_trx_stereo = gp_tr->mkdir("STEREOX");
  TDirectory * gp_try_stereo = gp_tr->mkdir("STEREOY");
  TDirectory * gp_trz_stereo = gp_tr->mkdir("STEREOZ");
  TDirectory * gp_rsx_stereo = gp_rs->mkdir("STEREOX");
  TDirectory * gp_rsy_stereo = gp_rs->mkdir("STEREOY");
  TDirectory * gp_rsz_stereo = gp_rs->mkdir("STEREOZ");

  chi2i->cd();
  hTotChi2Increment->Write();
  hProcess_vs_Chi2->Write();
  hClsize_vs_Chi2->Write();
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      chi2i->cd();
      title.str("");
      title << "Chi2Increment_" << i+1 << "-" << j+1;
      hChi2Increment[title.str()]->Write();

      gp_ts->cd();
      gp_tsx->cd();
      title.str("");
      title << "PullGP_X_" << i+1 << "-" << j+1 << "_ts";
      hPullGP_X_ts[title.str()]->Write();
      gp_tsy->cd();
      title.str("");
      title << "PullGP_Y_" << i+1 << "-" << j+1 << "_ts";
      hPullGP_Y_ts[title.str()]->Write();
      gp_tsz->cd();
      title.str("");
      title << "PullGP_Z_" << i+1 << "-" << j+1 << "_ts";
      hPullGP_Z_ts[title.str()]->Write();

      gm_ts->cd();
      gm_tsx->cd();
      title.str("");
      title << "PullGM_X_" << i+1 << "-" << j+1 << "_ts";
      hPullGM_X_ts[title.str()]->Write();
      gm_tsy->cd();
      title.str("");
      title << "PullGM_Y_" << i+1 << "-" << j+1 << "_ts";
      hPullGM_Y_ts[title.str()]->Write();
      gm_tsz->cd();
      title.str("");
      title << "PullGM_Z_" << i+1 << "-" << j+1 << "_ts";
      hPullGM_Z_ts[title.str()]->Write();

      gp_tr->cd();
      gp_trx->cd();
      title.str("");
      title << "PullGP_X_" << i+1 << "-" << j+1 << "_tr";
      hPullGP_X_tr[title.str()]->Write();
      gp_try->cd();
      title.str("");
      title << "PullGP_Y_" << i+1 << "-" << j+1 << "_tr";
      hPullGP_Y_tr[title.str()]->Write();
      gp_trz->cd();
      title.str("");
      title << "PullGP_Z_" << i+1 << "-" << j+1 << "_tr";
      hPullGP_Z_tr[title.str()]->Write();

      gp_rs->cd();
      gp_rsx->cd();
      title.str("");
      title << "PullGP_X_" << i+1 << "-" << j+1 << "_rs";
      hPullGP_X_rs[title.str()]->Write();
      gp_rsy->cd();
      title.str("");
      title << "PullGP_Y_" << i+1 << "-" << j+1 << "_rs";
      hPullGP_Y_rs[title.str()]->Write();
      gp_rsz->cd();
      title.str("");
      title << "PullGP_Z_" << i+1 << "-" << j+1 << "_rs";
      hPullGP_Z_rs[title.str()]->Write();

      if ( ((i==2||i==4)&&(j==0||j==1)) || (i==3||i==5) ){
	//mono
	gp_ts->cd();
	gp_tsx_mono->cd();
	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGP_X_ts_mono[title.str()]->Write();
	gp_tsy_mono->cd();
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGP_Y_ts_mono[title.str()]->Write();
	gp_tsz_mono->cd();
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGP_Z_ts_mono[title.str()]->Write();

	gm_ts->cd();
	gm_tsx_mono->cd();
	title.str("");
	title << "PullGM_X_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGM_X_ts_mono[title.str()]->Write();
	gm_tsy_mono->cd();
	title.str("");
	title << "PullGM_Y_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGM_Y_ts_mono[title.str()]->Write();
	gm_tsz_mono->cd();
	title.str("");
	title << "PullGM_Z_" << i+1 << "-" << j+1 << "_ts_mono";
	hPullGM_Z_ts_mono[title.str()]->Write();

	gp_tr->cd();
	gp_trx_mono->cd();
	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_tr_mono";
	hPullGP_X_tr_mono[title.str()]->Write();
	gp_try_mono->cd();
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_tr_mono";
	hPullGP_Y_tr_mono[title.str()]->Write();
	gp_trz_mono->cd();
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_tr_mono";
	hPullGP_Z_tr_mono[title.str()]->Write();

	gp_rs->cd();
	gp_rsx_mono->cd();
	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_rs_mono";
	hPullGP_X_rs_mono[title.str()]->Write();
	gp_rsy_mono->cd();
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_rs_mono";
	hPullGP_Y_rs_mono[title.str()]->Write();
	gp_rsz_mono->cd();
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_rs_mono";
	hPullGP_Z_rs_mono[title.str()]->Write();

	//stereo
	gp_ts->cd();
	gp_tsx_stereo->cd();
	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGP_X_ts_stereo[title.str()]->Write();
	gp_tsy_stereo->cd();
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGP_Y_ts_stereo[title.str()]->Write();
	gp_tsz_stereo->cd();
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGP_Z_ts_stereo[title.str()]->Write();

	gm_ts->cd();
	gm_tsx_stereo->cd();
	title.str("");
	title << "PullGM_X_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGM_X_ts_stereo[title.str()]->Write();
	gm_tsy_stereo->cd();
	title.str("");
	title << "PullGM_Y_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGM_Y_ts_stereo[title.str()]->Write();
	gm_tsz_stereo->cd();
	title.str("");
	title << "PullGM_Z_" << i+1 << "-" << j+1 << "_ts_stereo";
	hPullGM_Z_ts_stereo[title.str()]->Write();

	gp_tr->cd();
	gp_trx_stereo->cd();
	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_tr_stereo";
	hPullGP_X_tr_stereo[title.str()]->Write();
	gp_try_stereo->cd();
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_tr_stereo";
	hPullGP_Y_tr_stereo[title.str()]->Write();
	gp_trz_stereo->cd();
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_tr_stereo";
	hPullGP_Z_tr_stereo[title.str()]->Write();

	gp_rs->cd();
	gp_rsx_stereo->cd();
	title.str("");
	title << "PullGP_X_" << i+1 << "-" << j+1 << "_rs_stereo";
	hPullGP_X_rs_stereo[title.str()]->Write();
	gp_rsy_stereo->cd();
	title.str("");
	title << "PullGP_Y_" << i+1 << "-" << j+1 << "_rs_stereo";
	hPullGP_Y_rs_stereo[title.str()]->Write();
	gp_rsz_stereo->cd();
	title.str("");
	title << "PullGP_Z_" << i+1 << "-" << j+1 << "_rs_stereo";
	hPullGP_Z_rs_stereo[title.str()]->Write();
      }
    }

  file->Close();
}

//needed by to do the residual for matched hits
//taken from SiStripTrackingRecHitsValid.cc
std::pair<LocalPoint,LocalVector> 
TestHits::projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet, const BoundPlane& plane) 
{ 
   const StripTopology& topol = stripDet->specificTopology();
   GlobalPoint globalpos= stripDet->surface().toGlobal(hit.localPosition());
   LocalPoint localHit = plane.toLocal(globalpos);
   //track direction
   LocalVector locdir=hit.localDirection();
   //rotate track in new frame
   
   GlobalVector globaldir= stripDet->surface().toGlobal(locdir);
   LocalVector dir=plane.toLocal(globaldir);
   float scale = -localHit.z() / dir.z();
   
   LocalPoint projectedPos = localHit + scale*dir;
   
   float selfAngle = topol.stripAngle( topol.strip( hit.localPosition()));
   
   LocalVector stripDir( sin(selfAngle), cos(selfAngle), 0); // vector along strip in hit frame
   
   LocalVector localStripDir( plane.toLocal(stripDet->surface().toGlobal( stripDir)));
   
   return std::pair<LocalPoint,LocalVector>( projectedPos, localStripDir);
}

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TestHits);
