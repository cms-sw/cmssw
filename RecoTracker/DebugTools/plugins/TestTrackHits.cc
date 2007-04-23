#include "RecoTracker/DebugTools/interface/TestTrackHits.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include <TDirectory.h>

typedef TrajectoryStateOnSurface TSOS;
typedef TransientTrackingRecHit::ConstRecHitPointer CTTRHp;
using namespace std;
using namespace edm;

TestTrackHits::TestTrackHits(const edm::ParameterSet& iConfig):
  conf_(iConfig){
  LogTrace("TestTrackHits") << conf_;
  propagatorName = conf_.getParameter<std::string>("Propagator");   
  builderName = conf_.getParameter<std::string>("TTRHBuilder");   
  srcName = conf_.getParameter<std::string>("src");   
  updatorName = conf_.getParameter<std::string>("updator");
  mineta = conf_.getParameter<double>("mineta");
  maxeta = conf_.getParameter<double>("maxeta");
}

TestTrackHits::~TestTrackHits(){}

void TestTrackHits::beginJob(const edm::EventSetup& iSetup)
{
  LogVerbatim("CkfDebugger") << "pippo" ;
  iSetup.get<TrackerDigiGeometryRecord>().get(theG);
  iSetup.get<IdealMagneticFieldRecord>().get(theMF);  
  iSetup.get<TrackingComponentsRecord>().get(propagatorName,thePropagator);
  iSetup.get<TransientRecHitRecord>().get(builderName,theBuilder);
  iSetup.get<TrackingComponentsRecord>().get(updatorName,theUpdator);
 
  file = new TFile("test_track_hits.root","recreate");
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;

      title.str("");
      title << "Chi2Increment_" << i+1 << "-" << j+1 ;
      hChi2Increment[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),50,0,50);

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
}

void TestTrackHits::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  LogVerbatim("CkfDebugger") << "new event" ;
  iEvent.getByLabel(srcName,trajCollectionHandle);
  iEvent.getByLabel(srcName,trajTrackAssociationCollectionHandle);

  hitAssociator = new TrackerHitAssociator::TrackerHitAssociator(iEvent);

  int i=0;
  for(vector<Trajectory>::const_iterator it = trajCollectionHandle->begin(); it!=trajCollectionHandle->end();it++){
    
    LogVerbatim("CkfDebugger") << "new collection" ;
    double tchi2 = 0;

    vector<TrajectoryMeasurement> tmColl = it->measurements();

    // (theTSOS.globalMomentum().eta()>maxeta || theTSOS.globalMomentum().eta()<mineta) continue;
    
    for (vector<TrajectoryMeasurement>::iterator tm=tmColl.begin();tm!=tmColl.end();++tm){

      tchi2+=tm->estimate();

      LogVerbatim("CkfDebugger") << "new hit" ;
      CTTRHp rhit = tm->recHit();
      //TSOS state = tm->backwardPredictedState();
      TSOS state = tm->forwardPredictedState();

      if (rhit->isValid()==0) continue;

      int subdetId = rhit->det()->geographicalId().subdetId();
      int layerId  = ((int)(((rhit->det()->geographicalId().rawId() >>16) & 0xF)));
      title.str("");
      title << "Chi2Increment_" << subdetId << "-" << layerId ;
      hChi2Increment[title.str()]->Fill( tm->estimate() );
      const Surface * surf = &( rhit->det()->surface() );
      if (surf==0) continue;

      vector<PSimHit> assSimHits = hitAssociator->associateHit(*(rhit)->hit());
      if (assSimHits.size()==0) continue;
      PSimHit shit=*(assSimHits.begin());

      LocalVector shitLMom = shit.momentumAtEntry();
      GlobalVector shitGMom = surf->toGlobal(shitLMom);
      LocalPoint shitLPos = shit.localPosition();
      GlobalPoint shitGPos = surf->toGlobal(shitLPos);

      if (dynamic_cast<const SiStripMatchedRecHit2D*>(rhit->hit()) && assSimHits.size()==2) {
	const GluedGeomDet* gluedUnit = dynamic_cast<const GluedGeomDet*>(rhit->det());
	if ( gluedUnit ) {
	  shitGPos = matchSimHits(assSimHits[0],assSimHits[1],gluedUnit);
	}
      }

      GlobalVector tsosGMom = state.globalMomentum();
      GlobalError  tsosGMEr(state.cartesianError().matrix().sub(4,6));
      GlobalPoint  tsosGPos = state.globalPosition();
      GlobalError  tsosGPEr = state.cartesianError().position();

      GlobalPoint rhitGPos = (rhit)->globalPosition();
      GlobalError rhitGPEr = (rhit)->globalPositionError();

      LogVerbatim("CkfDebugger") << "\nsubdetId=" << subdetId << " layerId=" << layerId ;
      LogVerbatim("CkfDebugger") << "assSimHits.size()=" << assSimHits.size() ;
      LogVerbatim("CkfDebugger") << "tsosGPos=" << tsosGPos ;
      LogVerbatim("CkfDebugger") << "shitGPos=" << shitGPos ;
      LogVerbatim("CkfDebugger") << "rhitGPos=" << rhitGPos ;
      LogVerbatim("CkfDebugger") << "surf->position=" << surf->position() ;
      LogVerbatim("CkfDebugger") << "rhit->det()->geographicalId()=" << rhit->det()->geographicalId().rawId() ;
      if (rhit->detUnit()) LogVerbatim("CkfDebugger") << "rhit->detUnit()->geographicalId()=" << rhit->detUnit()->geographicalId().rawId() ;
      LogVerbatim("CkfDebugger") << "rhit->det()->surface().position()=" << rhit->det()->surface().position() ;
      if (rhit->detUnit()) LogVerbatim("CkfDebugger") << "rhit->detUnit()->surface().position()="  << rhit->detUnit()->surface().position() ;
      LogVerbatim("CkfDebugger") << "rhit->det()->position()=" << rhit->det()->position() ;
      if (rhit->detUnit()) LogVerbatim("CkfDebugger") << "rhit->detUnit()->position()="  << rhit->detUnit()->position() ;

      LogVerbatim("CkfDebugger") << "rhit->det()->surface().bounds().length()=" << rhit->det()->surface().bounds().length() ;
      if (rhit->detUnit()) LogVerbatim("CkfDebugger") << "rhit->detUnit()->surface().bounds().length()="  << rhit->detUnit()->surface().bounds().length() ;
      LogVerbatim("CkfDebugger") << "rhit->det()->surface().bounds().width()=" << rhit->det()->surface().bounds().width() ;
      if (rhit->detUnit()) LogVerbatim("CkfDebugger") << "rhit->detUnit()->surface().bounds().width()="  << rhit->detUnit()->surface().bounds().width() ;
      LogVerbatim("CkfDebugger") << "rhit->det()->surface().bounds().thickness()=" << rhit->det()->surface().bounds().thickness() ;
      if (rhit->detUnit()) LogVerbatim("CkfDebugger") << "rhit->detUnit()->surface().bounds().thickness()="  << rhit->detUnit()->surface().bounds().thickness() ;

      double pullGPX_rs = (rhitGPos.x()-shitGPos.x())/sqrt(rhitGPEr.cxx());
      double pullGPY_rs = (rhitGPos.y()-shitGPos.y())/sqrt(rhitGPEr.cyy());
      double pullGPZ_rs = (rhitGPos.z()-shitGPos.z())/sqrt(rhitGPEr.czz());
//       double pullGPX_rs = (rhitGPos.x()-shitGPos.x())/*/sqrt(rhitGPEr.cxx())*/;
//       double pullGPY_rs = (rhitGPos.y()-shitGPos.y())/*/sqrt(rhitGPEr.cyy())*/;
//       double pullGPZ_rs = (rhitGPos.z()-shitGPos.z())/*/sqrt(rhitGPEr.czz())*/;

      LogVerbatim("CkfDebugger") << "rs" ;

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
//       double pullGPX_tr = (tsosGPos.x()-rhitGPos.x())/*/sqrt(tsosGPEr.cxx()+rhitGPEr.cxx())*/;
//       double pullGPY_tr = (tsosGPos.y()-rhitGPos.y())/*/sqrt(tsosGPEr.cyy()+rhitGPEr.cyy())*/;
//       double pullGPZ_tr = (tsosGPos.z()-rhitGPos.z())/*/sqrt(tsosGPEr.czz()+rhitGPEr.czz())*/;

      LogVerbatim("CkfDebugger") << "tr" ;

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
//       double pullGPX_ts = (tsosGPos.x()-shitGPos.x())/*/sqrt(tsosGPEr.cxx())*/;
//       double pullGPY_ts = (tsosGPos.y()-shitGPos.y())/*/sqrt(tsosGPEr.cyy())*/;
//       double pullGPZ_ts = (tsosGPos.z()-shitGPos.z())/*/sqrt(tsosGPEr.czz())*/;

      LogVerbatim("CkfDebugger") << "ts1" ;

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
//       double pullGMX_ts = (tsosGMom.x()-shitGMom.x())/*/sqrt(tsosGMEr.cxx())*/;
//       double pullGMY_ts = (tsosGMom.y()-shitGMom.y())/*/sqrt(tsosGMEr.cyy())*/;
//       double pullGMZ_ts = (tsosGMom.z()-shitGMom.z())/*/sqrt(tsosGMEr.czz())*/;

      LogVerbatim("CkfDebugger") << "ts2" ;

      title.str("");
      title << "PullGM_X_" << subdetId << "-" << layerId << "_ts";
      hPullGM_X_ts[title.str()]->Fill( pullGMX_ts );
      title.str("");
      title << "PullGM_Y_" << subdetId << "-" << layerId << "_ts";
      hPullGM_Y_ts[title.str()]->Fill( pullGMY_ts );
      title.str("");
      title << "PullGM_Z_" << subdetId << "-" << layerId << "_ts";
      hPullGM_Z_ts[title.str()]->Fill( pullGMZ_ts );
      //#if 0
      if (dynamic_cast<const SiStripMatchedRecHit2D*>(rhit->hit())) {
	Propagator* thePropagatorAnyDir = new PropagatorWithMaterial(anyDirection,0.105,theMF.product(),1.6);
	//mono
	LogVerbatim("CkfDebugger") << "MONO HIT" ;
	CTTRHp tMonoHit = 
	  theBuilder->build(dynamic_cast<const SiStripMatchedRecHit2D*>(rhit->hit())->monoHit());
	if (tMonoHit==0) continue;
	vector<PSimHit> assMonoSimHits = hitAssociator->associateHit(*tMonoHit->hit());
	if (assMonoSimHits.size()==0) continue;
	const PSimHit sMonoHit = *(assMonoSimHits.begin());
	const Surface * monoSurf = &( tMonoHit->det()->surface() );
	if (monoSurf==0) continue;
	TSOS monoState = thePropagatorAnyDir->propagate(state,*monoSurf);
	if (monoState.isValid()==0) continue;

	LocalVector monoShitLMom = sMonoHit.momentumAtEntry();
	GlobalVector monoShitGMom = monoSurf->toGlobal(monoShitLMom);
	LocalPoint monoShitLPos = sMonoHit.localPosition();
	GlobalPoint monoShitGPos = monoSurf->toGlobal(monoShitLPos);

// 	LogVerbatim("CkfDebugger") << "assMonoSimHits.size()=" << assMonoSimHits.size() ;
// 	LogVerbatim("CkfDebugger") << "mono shit=" << monoShitGPos ;

	GlobalVector monoTsosGMom = monoState.globalMomentum();
	GlobalError  monoTsosGMEr(monoState.cartesianError().matrix().sub(4,6));
	GlobalPoint  monoTsosGPos = monoState.globalPosition();
	GlobalError  monoTsosGPEr = monoState.cartesianError().position();

	GlobalPoint monoRhitGPos = tMonoHit->globalPosition();
	GlobalError monoRhitGPEr = tMonoHit->globalPositionError();

	double pullGPX_rs_mono = (monoRhitGPos.x()-monoShitGPos.x())/sqrt(monoRhitGPEr.cxx());
	double pullGPY_rs_mono = (monoRhitGPos.y()-monoShitGPos.y())/sqrt(monoRhitGPEr.cyy());
	double pullGPZ_rs_mono = (monoRhitGPos.z()-monoShitGPos.z())/sqrt(monoRhitGPEr.czz());
// 	double pullGPX_rs_mono = (monoRhitGPos.x()-monoShitGPos.x())/*/sqrt(monoRhitGPEr.cxx())*/;
// 	double pullGPY_rs_mono = (monoRhitGPos.y()-monoShitGPos.y())/*/sqrt(monoRhitGPEr.cyy())*/;
// 	double pullGPZ_rs_mono = (monoRhitGPos.z()-monoShitGPos.z())/*/sqrt(monoRhitGPEr.czz())*/;

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
// 	double pullGPX_tr_mono = (monoTsosGPos.x()-monoRhitGPos.x())/*/sqrt(monoTsosGPEr.cxx()+monoRhitGPEr.cxx())*/;
// 	double pullGPY_tr_mono = (monoTsosGPos.y()-monoRhitGPos.y())/*/sqrt(monoTsosGPEr.cyy()+monoRhitGPEr.cyy())*/;
// 	double pullGPZ_tr_mono = (monoTsosGPos.z()-monoRhitGPos.z())/*/sqrt(monoTsosGPEr.czz()+monoRhitGPEr.czz())*/;

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
// 	double pullGPX_ts_mono = (monoTsosGPos.x()-monoShitGPos.x())/*/sqrt(monoTsosGPEr.cxx())*/;
// 	double pullGPY_ts_mono = (monoTsosGPos.y()-monoShitGPos.y())/*/sqrt(monoTsosGPEr.cyy())*/;
// 	double pullGPZ_ts_mono = (monoTsosGPos.z()-monoShitGPos.z())/*/sqrt(monoTsosGPEr.czz())*/;

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
// 	double pullGMX_ts_mono = (monoTsosGMom.x()-monoShitGMom.x())/*/sqrt(monoTsosGMEr.cxx())*/;
// 	double pullGMY_ts_mono = (monoTsosGMom.y()-monoShitGMom.y())/*/sqrt(monoTsosGMEr.cyy())*/;
// 	double pullGMZ_ts_mono = (monoTsosGMom.z()-monoShitGMom.z())/*/sqrt(monoTsosGMEr.czz())*/;

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
	LogVerbatim("CkfDebugger") << "STEREO HIT" ;
	CTTRHp tStereoHit = 
	  theBuilder->build(dynamic_cast<const SiStripMatchedRecHit2D*>(rhit->hit())->stereoHit());
	if (tStereoHit==0) continue;
	vector<PSimHit> assStereoSimHits = hitAssociator->associateHit(*tStereoHit->hit());
	if (assStereoSimHits.size()==0) continue;
	const PSimHit sStereoHit = *(assStereoSimHits.begin());
	const Surface * stereoSurf = &( tStereoHit->det()->surface() );
	if (stereoSurf==0) continue;
	TSOS stereoState = thePropagatorAnyDir->propagate(state,*stereoSurf);
	if (stereoState.isValid()==0) continue;

	LocalVector stereoShitLMom = sStereoHit.momentumAtEntry();
	GlobalVector stereoShitGMom = stereoSurf->toGlobal(stereoShitLMom);
	LocalPoint stereoShitLPos = sStereoHit.localPosition();
	GlobalPoint stereoShitGPos = stereoSurf->toGlobal(stereoShitLPos);

// 	LogVerbatim("CkfDebugger") << "assStereoSimHits.size()=" << assStereoSimHits.size() ;
// 	LogVerbatim("CkfDebugger") << "stereo shit=" << stereoShitGPos ;

	GlobalVector stereoTsosGMom = stereoState.globalMomentum();
	GlobalError  stereoTsosGMEr(stereoState.cartesianError().matrix().sub(4,6));
	GlobalPoint  stereoTsosGPos = stereoState.globalPosition();
	GlobalError  stereoTsosGPEr = stereoState.cartesianError().position();

	GlobalPoint stereoRhitGPos = tStereoHit->globalPosition();
	GlobalError stereoRhitGPEr = tStereoHit->globalPositionError();

	double pullGPX_rs_stereo = (stereoRhitGPos.x()-stereoShitGPos.x())/sqrt(stereoRhitGPEr.cxx());
	double pullGPY_rs_stereo = (stereoRhitGPos.y()-stereoShitGPos.y())/sqrt(stereoRhitGPEr.cyy());
	double pullGPZ_rs_stereo = (stereoRhitGPos.z()-stereoShitGPos.z())/sqrt(stereoRhitGPEr.czz());
// 	double pullGPX_rs_stereo = (stereoRhitGPos.x()-stereoShitGPos.x())/*/sqrt(stereoRhitGPEr.cxx())*/;
// 	double pullGPY_rs_stereo = (stereoRhitGPos.y()-stereoShitGPos.y())/*/sqrt(stereoRhitGPEr.cyy())*/;
// 	double pullGPZ_rs_stereo = (stereoRhitGPos.z()-stereoShitGPos.z())/*/sqrt(stereoRhitGPEr.czz())*/;

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
// 	double pullGPX_tr_stereo = (stereoTsosGPos.x()-stereoRhitGPos.x())/*/sqrt(stereoTsosGPEr.cxx()+stereoRhitGPEr.cxx())*/;
// 	double pullGPY_tr_stereo = (stereoTsosGPos.y()-stereoRhitGPos.y())/*/sqrt(stereoTsosGPEr.cyy()+stereoRhitGPEr.cyy())*/;
// 	double pullGPZ_tr_stereo = (stereoTsosGPos.z()-stereoRhitGPos.z())/*/sqrt(stereoTsosGPEr.czz()+stereoRhitGPEr.czz())*/;

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
// 	double pullGPX_ts_stereo = (stereoTsosGPos.x()-stereoShitGPos.x())/*/sqrt(stereoTsosGPEr.cxx())*/;
// 	double pullGPY_ts_stereo = (stereoTsosGPos.y()-stereoShitGPos.y())/*/sqrt(stereoTsosGPEr.cyy())*/;
// 	double pullGPZ_ts_stereo = (stereoTsosGPos.z()-stereoShitGPos.z())/*/sqrt(stereoTsosGPEr.czz())*/;

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
// 	double pullGMX_ts_stereo = (stereoTsosGMom.x()-stereoShitGMom.x())/*/sqrt(stereoTsosGMEr.cxx())*/;
// 	double pullGMY_ts_stereo = (stereoTsosGMom.y()-stereoShitGMom.y())/*/sqrt(stereoTsosGMEr.cyy())*/;
// 	double pullGMZ_ts_stereo = (stereoTsosGMom.z()-stereoShitGMom.z())/*/sqrt(stereoTsosGMEr.czz())*/;

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
      //#endif
    }
    LogVerbatim("CkfDebugger") << "tchi2="  << tchi2 ;
    edm::Ref<vector<Trajectory> >  traj(trajCollectionHandle, i);
    LogVerbatim("CkfDebugger") << "chi2=" << (*trajTrackAssociationCollectionHandle.product())[traj]->chi2() ;
    i++;
  }
  delete hitAssociator;
  LogVerbatim("CkfDebugger") << "end of event" ;
}

void TestTrackHits::endJob() {
  //file->Write();
  TDirectory * chi2d = file->mkdir("Chi2Increment");
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

  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      chi2d->cd();
      title.str("");
      title << "Chi2Increment_" << i+1 << "-" << j+1 ;
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

#include "RecoLocalTracker/SiStripRecHitConverter/interface/SiStripRecHitMatcher.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

typedef SiStripRecHitMatcher::StripPosition StripPosition;

GlobalPoint TestTrackHits::matchSimHits(PSimHit& monoH, PSimHit& stereoH,const GluedGeomDet* gluedDet) {

  const PSimHit hit = stereoH;
  const StripGeomDetUnit* stripDet =(StripGeomDetUnit*) gluedDet->stereoDet();
  const BoundPlane plane = gluedDet->surface();


  //const StripTopology& topol = stripDet->specificTopology();
  GlobalPoint globalpos= stripDet->surface().toGlobal(hit.localPosition());
  LocalPoint localHit = plane.toLocal(globalpos);
  //track direction
  LocalVector locdir=hit.localDirection();
  //rotate track in new frame
  
  GlobalVector globaldir= stripDet->surface().toGlobal(locdir);
  LocalVector dir=plane.toLocal(globaldir);
  float scale = -localHit.z() / dir.z();
  
  LocalPoint projectedPos = localHit + scale*dir;
  
  //  std::LogVerbatim("CkfDebugger") << "projectedPos " << projectedPos ;
  
  //float selfAngle = topol.stripAngle( topol.strip( hit.localPosition()));
  
  //LocalVector stripDir( sin(selfAngle), cos(selfAngle), 0); // vector along strip in hit frame  
  //LocalVector localStripDir( plane.toLocal(stripDet->surface().toGlobal( stripDir)));

  return plane.toGlobal(projectedPos);
#if 0
  const GeomDetUnit* monoDet = gluedDet->monoDet();
  const GeomDetUnit* stereoDet = gluedDet->stereoDet();

  const StripTopology& topol=(const StripTopology&)monoDet->topology();
  LocalPoint position;    

  // position of the initial and final point of the strip (RPHI cluster) in local strip coordinates
  MeasurementPoint RPHIpoint=topol.measurementPosition(monoH.localPosition());
  MeasurementPoint RPHIpointini=MeasurementPoint(RPHIpoint.x(),-0.5);
  MeasurementPoint RPHIpointend=MeasurementPoint(RPHIpoint.x(),0.5);

  // position of the initial and final point of the strip in local coordinates (mono det)
  StripPosition stripmono=StripPosition(topol.localPosition(RPHIpointini),topol.localPosition(RPHIpointend));
  LocalPoint lcenterofstrip=monoH.localPosition();
  GlobalPoint gcenterofstrip=(monoDet->surface()).toGlobal(lcenterofstrip);
  GlobalVector gtrackdirection=gcenterofstrip-GlobalPoint(0,0,0);
  LocalVector trackdirection=(gluedDet->surface()).toLocal(gtrackdirection);

  //project mono hit on glued det
  GlobalPoint globalpointini=(monoDet->surface()).toGlobal(stripmono.first);
  GlobalPoint globalpointend=(monoDet->surface()).toGlobal(stripmono.second);
  // position of the initial and final point of the strip in glued local coordinates
  LocalPoint positiononGluedini=(gluedDet->surface()).toLocal(globalpointini);
  LocalPoint positiononGluedend=(gluedDet->surface()).toLocal(globalpointend);
  //correct the position with the track direction
  float scale=-positiononGluedini.z()/trackdirection.z();
  LocalPoint projpositiononGluedini= positiononGluedini + scale*trackdirection;
  LocalPoint projpositiononGluedend= positiononGluedend + scale*trackdirection;
  StripPosition projectedstripmono=StripPosition(projpositiononGluedini,projpositiononGluedend);

  const StripTopology& partnertopol=(const StripTopology&)stereoDet->topology();

//   //error calculation (the part that depends on mono RH only)
//   LocalVector  RPHIpositiononGluedendvector=projectedstripmono.second-projectedstripmono.first;
//   double c1=sin(RPHIpositiononGluedendvector.phi()); 
//   double s1=-cos(RPHIpositiononGluedendvector.phi());
//   MeasurementError errormonoRH=topol.measurementError(monoH->localPosition(),monoH->localPositionError());
//   double sigmap12=errormonoRH.uu()*pow(topol.localPitch(monoH->localPosition()),2);

  // position of the initial and final point of the strip (STEREO cluster)
  MeasurementPoint STEREOpoint=partnertopol.measurementPosition(stereoH.localPosition());
  MeasurementPoint STEREOpointini=MeasurementPoint(STEREOpoint.x(),-0.5);
  MeasurementPoint STEREOpointend=MeasurementPoint(STEREOpoint.x(),0.5);

  // position of the initial and final point of the strip in local coordinates (stereo det)
  StripPosition stripstereo(partnertopol.localPosition(STEREOpointini),partnertopol.localPosition(STEREOpointend));
 
  //project stereo hit on glued det
  globalpointini=(stereoDet->surface()).toGlobal(stripstereo.first);
  globalpointend=(stereoDet->surface()).toGlobal(stripstereo.second);
  // position of the initial and final point of the strip in glued local coordinates
  positiononGluedini=(gluedDet->surface()).toLocal(globalpointini);
  positiononGluedend=(gluedDet->surface()).toLocal(globalpointend);
  //correct the position with the track direction
  scale=-positiononGluedini.z()/trackdirection.z();
  projpositiononGluedini= positiononGluedini + scale*trackdirection;
  projpositiononGluedend= positiononGluedend + scale*trackdirection;
  StripPosition projectedstripstereo=StripPosition(projpositiononGluedini,projpositiononGluedend);
  
  //perform the matching
  //(x2-x1)(y-y1)=(y2-y1)(x-x1)
  AlgebraicMatrix m(2,2); AlgebraicVector c(2), solution(2);
  m(1,1)=-(projectedstripmono.second.y()-projectedstripmono.first.y()); 
  m(1,2)=(projectedstripmono.second.x()-projectedstripmono.first.x());
  m(2,1)=-(projectedstripstereo.second.y()-projectedstripstereo.first.y()); 
  m(2,2)=(projectedstripstereo.second.x()-projectedstripstereo.first.x());
  c(1)=m(1,2)*projectedstripmono.first.y()+m(1,1)*projectedstripmono.first.x();
  c(2)=m(2,2)*projectedstripstereo.first.y()+m(2,1)*projectedstripstereo.first.x();
  solution=solve(m,c);
  position=LocalPoint(solution(1),solution(2));

  return (gluedDet->surface()).toGlobal(position);
#endif
}

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_ANOTHER_FWK_MODULE(TestTrackHits);
