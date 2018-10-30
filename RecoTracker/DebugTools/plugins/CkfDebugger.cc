#include "CkfDebugger.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "TSOSFromSimHitFactory.h"
#include "TrackingTools/MeasurementDet/interface/MeasurementDet.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include "GluedDetFromDetUnit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "TrackingTools/DetLayers/interface/TkLayerLess.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
// #include "RecoTracker/TkDetLayers/interface/PixelBarrelLayer.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TrackingRecHitProjector.h"
#include "RecoTracker/TransientTrackingRecHit/interface/ProjectedRecHit2D.h"

#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;

CkfDebugger::CkfDebugger( edm::EventSetup const & es, edm::ConsumesCollector&& iC):trackerHitAssociatorConfig_(std::move(iC)), totSeeds(0)
{
  file = new TFile("out.root","recreate");
  hchi2seedAll = new TH1F("hchi2seedAll","hchi2seedAll",2000,0,200);
  hchi2seedProb = new TH1F("hchi2seedProb","hchi2seedProb",2000,0,200);

  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTrackerGeom = &(*tracker);

  edm::ESHandle<MagneticField>                theField;
  es.get<IdealMagneticFieldRecord>().get(theField);
  theMagField = &(*theField);
  
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  es.get<IdealGeometryRecord>().get(tTopoHand);
  theTopo=tTopoHand.product();

  edm::ESHandle<NavigationSchool> nav;
  es.get<NavigationSchoolRecord>().get("SimpleNavigationSchool", nav);
  theNavSchool = nav.product();

  for (int i=0; i!=17; i++){
    dump.push_back(0);
  }

  std::stringstream title;
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      dump2[pair<int,int>(i,j)]=0;
      dump3[pair<int,int>(i,j)]=0;
      dump4[pair<int,int>(i,j)]=0;
      dump5[pair<int,int>(i,j)]=0;
      dump6[pair<int,int>(i,j)]=0;
      title.str("");
      title << "pullX_" << i+1 << "-" << j+1 << "_sh-rh";
      hPullX_shrh[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "pullY_" << i+1 << "-" << j+1 << "_sh-rh";
      hPullY_shrh[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "pullX_" << i+1 << "-" << j+1 << "_sh-st";
      hPullX_shst[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "pullY_" << i+1 << "-" << j+1 << "_sh-st";
      hPullY_shst[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "pullX_" << i+1 << "-" << j+1 << "_st-rh";
      hPullX_strh[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "pullY_" << i+1 << "-" << j+1 << "_st-rh";
      hPullY_strh[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGP_X_" << i+1 << "-" << j+1 << "_sh-st";
      hPullGP_X_shst[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGP_Y_" << i+1 << "-" << j+1 << "_sh-st";
      hPullGP_Y_shst[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      title.str("");
      title << "PullGP_Z_" << i+1 << "-" << j+1 << "_sh-st";
      hPullGP_Z_shst[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      if ( ((i==2||i==4)&&(j==0||j==1)) || (i==3||i==5) ){
	title.str("");
	title << "pullM_" << i+1 << "-" << j+1 << "_sh-rh";
	hPullM_shrh[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "pullS_" << i+1 << "-" << j+1 << "_sh-rh";
	hPullS_shrh[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "pullM_" << i+1 << "-" << j+1 << "_sh-st";
	hPullM_shst[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "pullS_" << i+1 << "-" << j+1 << "_sh-st";
	hPullS_shst[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "pullM_" << i+1 << "-" << j+1 << "_st-rh";
	hPullM_strh[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
	title.str("");
	title << "pullS_" << i+1 << "-" << j+1 << "_st-rh";
	hPullS_strh[title.str()] = new TH1F(title.str().c_str(),title.str().c_str(),1000,-50,50);
      }
    }

  hPullGPXvsGPX_shst   = new TH2F("PullGPXvsGPX_shst","PullGPXvsGPX_shst",1000,-50,50,100,-50,50);
  hPullGPXvsGPY_shst   = new TH2F("PullGPXvsGPY_shst","PullGPXvsGPY_shst",1000,-50,50,100,-50,50);
  hPullGPXvsGPZ_shst   = new TH2F("PullGPXvsGPZ_shst","PullGPXvsGPZ_shst",1000,-50,50,200,-100,100);
  hPullGPXvsGPr_shst   = new TH2F("PullGPXvsGPr_shst","PullGPXvsGPr_shst",1000,-50,50,300,-150,150);
  hPullGPXvsGPeta_shst = new TH2F("PullGPXvsGPeta_shst","PullGPXvsGPeta_shst",1000,-50,50,50,-2.5,2.5);
  hPullGPXvsGPphi_shst = new TH2F("PullGPXvsGPphi_shst","PullGPXvsGPphi_shst",1000,-50,50,63,0,6.3);
 
  seedWithDelta=0;
  problems=0;
  no_sim_hit=0;
  no_layer=0;
  layer_not_found=0;
  det_not_found=0;
  chi2gt30=0;
  chi2gt30delta=0;
  chi2gt30deltaSeed=0;
  chi2ls30=0;
  simple_hit_not_found=0;
  no_component=0;
  only_one_component=0;
  matched_not_found=0;
  matched_not_associated=0;
  partner_det_not_fuond=0;
  glued_det_not_fuond=0;
  propagation=0;
  other=0;
  totchi2gt30=0;
}

void CkfDebugger::printSimHits( const edm::Event& iEvent)
{
  edm::LogVerbatim("CkfDebugger") << "\nEVENT #" << iEvent.id();

  hitAssociator = new TrackerHitAssociator(iEvent, trackerHitAssociatorConfig_);//delete deleteHitAssociator() in TrackCandMaker.cc

  std::map<unsigned int, std::vector<PSimHit> >& theHitsMap = hitAssociator->SimHitMap;
  idHitsMap.clear();

  for (std::map<unsigned int, std::vector<PSimHit> >::iterator it=theHitsMap.begin();
       it!=theHitsMap.end();it++){
    for (std::vector<PSimHit>::iterator isim = it->second.begin();
	 isim != it->second.end(); ++isim){
      idHitsMap[isim->trackId()].push_back(&*isim);
    }
  }
  
  for (std::map<unsigned int,std::vector<PSimHit*> >::iterator it=idHitsMap.begin();
       it!=idHitsMap.end();it++){
    sort(it->second.begin(),it->second.end(),[](auto* a, auto* b){ return a->timeOfFlight()< b->timeOfFlight(); });
    for (std::vector<PSimHit*>::iterator isim = it->second.begin();
	 isim != it->second.end(); ++isim){
      const GeomDetUnit* detUnit = theTrackerGeom->idToDetUnit( DetId((*isim)->detUnitId()));
      dumpSimHit(SimHit( (*isim), detUnit));
    }
  }
  
}

void CkfDebugger::dumpSimHit(const SimHit& hit) const
{
  GlobalPoint pos = hit.globalPosition();
  edm::LogVerbatim("CkfDebugger") << "SimHit pos" << pos
				  << " r=" << pos.perp() << " phi=" << pos.phi() 
				  << " trackId=" << hit.trackId() 
				  << " particleType=" << hit.particleType() 
				  << " pabs=" << hit.pabs() 
				  << " processType=" << hit. processType();
}


bool CkfDebugger::analyseCompatibleMeasurements(const Trajectory& traj,
						const std::vector<TrajectoryMeasurement>& meas,
						const MeasurementTracker* aMeasurementTracker, 
						const Propagator*                     propagator,
						const Chi2MeasurementEstimatorBase*   estimator,
						const TransientTrackingRecHitBuilder* aTTRHBuilder)
{
  LogTrace("CkfDebugger") << "\nnow in analyseCompatibleMeasurements" ;
  LogTrace("CkfDebugger") << "number of input hits:" << meas.size() ;
  for(std::vector<TrajectoryMeasurement>::const_iterator tmpIt=meas.begin();tmpIt!=meas.end();tmpIt++){
    if (tmpIt->recHit()->isValid()) LogTrace("CkfDebugger") << "valid hit at position:" << tmpIt->recHit()->globalPosition() ;
  }
  theForwardPropagator = propagator;
  theChi2 = estimator;
  theMeasurementTracker = aMeasurementTracker;
  theGeomSearchTracker = theMeasurementTracker->geometricSearchTracker();
  theTTRHBuilder = aTTRHBuilder;
  unsigned int trajId = 0;
  if ( !correctTrajectory(traj, trajId)) {
    LogTrace("CkfDebugger") << "trajectory not correct" ;
    return true;
  } // only correct trajectories analysed
  LogTrace("CkfDebugger") << "correct trajectory" ;

  if (traj.measurements().size() == 2){
    if ( testSeed(traj.firstMeasurement().recHit(),traj.lastMeasurement().recHit(),traj.lastMeasurement().updatedState()) == -1 ) {
      LogTrace("CkfDebugger") << "Seed has delta" ;
      seedWithDelta++;
      return false;//true;//false?
    }
  }

  //const PSimHit* correctHit = nextCorrectHit(traj, trajId);
  //if ( correctHit == 0) return true; // no more simhits on this track
  std::vector<const PSimHit*> correctHits = nextCorrectHits(traj, trajId);
  if ( correctHits.empty()) return true; // no more simhits on this track

  for (std::vector<const PSimHit*>::iterator corHit=correctHits.begin();corHit!=correctHits.end();corHit++){
    for (std::vector<TM>::const_iterator i=meas.begin(); i!=meas.end(); i++) {
      if (correctMeas( *i, *corHit)) {
	LogTrace("CkfDebugger") << "Correct hit found at position " << i-meas.begin() ;
	return true;
      }
    }
  }

  //debug why the first hit in correctHits is not found 
  //FIXME should loop over all hits
  const PSimHit* correctHit = *(correctHits.begin());

  // correct hit not found
  edm::LogVerbatim("CkfDebugger") << std::endl << "CkfDebugger: problem found: correct hit not found by findCompatibleMeasurements" ;
  edm::LogVerbatim("CkfDebugger") << "The correct hit position is " << position(correctHit)  << " lp " << correctHit->localPosition() ;
  edm::LogVerbatim("CkfDebugger") << "The size of the meas vector is " << meas.size() ;
  dump[0]++;problems++;

  for (std::vector<TM>::const_iterator i=meas.begin(); i!=meas.end(); i++) {
    edm::LogVerbatim("CkfDebugger") << "Is the hit valid? " << i->recHit()->isValid() ;
    if (i->recHit()->isValid()) {
      edm::LogVerbatim("CkfDebugger") << "RecHit at " << i->recHit()->globalPosition()
				      << " layer " <<   ((i->recHit()->det()->geographicalId().rawId() >>16) & 0xF)
				      << " subdet " << i->recHit()->det()->geographicalId().subdetId() 
				      << " Chi2 " << i->estimate() ;
    }
    else if (i->recHit()->det() == nullptr) {
      edm::LogVerbatim("CkfDebugger") << "Invalid RecHit returned with zero Det pointer" ;
    }
    else if (i->recHit()->det() == det(correctHit)) {
      edm::LogVerbatim("CkfDebugger") << "Invalid hit returned in correct Det" ;
    }
    else {
      edm::LogVerbatim("CkfDebugger") << "Invalid hit returned in Det at gpos " << i->recHit()->det()->position()
				      << " correct Det is at " << det(correctHit)->position() ;
    }
  }

  //Look if the correct RecHit exists
  std::pair<CTTRHp, double> correctRecHit = 
    analyseRecHitExistance( *correctHit, traj.lastMeasurement().updatedState());
  if (correctRecHit.first==nullptr ) {
    //the hit does not exist or is uncorrectly matched
    if ( fabs(correctRecHit.second-0)<0.01 ) {dump[1]++;}//other
    if ( fabs(correctRecHit.second+1)<0.01 ) {dump[8]++;}//propagation
    if ( fabs(correctRecHit.second+2)<0.01 ) {dump[9]++;}//No component is found
    if ( fabs(correctRecHit.second+3)<0.01 ) {dump[10]++;}//Partner measurementDet not found
    if ( fabs(correctRecHit.second+4)<0.01 ) {dump[11]++;}//glued MeasurementDet not found
    if ( fabs(correctRecHit.second+5)<0.01 ) {dump[12]++;}//matched not found
    if ( fabs(correctRecHit.second+6)<0.01 ) {dump[13]++;}//Matched not associated
    if ( fabs(correctRecHit.second+7)<0.01 ) {dump[14]++;}//Only one component is found
    if ( fabs(correctRecHit.second+8)<0.01 ) {dump[15]++;}//not found (is not a glued det)
  }
  else {
    //the hit exists: why wasn't it found?
    int result = analyseRecHitNotFound(traj,correctRecHit.first);
    if (result == 5){
      if (correctRecHit.second>30) {
	edm::LogVerbatim("CkfDebugger") << "Outling RecHit at pos=" << correctRecHit.first->globalPosition()
					<< " from SimHit at pos="<< position(correctHit) 
					<< " det=" << correctHit->detUnitId() << " process=" << correctHit->processType() ;
	if (hasDelta(correctHit)){
	  edm::LogVerbatim("CkfDebugger") << "there are deltas on this det" ;
	  chi2gt30delta++;
	  dump5[pair<int,int>((correctRecHit.first->det()->geographicalId().subdetId()-1),(layer(correctRecHit.first->det()))-1)]++;  
	}else{
	  edm::LogVerbatim("CkfDebugger") << "no deltas on this det" ;
	  dump[5]++;
	  chi2gt30++;
	  dump3[pair<int,int>((correctRecHit.first->det()->geographicalId().subdetId()-1),(layer(correctRecHit.first->det()))-1)]++;
	  CTTRHp h1 = traj.measurements()[0].recHit();
	  CTTRHp h2 = traj.measurements()[1].recHit();
	  TSOS t = traj.measurements()[1].updatedState();
	  double chi2 = testSeed(h1,h2,t);
	  if (chi2==-1) {
	    edm::LogVerbatim("CkfDebugger") << "there were deltas in the seed" ;
	    chi2gt30deltaSeed++;
	  }
	  else {
	    hchi2seedProb->Fill(chi2);
	    edm::LogVerbatim("CkfDebugger") << "no deltas in the seed. What is wrong?" ;

	    TSOS detState = theForwardPropagator->propagate( traj.lastMeasurement().updatedState(), correctRecHit.first->det()->surface());
	    TSOS simDetState = theForwardPropagator->propagate( traj.lastMeasurement().updatedState(), det(correctHit)->surface());

	    if (true/*detState.globalMomentum().y()>0*/){
	      int subdetId = correctRecHit.first->det()->geographicalId().subdetId();
	      int layerId  = layer(correctRecHit.first->det());


	      LogTrace("CkfDebugger") << "position(correctHit)=" << position(correctHit) ;
	      LogTrace("CkfDebugger") << "correctRecHit.first->globalPosition()=" << correctRecHit.first->globalPosition() ;
	      LogTrace("CkfDebugger") << "detState.globalPosition()=" << detState.globalPosition() ;
	      LogTrace("CkfDebugger") << "simDetState.globalPosition()=" << simDetState.globalPosition() ;

	      LogTrace("CkfDebugger") << "correctHit->localPosition()=" << correctHit->localPosition() ;
	      LogTrace("CkfDebugger") << "correctRecHit.first->localPosition()=" << correctRecHit.first->localPosition() ;
	      LogTrace("CkfDebugger") << "correctRecHit.first->localPositionError()=" << correctRecHit.first->localPositionError() ;
	      LogTrace("CkfDebugger") << "detState.localPosition()=" << detState.localPosition() ;
	      LogTrace("CkfDebugger") << "detState.localError().positionError()=" << detState.localError().positionError() ;
	      LogTrace("CkfDebugger") << "simDetState.localPosition()=" << simDetState.localPosition() ;
	      LogTrace("CkfDebugger") << "simDetState.localError().positionError()=" << simDetState.localError().positionError() ;
	      double pullx_shrh = (correctHit->localPosition().x()-correctRecHit.first->localPosition().x())/
		sqrt(correctRecHit.first->localPositionError().xx());
	      double pully_shrh = 0;
	      if (correctRecHit.first->localPositionError().yy()!=0) 
		pully_shrh = (correctHit->localPosition().y()-correctRecHit.first->localPosition().y())/
		  sqrt(correctRecHit.first->localPositionError().yy());
	      double pullx_shst = (correctHit->localPosition().x()-simDetState.localPosition().x())/
		sqrt(simDetState.localError().positionError().xx());
	      double pully_shst = (correctHit->localPosition().y()-simDetState.localPosition().y())/
		sqrt(simDetState.localError().positionError().yy());

	      LogTrace("CkfDebugger") << "pullx(sh-rh)=" << pullx_shrh ;
	      LogTrace("CkfDebugger") << "pully(sh-rh)=" << pully_shrh ;
	      LogTrace("CkfDebugger") << "pullx(sh-st)=" << pullx_shst ;
	      LogTrace("CkfDebugger") << "pully(sh-st)=" << pully_shst ;

	      LogTrace("CkfDebugger") << "pullx(st-rh)=" << (detState.localPosition().x()-correctRecHit.first->localPosition().x())/
		sqrt(correctRecHit.first->localPositionError().xx()+detState.localError().positionError().xx()) ;

	      std::pair<double,double> pulls = computePulls(correctRecHit.first, detState);
	      if (subdetId>0 &&subdetId<7 && layerId>0 && layerId<10) {
		stringstream title;
		title.str("");
		title << "pullX_" << subdetId << "-" << layerId << "_sh-rh";
		hPullX_shrh[title.str()]->Fill( pullx_shrh );
		title.str("");
		title << "pullY_" << subdetId << "-" << layerId << "_sh-rh";
		hPullY_shrh[title.str()]->Fill( pully_shrh );
		title.str("");
		title << "pullX_" << subdetId << "-" << layerId <<"_sh-st";
		hPullX_shst[title.str()]->Fill( pullx_shst );
		title.str("");
		title << "pullY_" << subdetId << "-" << layerId <<"_sh-st";
		hPullY_shst[title.str()]->Fill( pully_shst );
		title.str("");
		title << "pullX_" << subdetId << "-" << layerId <<"_st-rh";
		hPullX_strh[title.str()]->Fill(pulls.first);
		title.str("");
		title << "pullY_" << subdetId << "-" << layerId <<"_st-rh";
		hPullY_strh[title.str()]->Fill(pulls.second);

		GlobalPoint shGPos = position(correctHit);
		GlobalPoint stGPos = simDetState.globalPosition();
		GlobalError stGPosErr = simDetState.cartesianError().position();
		double pullGPx = (shGPos.x()-stGPos.x())/sqrt(stGPosErr.cxx());
		title.str("");
		title << "PullGP_X_" << subdetId << "-" << layerId << "_sh-st";
		hPullGP_X_shst[title.str()]->Fill(pullGPx);
		title.str("");
		title << "PullGP_Y_" << subdetId << "-" << layerId << "_sh-st";
		hPullGP_Y_shst[title.str()]->Fill((shGPos.y()-stGPos.y())/sqrt(stGPosErr.cyy()));
		title.str("");
		title << "PullGP_Z_" << subdetId << "-" << layerId << "_sh-st";
		hPullGP_Z_shst[title.str()]->Fill((shGPos.z()-stGPos.z())/sqrt(stGPosErr.czz()));

		if (subdetId==3&&layerId==1){
		  hPullGPXvsGPX_shst->Fill(pullGPx,shGPos.x());
		  hPullGPXvsGPY_shst->Fill(pullGPx,shGPos.y());
		  hPullGPXvsGPZ_shst->Fill(pullGPx,shGPos.z());
		  hPullGPXvsGPr_shst->Fill(pullGPx,shGPos.mag());
		  hPullGPXvsGPeta_shst->Fill(pullGPx,shGPos.eta());
		  hPullGPXvsGPphi_shst->Fill(pullGPx,shGPos.phi());
		}
		if (dynamic_cast<const SiStripMatchedRecHit2D*>(correctRecHit.first->hit())) {
		  LogTrace("CkfDebugger") << "MONO HIT";
                  auto m = dynamic_cast<const SiStripMatchedRecHit2D*>(correctRecHit.first->hit())->monoHit();
		  CTTRHp tMonoHit = theTTRHBuilder->build(&m);
		  const PSimHit sMonoHit = *(hitAssociator->associateHit(*tMonoHit->hit()).begin());
		  TSOS monoState = theForwardPropagator->propagate( traj.lastMeasurement().updatedState(), tMonoHit->det()->surface());
		  double pullM_shrh = (sMonoHit.localPosition().x()-tMonoHit->localPosition().x())/
		    sqrt(tMonoHit->localPositionError().xx());
		  double pullM_shst = (sMonoHit.localPosition().x()-monoState.localPosition().x())/
		    sqrt(monoState.localError().positionError().xx());
		  std::pair<double,double> pullsMono = computePulls(tMonoHit, monoState);
		  title.str("");
		  title << "pullM_" << subdetId << "-" << layerId << "_sh-rh";
		  hPullM_shrh[title.str()]->Fill(pullM_shrh);
		  title.str("");
		  title << "pullM_" << subdetId << "-" << layerId << "_sh-st";
		  hPullM_shst[title.str()]->Fill(pullM_shst);
		  title.str("");
		  title << "pullM_" << subdetId << "-" << layerId << "_st-rh";
		  hPullM_strh[title.str()]->Fill(pullsMono.first);

		  LogTrace("CkfDebugger") << "STEREO HIT";
                  auto s= dynamic_cast<const SiStripMatchedRecHit2D*>(correctRecHit.first->hit())->stereoHit();
		  CTTRHp tStereoHit = theTTRHBuilder->build(&s);
		  const PSimHit sStereoHit = *(hitAssociator->associateHit(*tStereoHit->hit()).begin());
		  TSOS stereoState = theForwardPropagator->propagate( traj.lastMeasurement().updatedState(), tStereoHit->det()->surface());
		  double pullS_shrh = (sStereoHit.localPosition().x()-tStereoHit->localPosition().x())/
		    sqrt(tStereoHit->localPositionError().xx());
		  double pullS_shst = (sStereoHit.localPosition().x()-stereoState.localPosition().x())/
		    sqrt(stereoState.localError().positionError().xx());
		  std::pair<double,double> pullsStereo = computePulls(tStereoHit, stereoState);
		  title.str("");
		  title << "pullS_" << subdetId  << "-" << layerId << "_sh-rh";
		  hPullS_shrh[title.str()]->Fill(pullS_shrh);
		  title.str("");
		  title << "pullS_" << subdetId << "-" << layerId << "_sh-st";
		  hPullS_shst[title.str()]->Fill(pullS_shst);
		  title.str("");
		  title << "pullS_" << subdetId << "-" << layerId << "_st-rh";
		  hPullS_strh[title.str()]->Fill(pullsStereo.first);
		}
	      } else 
		edm::LogVerbatim("CkfDebugger") << "unexpected result: wrong det or layer id " 
						<< subdetId << " " << layerId << " " 
						<< correctRecHit.first->det()->geographicalId().rawId();
	    }
	  }
	}
      }
      else {
	edm::LogVerbatim("CkfDebugger") << "unexpected result " << correctRecHit.second ;
	dump[6]++;chi2ls30++;
      }
    }
    else dump[result]++;
    if (result == 3){
      dump2[pair<int,int>((correctRecHit.first->det()->geographicalId().subdetId()-1),(layer(correctRecHit.first->det()))-1)]++; 
    }
    if (result == 4){
      dump4[pair<int,int>((correctRecHit.first->det()->geographicalId().subdetId()-1),(layer(correctRecHit.first->det()))-1)]++; 
    }
    if (correctRecHit.second>30) {
      dump[7]++;totchi2gt30++; 
    }
  }
  return false;
}


bool CkfDebugger::correctTrajectory( const Trajectory& traj,unsigned int& trajId) const
{
  LogTrace("CkfDebugger") << "now in correctTrajectory" ;
  Trajectory::RecHitContainer hits = traj.recHits();

  std::vector<SimHitIdpr> currentTrackId = hitAssociator->associateHitId(*hits.front()->hit());
  if (currentTrackId.empty()) return false;

  for (Trajectory::RecHitContainer::const_iterator rh=hits.begin(); rh!=hits.end(); ++rh) {

    //if invalid hit exit
    if (!(*rh)->hit()->isValid()) {
      //LogTrace("CkfDebugger") << "invalid hit" ;
      return false;
    }

    //if hits from deltas exit
    bool nogoodhit = true;
    std::vector<PSimHit> assSimHits = hitAssociator->associateHit(*(*rh)->hit());
    for (std::vector<PSimHit>::iterator shit=assSimHits.begin();shit!=assSimHits.end();shit++){
      if (goodSimHit(*shit)) nogoodhit=false;
    }
    if (nogoodhit) return false;
    
    //all hits must be associated to the same sim track
    bool test = true;
    std::vector<SimHitIdpr> nextTrackId = hitAssociator->associateHitId(*(*rh)->hit());
    for (std::vector<SimHitIdpr>::iterator i=currentTrackId.begin();i!=currentTrackId.end();i++){
      for (std::vector<SimHitIdpr>::iterator j=nextTrackId.begin();j!=nextTrackId.end();j++){
	if (i->first == j->first) test = false;
	//LogTrace("CkfDebugger") << "valid " << *i << " " << *j ;
	trajId = j->first;
      }
    }
    if (test) {/*LogTrace("CkfDebugger") << "returning false" ;*/return false;}
    //     std::vector<PSimHit*> simTrackHits = idHitsMap[trajId];
    //     if (!goodSimHit(simTrackHits.))
  }
  //LogTrace("CkfDebugger") << "returning true" ;
  return true;
}

int CkfDebugger::assocTrackId(CTTRHp rechit) const
{
  LogTrace("CkfDebugger") << "now in assocTrackId" ;

  if (!rechit->hit()->isValid()) {
    return -1;
  }

  std::vector<SimHitIdpr> ids = hitAssociator->associateHitId(*rechit->hit());
  if (!ids.empty()) {
    return ids[0].first;//FIXME if size>1!!
  }
  else {
    return -1;
  }
}


vector<const PSimHit*> CkfDebugger::nextCorrectHits( const Trajectory& traj, unsigned int& trajId)
{
  std::vector<const PSimHit*> result;
  // find the component of the RecHit at largest distance from origin (FIXME: should depend on propagation direction)
  LogTrace("CkfDebugger") << "now in nextCorrectHits" ;
  TransientTrackingRecHit::ConstRecHitPointer lastRecHit = traj.lastMeasurement().recHit();
  TransientTrackingRecHit::RecHitContainer comp = lastRecHit->transientHits();
  if (!comp.empty()) {
    float maxR = 0;
    for (TransientTrackingRecHit::RecHitContainer::const_iterator ch=comp.begin(); 
	 ch!=comp.end(); ++ch) {
      if ((*ch)->globalPosition().mag() > maxR) lastRecHit = *ch; 
      maxR = (*ch)->globalPosition().mag();
    }
  }
  edm::LogVerbatim("CkfDebugger") << "CkfDebugger: lastRecHit is at gpos " << lastRecHit->globalPosition() 
				  << " layer " << layer((lastRecHit->det())) 
				  << " subdet " << lastRecHit->det()->geographicalId().subdetId() ;

  //find the simHits associated to the recHit
  const std::vector<PSimHit>& pSimHitVec = hitAssociator->associateHit(*lastRecHit->hit());
  for (std::vector<PSimHit>::const_iterator shit=pSimHitVec.begin();shit!=pSimHitVec.end();shit++){
    const GeomDetUnit* detUnit = theTrackerGeom->idToDetUnit( DetId(shit->detUnitId()));
    LogTrace("CkfDebugger") << "from hitAssociator SimHits are at GP=" << detUnit->toGlobal( shit->localPosition())
				    << " traId=" << shit->trackId() << " particleType " << shit->particleType() 
				    << " pabs=" << shit->pabs() << " detUnitId=" << shit->detUnitId() << " layer " << layer((det(&*shit))) 
				    << " subdet " << det(&*shit)->geographicalId().subdetId() ;
  }

  //choose the simHit from the same track that has the highest tof
  const PSimHit * lastPSH = nullptr;
  if (!pSimHitVec.empty()) {
    float maxTOF = 0;
    for (std::vector<PSimHit>::const_iterator ch=pSimHitVec.begin(); ch!=pSimHitVec.end(); ++ch) {
      if ( ( ch->trackId()== trajId) && (ch->timeOfFlight() > maxTOF)  && ( goodSimHit(*ch) )) {
	lastPSH = &*ch; 
	maxTOF = lastPSH->timeOfFlight();
      }
    }
  }
  else return result;//return empty vector: no more hits on the sim track
  if (lastPSH == nullptr) return result; //return empty vector: no more good hits on the sim track
  edm::LogVerbatim("CkfDebugger") << "CkfDebugger: corresponding SimHit is at gpos " << position(&*lastPSH) ;

  //take the simHits on the simTrack that are in the nextLayer (could be > 1 if overlap or matched)
  std::vector<PSimHit*> trackHits = idHitsMap[trajId];
  if (fabs((double)(trackHits.back()->detUnitId()-lastPSH->detUnitId()))<1 ) return result;//end of sim track
  std::vector<PSimHit*>::iterator currentIt = trackHits.end();
  for (std::vector<PSimHit*>::iterator it=trackHits.begin();
       it!=trackHits.end();it++){
    if (goodSimHit(**it) && //good hit
	( lastPSH->timeOfFlight()<(*it)->timeOfFlight() ) && //greater tof
	//( fabs((double)((*it)->detUnitId()-(lastPSH->detUnitId()) ))>1) && //not components of the same matched hit
	( (det(lastPSH)->geographicalId().subdetId()!=det(*it)->geographicalId().subdetId()) || 
	  (layer(det(lastPSH))!=layer(det(*it)) ) ) //change layer or detector(tib,tob,...)
	){
      edm::LogVerbatim("CkfDebugger") << "Next good PSimHit is at gpos " << position(*it) ;
      result.push_back(*it);
      currentIt = it;
      break;
    }
  }
  bool samelayer = true;
  if (currentIt!=(trackHits.end()-1) && currentIt!=trackHits.end()) {
    for (std::vector<PSimHit*>::iterator nextIt = currentIt; (samelayer && nextIt!=trackHits.end()) ;nextIt++){
      if (goodSimHit(**nextIt)){
	if ( (det(*nextIt)->geographicalId().subdetId()==det(*currentIt)->geographicalId().subdetId()) && 
	     (layer(det(*nextIt))==layer(det(*currentIt)) ) ) {
	  result.push_back(*nextIt);
	}
	else samelayer = false;
      }
    }
  }
  
  return result;
}

bool CkfDebugger::goodSimHit(const PSimHit& sh) const
{
  if (sh.pabs() > 0.9) return true; // GeV, reject delta rays from association
  else return false;
}


bool CkfDebugger::associated(CTTRHp rechit, const PSimHit& pSimHit) const
{
  LogTrace("CkfDebugger") << "now in associated" ;

  if (!rechit->isValid()) return false;
  //   LogTrace("CkfDebugger") << "rec hit valid" ;
  const std::vector<PSimHit>& pSimHitVec = hitAssociator->associateHit(*rechit->hit());
  //   LogTrace("CkfDebugger") << "size=" << pSimHitVec.size() ;
  for (std::vector<PSimHit>::const_iterator shit=pSimHitVec.begin();shit!=pSimHitVec.end();shit++){
    //const GeomDetUnit* detUnit = theTrackerGeom->idToDetUnit( DetId(shit->detUnitId()));
    //         LogTrace("CkfDebugger") << "pSimHit.timeOfFlight()=" << pSimHit.timeOfFlight() 
    //     	 << " pSimHit.pabs()=" << pSimHit.pabs() << " GP=" << position(&pSimHit);
    //         LogTrace("CkfDebugger") << "(*shit).timeOfFlight()=" << (*shit).timeOfFlight() 
    //     	 << " (*shit).pabs()=" << (*shit).pabs() << " GP=" << detUnit->toGlobal( shit->localPosition());
    if ( ( fabs((*shit).timeOfFlight()-pSimHit.timeOfFlight())<1e-9  ) && 
	 ( fabs((*shit).pabs()-pSimHit.pabs())<1e-9 ) ) return true;
  }
  return false;
}

bool CkfDebugger::correctMeas( const TM& tm, const PSimHit* correctHit) const
{
  LogTrace("CkfDebugger") << "now in correctMeas" ;
  const CTTRHp& recHit = tm.recHit();
  if (recHit->isValid()) LogTrace("CkfDebugger") << "hit at position:" << recHit->globalPosition() ;
  TransientTrackingRecHit::RecHitContainer comp = recHit->transientHits();
  if (comp.empty()) {
    //     LogTrace("CkfDebugger") << "comp.empty()==true" ;
    return associated( recHit, *correctHit);
  }
  else {
    for (TransientTrackingRecHit::RecHitContainer::const_iterator ch=comp.begin(); 
	 ch!=comp.end(); ++ch) {
      if ( associated( recHit, *correctHit)) {
	// check if the other components are associated to the same trackId
	for (TransientTrackingRecHit::RecHitContainer::const_iterator ch2=comp.begin(); 
	     ch2!=comp.end(); ++ch2) {
	  if (ch2 == ch) continue;
	  //////////
	  // 	  LogTrace("CkfDebugger") << "correctHit->trackId()=" << correctHit->trackId() ;
	  bool test=true;
	  std::vector<SimHitIdpr> ids = hitAssociator->associateHitId(*(*ch2)->hit());
	  for (std::vector<SimHitIdpr>::iterator j=ids.begin();j!=ids.end();j++){
	    // 	    LogTrace("CkfDebugger") << "j=" <<j->first;
	    if (correctHit->trackId()==j->first) {
	      test=false;
	      // 	      LogTrace("CkfDebugger") << correctHit->trackId()<< " " <<j->first;
	    }
	  }
	  if (assocTrackId( *ch2) != ((int)( correctHit->trackId())) ) {LogTrace("CkfDebugger") << "returning false 1" ;/*return false;*/}//fixme
	  if (test) {
	    // 	    LogTrace("CkfDebugger") << "returning false 2" ;
	    return false; // not all components from same simtrack
	  }
	  // 	  if (assocTrackId( **ch2) != ((int)( correctHit->trackId())) ) {
	  // 	    return false; // not all components from same simtrack
	  // 	  }
	}
	return true; // if all components from same simtrack
      }
    }
    return false;
  }
}

//this checks only if there is the rechit on the det where the sim hit is 
pair<CTTRHp, double> CkfDebugger::analyseRecHitExistance( const PSimHit& sh, const TSOS& startingState)
{
  LogTrace("CkfDebugger") << "now in analyseRecHitExistance" ;

#if 0
  std::pair<CTTRHp, double> result;
  
  const MeasurementDet* simHitDet = theMeasurementTracker->idToDet( DetId( sh.detUnitId()));
  TSOS simHitState = TSOSFromSimHitFactory()(sh, *det(&sh), *theMagField);
  MeasurementDet::RecHitContainer recHits = simHitDet->recHits( simHitState);//take all hits from det

  //check if the hit is not present or is a problem of association
  TSOS firstDetState = theForwardPropagator->propagate( startingState, det(&sh)->surface());
  if (!firstDetState.isValid()) {
    edm::LogVerbatim("CkfDebugger") << "CkfDebugger: propagation failed from state " << startingState << " to first det surface " 
				    << position(&sh) ;
    propagation++;
    return std::pair<CTTRHp, double>((CTTRHp)(0),-1);
  }

  bool found = false;
  for ( MeasurementDet::RecHitContainer::const_iterator rh = recHits.begin(); rh != recHits.end(); rh++) {
    if ( associated( *rh, sh)) {
      found = true;
      result = std::pair<CTTRHp, double>(*rh,theChi2->estimate( firstDetState, **rh).second);
      edm::LogVerbatim("CkfDebugger") << "CkfDebugger: A RecHit associated to the correct Simhit exists at lpos " 
				      << (**rh).localPosition()
				      << " gpos " << (**rh).globalPosition()
				      << " layer " <<   layer((**rh).det())
				      << " subdet " << (**rh).det()->geographicalId().subdetId() 
				      << " Chi2 " << theChi2->estimate( firstDetState, **rh).second;
    }
  }
  if (!found) {
    edm::LogVerbatim("CkfDebugger") << "CkfDebugger: there is no RecHit associated to the correct SimHit." ;
    edm::LogVerbatim("CkfDebugger") << " There are " <<  recHits.size() << " RecHits in the simHit DetUnit" ;
    edm::LogVerbatim("CkfDebugger") << "SH GP=" << position(&sh) << " subdet=" << det(&sh)->geographicalId().subdetId() 
				    << " layer=" << layer(det(&sh)) ;
    int y=0;
    for (MeasurementDet::RecHitContainer::const_iterator rh = recHits.begin(); rh != recHits.end(); rh++)
      edm::LogVerbatim("CkfDebugger") << "RH#" << y++ << " GP=" << (**rh).globalPosition() << " subdet=" << (**rh).det()->geographicalId().subdetId() 
				      << " layer=" << layer((**rh).det()) ;
    for ( MeasurementDet::RecHitContainer::const_iterator rh = recHits.begin(); rh != recHits.end(); rh++) {
      edm::LogVerbatim("CkfDebugger") << "Non-associated RecHit at pos " << (**rh).localPosition() ;
    }
  }

  bool found2 = false;
  const PSimHit* sh2;
  StripSubdetector subdet( det(&sh)->geographicalId());
  if (!subdet.glued()) {
    edm::LogVerbatim("CkfDebugger") << "The DetUnit is not part of a GluedDet" ;
    if (found) {
      if (result.second>30){
	LogTrace("CkfDebugger") << "rh->parameters()=" << result.first->parameters() ;
	LogTrace("CkfDebugger") << "rh->parametersError()=" << result.first->parametersError() ;
	MeasurementExtractor me(firstDetState);
	AlgebraicVector r(result.first->parameters() - me.measuredParameters(*result.first));
	LogTrace("CkfDebugger") << "me.measuredParameters(**rh)=" << me.measuredParameters(*result.first) ;
	LogTrace("CkfDebugger") << "me.measuredError(**rh)=" << me.measuredError(*result.first) ;
	AlgebraicSymMatrix R(result.first->parametersError() + me.measuredError(*result.first));
	LogTrace("CkfDebugger") << "r=" << r ;
	LogTrace("CkfDebugger") << "R=" << R ;
	int ierr; 
	R.invert(ierr);
	LogTrace("CkfDebugger") << "R(-1)=" << R ;
	LogTrace("CkfDebugger") << "chi2=" << R.similarity(r) ;
      }
      return result;
    }
    else {
      simple_hit_not_found++;
      return std::pair<CTTRHp, double>((CTTRHp)(0),-8);//not found (is not a glued det)
    }
  } else {
    edm::LogVerbatim("CkfDebugger") << "The DetUnit is part of a GluedDet" ;
    DetId partnerDetId = DetId( subdet.partnerDetId());

    sh2 = pSimHit( sh.trackId(), partnerDetId);
    if (sh2 == 0) {
      edm::LogVerbatim("CkfDebugger") << "Partner DetUnit does not have a SimHit from the same track" ;
      if (found) {
	//projected rec hit
	TrackingRecHitProjector<ProjectedRecHit2D> proj;
	DetId gid = gluedId( subdet);
	const MeasurementDet* gluedDet = theMeasurementTracker->idToDet( gid);
	TSOS gluedTSOS = theForwardPropagator->propagate(startingState, gluedDet->geomDet().surface());
	CTTRHp projHit = proj.project( *result.first,gluedDet->geomDet(),gluedTSOS).get();
	//LogTrace("CkfDebugger") << proj.project( *result.first,gluedDet->geomDet(),gluedTSOS)->parameters() ;
	//LogTrace("CkfDebugger") << projHit->parametersError() ;
	double chi2 = theChi2->estimate(gluedTSOS, *proj.project( *result.first,gluedDet->geomDet(),gluedTSOS)).second;
	return std::pair<CTTRHp, double>(projHit,chi2);
      }
    }
    else {
      edm::LogVerbatim("CkfDebugger") << "Partner DetUnit has a good SimHit at gpos " << position(sh2) 
				      << " lpos " << sh2->localPosition() ;
      //}
    
      const MeasurementDet* partnerDet = theMeasurementTracker->idToDet( partnerDetId);
      if (partnerDet == 0) {
	edm::LogVerbatim("CkfDebugger") << "Partner measurementDet not found!!!" ;
	partner_det_not_fuond++;
	return std::pair<CTTRHp, double>((CTTRHp)(0),-3);
      }
      TSOS simHitState2 = TSOSFromSimHitFactory()(*sh2, *det(sh2), *theMagField);
      MeasurementDet::RecHitContainer recHits2 = partnerDet->recHits( simHitState2);

      TSOS secondDetState = theForwardPropagator->propagate( startingState, det(sh2)->surface());
      if (!secondDetState.isValid()) {
	edm::LogVerbatim("CkfDebugger") << "CkfDebugger: propagation failed from state " << startingState << " to second det surface " 
					<< position(sh2) ;
	propagation++;
	return std::pair<CTTRHp, double>((CTTRHp)(0),-1);
      }

      for ( MeasurementDet::RecHitContainer::const_iterator rh = recHits2.begin(); rh != recHits2.end(); rh++) {
	if ( associated( *rh, *sh2)) {
	  found2 = true;
	  edm::LogVerbatim("CkfDebugger") << "CkfDebugger: A RecHit associated to the correct Simhit exists at lpos " 
					  << (**rh).localPosition()
					  << " gpos " << (**rh).globalPosition()
					  << " Chi2 " << theChi2->estimate( secondDetState, **rh).second
	    ;
	}
      }
      if (!found2) {
	edm::LogVerbatim("CkfDebugger") << "CkfDebugger: there is no RecHit associated to the correct SimHit." ;
	LogTrace("CkfDebugger") << " There are " <<  recHits.size() << " RecHits in the simHit DetUnit" ;
	for ( MeasurementDet::RecHitContainer::const_iterator rh = recHits.begin(); rh != recHits.end(); rh++) {
	  LogTrace("CkfDebugger") << "Non-associated RecHit at pos " << (**rh).localPosition() ;
	}
      }
    }
  }

  MeasurementDet::RecHitContainer gluedHits;
  if (found && found2) {
    // look in the glued det
    DetId gid = gluedId( subdet);
    const MeasurementDet* gluedDet = theMeasurementTracker->idToDet( gid);
    if ( gluedDet == 0) {
      edm::LogVerbatim("CkfDebugger") << "CkfDebugger ERROR: glued MeasurementDet not found!" ;
      glued_det_not_fuond++;
      return std::pair<CTTRHp, double>((CTTRHp)(0),-4);
    }

    TSOS gluedDetState = theForwardPropagator->propagate( startingState, gluedDet->surface());
    if (!gluedDetState.isValid()) {
      edm::LogVerbatim("CkfDebugger") << "CkfDebugger: propagation failed from state " << startingState << " to det surface " 
				      << gluedDet->position() ;
      propagation++;
      return std::pair<CTTRHp, double>((CTTRHp)(0),-1);
    }

    gluedHits = gluedDet->recHits( gluedDetState);
    edm::LogVerbatim("CkfDebugger") << "CkfDebugger: the GluedDet returned " << gluedHits.size() << " hits" ;
    if (gluedHits.size()==0){
      edm::LogVerbatim("CkfDebugger") << "Found and associated mono and stereo recHits but not matched!!!" ;
      matched_not_found++;
      return std::pair<CTTRHp, double>((CTTRHp)(0),-5);
    } 
    bool found3 = false;
    for ( MeasurementDet::RecHitContainer::const_iterator rh = gluedHits.begin(); rh != gluedHits.end(); rh++) {
      if ( associated( *rh, sh) && associated( *rh, *sh2)) {
	double chi2 = theChi2->estimate(gluedDetState, **rh).second;
	edm::LogVerbatim("CkfDebugger") << "Matched hit at lpos " << (**rh).localPosition()
					<< " gpos " << (**rh).globalPosition()
					<< " has Chi2 " << chi2
	  ;
	result = std::pair<CTTRHp, double>(&**rh,chi2);
	found3 = true;
	if (chi2>30){
	  LogTrace("CkfDebugger") << "rh->parameters()=" << (*rh)->parameters() ;
	  LogTrace("CkfDebugger") << "rh->parametersError()=" << (*rh)->parametersError() ;
	  MeasurementExtractor me(gluedDetState);
	  AlgebraicVector r((*rh)->parameters() - me.measuredParameters(**rh));
	  LogTrace("CkfDebugger") << "me.measuredParameters(**rh)=" << me.measuredParameters(**rh) ;
	  LogTrace("CkfDebugger") << "me.measuredError(**rh)=" << me.measuredError(**rh) ;
	  AlgebraicSymMatrix R((*rh)->parametersError() + me.measuredError(**rh));
	  LogTrace("CkfDebugger") << "r=" << r ;
	  LogTrace("CkfDebugger") << "R=" << R ;
	  int ierr; 
	  R.invert(ierr);
	  LogTrace("CkfDebugger") << "R(-1)=" << R ;
	  LogTrace("CkfDebugger") << "chi2=" << R.similarity(r) ;
	}
	break;
      }
    }
    if (found3) return result;
    else {
      edm::LogVerbatim("CkfDebugger") << "Found and associated mono and stereo recHits. Matched found but not associated!!!" ;
      matched_not_associated++;
      return std::pair<CTTRHp, double>((CTTRHp)(0),-6);
    }
  }
  else if ( (found && !found2) || (!found && found2) ) {
    edm::LogVerbatim("CkfDebugger") << "Only one component is found" ;
    only_one_component++;
    return std::pair<CTTRHp, double>((CTTRHp)(0),-7);
  }
  else {
    edm::LogVerbatim("CkfDebugger") << "No component is found" ;
    no_component++;
    return std::pair<CTTRHp, double>((CTTRHp)(0),-2);
  }
  other++;
#endif
  return std::pair<CTTRHp, double>((CTTRHp)(nullptr),0);//other
}

const PSimHit* CkfDebugger::pSimHit(unsigned int tkId, DetId detId)
{
  for (std::vector<PSimHit*>::iterator shi=idHitsMap[tkId].begin(); shi!=idHitsMap[tkId].end(); ++shi) {    
    if ( (*shi)->detUnitId() == detId.rawId() && 
	 //(shi)->trackId() == tkId &&
	 goodSimHit(**shi) ) {
      return (*shi);
    }
  }
  return nullptr;
}

int CkfDebugger::analyseRecHitNotFound(const Trajectory& traj, CTTRHp correctRecHit)
{
  unsigned int correctDetId = correctRecHit->det()->geographicalId().rawId();
  int correctLayId = layer(correctRecHit->det());
  LogTrace("CkfDebugger") << "correct layer id=" << correctLayId ;

  TSOS currentState( traj.lastMeasurement().updatedState() );
  std::vector<const DetLayer*> nl = theNavSchool->nextLayers(*traj.lastLayer(),*currentState.freeState(),traj.direction() );
  if (nl.empty()) {
    edm::LogVerbatim("CkfDebugger") << "no compatible layers" ;
    no_layer++;return 2;
  }

  TkLayerLess lless;//FIXME - was lless(traj.direction())
  const DetLayer* detLayer = nullptr;
  bool navLayerAfter = false;
  bool test = false;
  for (std::vector<const DetLayer*>::iterator il = nl.begin(); il != nl.end(); il++) {
    if ( dynamic_cast<const BarrelDetLayer*>(*il) ){
      const BarrelDetLayer* pbl = dynamic_cast<const BarrelDetLayer*>(*il);
      LogTrace("CkfDebugger") << "pbl->specificSurface().bounds().length()=" << pbl->specificSurface().bounds().length() ;
      LogTrace("CkfDebugger") << "pbl->specificSurface().bounds().width()=" << pbl->specificSurface().bounds().width() ;
    }
    int layId = layer(((*(*il)->basicComponents().begin())));
    LogTrace("CkfDebugger") << " subdet=" << (*(*il)->basicComponents().begin())->geographicalId().subdetId() << "layer id=" << layId ;
    if (layId==correctLayId) {
      test = true;
      detLayer = &**il;
      break;
    }
    if ( lless( *il, theGeomSearchTracker->detLayer(correctRecHit->det()->geographicalId()) )) 
      navLayerAfter = true; //it is enough that only one layer is after the correct one?
  }

  if (test) {
    edm::LogVerbatim("CkfDebugger") << "correct layer taken into account. layer id: " << correctLayId ;
  } else if (navLayerAfter){
    edm::LogVerbatim("CkfDebugger")<< "SimHit layer after the layers returned by Navigation.";
    edm::LogVerbatim("CkfDebugger")<< "Probably a missing SimHit." ;
    edm::LogVerbatim("CkfDebugger")<< "check: " << (correctRecHit->det()->geographicalId().subdetId()) << " " << (layer(correctRecHit->det()));
    dump6[pair<int,int>((correctRecHit->det()->geographicalId().subdetId()-1),(layer(correctRecHit->det()))-1)]++; 
    no_sim_hit++;return 16;
  }
  else {
    edm::LogVerbatim("CkfDebugger") << "correct layer NOT taken into account. correct layer id: " << correctLayId ;
    layer_not_found++;
    return 3;
  }

  typedef DetLayer::DetWithState  DetWithState;
  std::vector<DetWithState> compatDets = detLayer->compatibleDets(currentState,*theForwardPropagator,*theChi2);
  //   LogTrace("CkfDebugger") << "DEBUGGER" ;
  //   LogTrace("CkfDebugger") << "runned compatDets." ;
  //   LogTrace("CkfDebugger") << "started from the following TSOS:" ;
  //   LogTrace("CkfDebugger") << currentState ;
  //   LogTrace("CkfDebugger") << "number of dets found=" << compatDets.size() ;
  //   for (std::vector<DetWithState>::iterator det=compatDets.begin();det!=compatDets.end();det++){
  //     unsigned int detId = det->first->geographicalId().rawId();
  //     LogTrace("CkfDebugger") << "detId=" << detId ; 
  //   }
  bool test2 = false;
  for (std::vector<DetWithState>::iterator det=compatDets.begin();det!=compatDets.end();det++){
    unsigned int detId = det->first->geographicalId().rawId();
    //     LogTrace("CkfDebugger") << "detId=" << detId 
    // 	 << "\ncorrectRecHit->det()->geographicalId().rawId()=" << correctRecHit->det()->geographicalId().rawId() 
    // 	 << "\ngluedId(correctRecHit->det()->geographicalId()).rawId()=" << gluedId(correctRecHit->det()->geographicalId()).rawId()
    // 	 ; 
    if (detId==gluedId(correctRecHit->det()->geographicalId()).rawId()) {
      test2=true;
      break;
    }
  }
  
  if (test2){
    edm::LogVerbatim("CkfDebugger") << "correct det taken into account. correctDetId is: " << correctDetId 
				    << ". please check chi2." ;
    return 5;
  }
  else {
    edm::LogVerbatim("CkfDebugger") << "correct det NOT taken into account. correctDetId: " << correctDetId ;
    det_not_found++;return 4;
  }

}

double CkfDebugger::testSeed(CTTRHp recHit1, CTTRHp recHit2, TSOS state){
  //edm::LogVerbatim("CkfDebugger") << "CkfDebugger::testSeed";
  //test Deltas
  const std::vector<PSimHit>& pSimHitVec1 = hitAssociator->associateHit(*recHit1->hit());
  const std::vector<PSimHit>& pSimHitVec2 = hitAssociator->associateHit(*recHit2->hit());
  
  if ( pSimHitVec1.empty() || pSimHitVec2.empty() || hasDelta(&(*pSimHitVec1.begin())) || hasDelta(&(*pSimHitVec2.begin())) ) {
    edm::LogVerbatim("CkfDebugger") << "Seed has delta or problems" ;
    return -1;
  }

  //   LogTrace("CkfDebugger") << "state=\n" << state ;
  //   double stlp1 = state.localParameters().vector()[0];
  //   double stlp2 = state.localParameters().vector()[1];
  //   double stlp3 = state.localParameters().vector()[2];
  //   double stlp4 = state.localParameters().vector()[3];
  //   double stlp5 = state.localParameters().vector()[4];

  if (!pSimHitVec2.empty()) {
    const PSimHit& simHit = *pSimHitVec2.begin();
    
    double shlp1 = -1/simHit.momentumAtEntry().mag();
    double shlp2 = simHit.momentumAtEntry().x()/simHit.momentumAtEntry().z();
    double shlp3 = simHit.momentumAtEntry().y()/simHit.momentumAtEntry().z();
    double shlp4 = simHit.localPosition().x();
    double shlp5 = simHit.localPosition().y();
    AlgebraicVector5 v;
    v[0] = shlp1;
    v[1] = shlp2;
    v[2] = shlp3;
    v[3] = shlp4;
    v[4] = shlp5;
  
    //     LogTrace("CkfDebugger") << "simHit.localPosition()=" << simHit.localPosition() ;
    //     LogTrace("CkfDebugger") << "simHit.momentumAtEntry()=" << simHit.momentumAtEntry() ;
    //     LogTrace("CkfDebugger") << "recHit2->localPosition()=" << recHit2->localPosition() ;
    //     LogTrace("CkfDebugger") << "recHit2->localPositionError()=" << recHit2->localPositionError() ;
    //     LogTrace("CkfDebugger") << "state.localPosition()=" << state.localPosition() ;
    //     LogTrace("CkfDebugger") << "state.localError().positionError()=" << state.localError().positionError() ;

    //     LogTrace("CkfDebugger") << "pullx(sh-rh)=" << (simHit.localPosition().x()-recHit2->localPosition().x())/sqrt(recHit2->localPositionError().xx()) ;
    //     LogTrace("CkfDebugger") << "pullx(sh-st)=" << (simHit.localPosition().x()-state.localPosition().x())/sqrt(state.localError().positionError().xx()) ;
    //     LogTrace("CkfDebugger") << "pullx(st-rh)=" << (state.localPosition().x()-recHit2->localPosition().x())/
    //       sqrt(recHit2->localPositionError().xx()+state.localError().positionError().xx()) ;

    //     LogTrace("CkfDebugger") << "local parameters" ;
    //     LogTrace("CkfDebugger") << left;
    //     LogTrace("CkfDebugger") << setw(15) << stlp1 << setw(15) << shlp1 << setw(15) << sqrt(state.localError().matrix()[0][0]) 
    // 	 << setw(15) << (stlp1-shlp1)/stlp1 << setw(15) << (stlp1-shlp1)/sqrt(state.localError().matrix()[0][0]) ;
    //     LogTrace("CkfDebugger") << setw(15) << stlp2 << setw(15) << shlp2 << setw(15) << sqrt(state.localError().matrix()[1][1]) 
    // 	 << setw(15) << (stlp2-shlp2)/stlp2 << setw(15) << (stlp2-shlp2)/sqrt(state.localError().matrix()[1][1]) ;
    //     LogTrace("CkfDebugger") << setw(15) << stlp3 << setw(15) << shlp3 << setw(15) << sqrt(state.localError().matrix()[2][2]) 
    //       << setw(15) << (stlp3-shlp3)/stlp3 << setw(15) << (stlp3-shlp3)/sqrt(state.localError().matrix()[2][2]) ;
    //     LogTrace("CkfDebugger") << setw(15) << stlp4 << setw(15) << shlp4 << setw(15) << sqrt(state.localError().matrix()[3][3]) 
    //       << setw(15) << (stlp4-shlp4)/stlp4 << setw(15) << (stlp4-shlp4)/sqrt(state.localError().matrix()[3][3]) ;
    //     LogTrace("CkfDebugger") << setw(15) << stlp5 << setw(15) << shlp5 << setw(15) << sqrt(state.localError().matrix()[4][4]) << 
    //       setw(15) << (stlp5-shlp5)/stlp5 << setw(15) << (stlp5-shlp5)/sqrt(state.localError().matrix()[4][4]) ;

    AlgebraicSymMatrix55 R = state.localError().matrix();
    R.Invert();
    double chi2 = ROOT::Math::Similarity(v-state.localParameters().vector(), R);
    LogTrace("CkfDebugger") << "chi2=" << chi2 ;
    return chi2;
  }

  return 0;//fixme

}


CkfDebugger::~CkfDebugger(){
  for (int it=0; it!=((int)(dump.size())); it++)
    edm::LogVerbatim("CkfDebugger") << "dump " << it << " " << dump[it] ;
  
  edm::LogVerbatim("CkfDebugger") ;
  edm::LogVerbatim("CkfDebugger") << "seedWithDelta=" <<  ((double)seedWithDelta/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "problems=" << ((double)problems/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "no_sim_hit=" << ((double)no_sim_hit/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "no_layer=" << ((double)no_layer/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "layer_not_found=" << ((double)layer_not_found/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "det_not_found=" << ((double)det_not_found/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "chi2gt30=" << ((double)chi2gt30/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "chi2gt30deltaSeed=" << ((double)chi2gt30deltaSeed/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "chi2gt30delta=" << ((double)chi2gt30delta/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "chi2ls30=" << ((double)chi2ls30/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "simple_hit_not_found=" << ((double)simple_hit_not_found/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "no_component=" << ((double)no_component/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "only_one_component=" << ((double)only_one_component/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "matched_not_found=" << ((double)matched_not_found/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "matched_not_associated=" << ((double)matched_not_associated/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "partner_det_not_fuond=" << ((double)partner_det_not_fuond/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "glued_det_not_fuond=" << ((double)glued_det_not_fuond/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "propagation=" << ((double)propagation/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "other=" << ((double)other/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "totchi2gt30=" << ((double)totchi2gt30/totSeeds) ;
  edm::LogVerbatim("CkfDebugger") << "totSeeds=" << totSeeds ;
  edm::LogVerbatim("CkfDebugger") ;
  
  edm::LogVerbatim("CkfDebugger") << "layer navigation problems:" ;
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      edm::LogVerbatim("CkfDebugger") << "det=" << i+1 << " lay=" << j+1 << " " << dump2[pair<int,int>(i,j)] ;
    }
  edm::LogVerbatim("CkfDebugger") << "\nlayer with hit having chi2>30:" ;
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      edm::LogVerbatim("CkfDebugger") << "det=" << i+1 << " lay=" << j+1 << " " << dump3[pair<int,int>(i,j)] ;
    }
  edm::LogVerbatim("CkfDebugger") << "\nlayer with hit having chi2>30 for delta rays:" ;
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      edm::LogVerbatim("CkfDebugger") << "det=" << i+1 << " lay=" << j+1 << " " << dump5[pair<int,int>(i,j)] ;
    }
  edm::LogVerbatim("CkfDebugger") << "\nlayer with det not found:" ;
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      edm::LogVerbatim("CkfDebugger") << "det=" << i+1 << " lay=" << j+1 << " " << dump4[pair<int,int>(i,j)] ;
    }
  edm::LogVerbatim("CkfDebugger") << "\nlayer with correct RecHit after missing Sim Hit:" ;
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      edm::LogVerbatim("CkfDebugger") << "det=" << i+1 << " lay=" << j+1 << " " << dump6[pair<int,int>(i,j)] ;
    }
  hchi2seedAll->Write();
  hchi2seedProb->Write();
  std::stringstream title;
  for (int i=0; i!=6; i++)
    for (int j=0; j!=9; j++){
      if (i==0 && j>2) break;
      if (i==1 && j>1) break;
      if (i==2 && j>3) break;
      if (i==3 && j>2) break;
      if (i==4 && j>5) break;
      if (i==5 && j>8) break;
      title.str("");
      title << "pullX_" << i+1 << "-" << j+1 << "_sh-rh";
      hPullX_shrh[title.str()]->Write();
      title.str("");
      title << "pullY_" << i+1 << "-" << j+1 << "_sh-rh";
      hPullY_shrh[title.str()]->Write();
      title.str("");
      title << "pullX_" << i+1 << "-" << j+1 << "_sh-st";
      hPullX_shst[title.str()]->Write();
      title.str("");
      title << "pullY_" << i+1 << "-" << j+1 << "_sh-st";
      hPullY_shst[title.str()]->Write();
      title.str("");
      title << "pullX_" << i+1 << "-" << j+1 << "_st-rh";
      hPullX_strh[title.str()]->Write();
      title.str("");
      title << "pullY_" << i+1 << "-" << j+1 << "_st-rh";
      hPullY_strh[title.str()]->Write();
      title.str("");
      title << "PullGP_X_" << i+1 << "-" << j+1 << "_sh-st";
      hPullGP_X_shst[title.str()]->Write();
      title.str("");
      title << "PullGP_Y_" << i+1 << "-" << j+1 << "_sh-st";
      hPullGP_Y_shst[title.str()]->Write();
      title.str("");
      title << "PullGP_Z_" << i+1 << "-" << j+1 << "_sh-st";
      hPullGP_Z_shst[title.str()]->Write();
      if ( ((i==2||i==4)&&(j==0||j==1)) || (i==3||i==5) ){
	title.str("");
	title << "pullM_" << i+1 << "-" << j+1 << "_sh-rh";
	hPullM_shrh[title.str()]->Write();
	title.str("");
	title << "pullS_" << i+1 << "-" << j+1 << "_sh-rh";
	hPullS_shrh[title.str()]->Write();
	title.str("");
	title << "pullM_" << i+1 << "-" << j+1 << "_sh-st";
	hPullM_shst[title.str()]->Write();
	title.str("");
	title << "pullS_" << i+1 << "-" << j+1 << "_sh-st";
	hPullS_shst[title.str()]->Write();
	title.str("");
	title << "pullM_" << i+1 << "-" << j+1 << "_st-rh";
	hPullM_strh[title.str()]->Write();
	title.str("");
	title << "pullS_" << i+1 << "-" << j+1 << "_st-rh";
	hPullS_strh[title.str()]->Write();
      }
    }
  hPullGPXvsGPX_shst->Write();
  hPullGPXvsGPY_shst->Write();
  hPullGPXvsGPZ_shst->Write();
  hPullGPXvsGPr_shst->Write();
  hPullGPXvsGPeta_shst->Write();
  hPullGPXvsGPphi_shst->Write();
  
  //file->Write();
  file->Close();
}
