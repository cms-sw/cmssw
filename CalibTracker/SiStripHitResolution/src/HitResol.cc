////////////////////////////////////////////////////////////////////////////////
// Package:          CalibTracker/SiStripHitResolution
// Class:            HitResol
// Original Author:  DG
//                   adapted from HitEff
//
///////////////////////////////////////////////////////////////////////////////

// system include files
#include <memory>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "UserCode/SiStripHitResolution/interface/HitResol.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "UserCode/SiStripHitResolution/interface/TrajectoryAtInvalidHit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "Geometry/CommonDetUnit/interface/GluedGeomDet.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


#include "TMath.h"
#include "TH1F.h"

//ModifDG
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "TrackingTools/TrackFitters/interface/TrajectoryStateCombiner.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
//
// constructors and destructor
//

using namespace std;
HitResol::HitResol(const edm::ParameterSet& conf) :
  scalerToken_( consumes< LumiScalersCollection >(conf.getParameter<edm::InputTag>("lumiScalers")) ),
  commonModeToken_( mayConsume< edm::DetSetVector<SiStripRawDigi> >(conf.getParameter<edm::InputTag>("commonMode")) ),
  combinatorialTracks_token_( consumes< reco::TrackCollection >(conf.getParameter<edm::InputTag>("combinatorialTracks")) ),
  trajectories_token_( consumes< std::vector<Trajectory> >(conf.getParameter<edm::InputTag>("trajectories")) ),
  trajTrackAsso_token_( consumes< TrajTrackAssociationCollection >(conf.getParameter<edm::InputTag>("trajectories")) ),
//ModifDG
      tjToken_( consumes<  std::vector<Trajectory  > >( conf.getParameter<edm::InputTag>("trajectories") )),
//   tkToken_( consumes< reco::TrackCollection  >( conf.getParameter<edm::InputTag>("tracks") )),
//EndModifDG
  clusters_token_( consumes< edmNew::DetSetVector<SiStripCluster> >(conf.getParameter<edm::InputTag>("siStripClusters")) ),
  digis_token_( consumes< DetIdCollection >(conf.getParameter<edm::InputTag>("siStripDigis")) ),
//ModifDG
//   tracks_token_( consumes<edm::View<reco::Track> >( conf.getParameter<edm::InputTag>("Tracks") )),
//EndModifDG
  trackerEvent_token_( consumes< MeasurementTrackerEvent>(conf.getParameter<edm::InputTag>("trackerEvent")) ),
  conf_(conf)
{
  compSettings=conf_.getUntrackedParameter<int>("CompressionSettings",-1);
  layers =conf_.getParameter<int>("Layer");
  DEBUG = conf_.getParameter<bool>("Debug");
  addLumi_ = conf_.getUntrackedParameter<bool>("addLumi", false);
  addCommonMode_ = conf_.getUntrackedParameter<bool>("addCommonMode", false);
  cutOnTracks_ = conf_.getUntrackedParameter<bool>("cutOnTracks", false);
  trackMultiplicityCut_ = conf.getUntrackedParameter<unsigned int>("trackMultiplicity",100);
  useFirstMeas_ = conf_.getUntrackedParameter<bool>("useFirstMeas", false);
  useLastMeas_ = conf_.getUntrackedParameter<bool>("useLastMeas", false);
  useAllHitsFromTracksWithMissingHits_ = conf_.getUntrackedParameter<bool>("useAllHitsFromTracksWithMissingHits", false);
  MomentumCut_ = conf_.getUntrackedParameter<double>("MomentumCut", 3.);
  UsePairsOnly_ = conf.getUntrackedParameter<unsigned int>("UsePairsOnly",1);
}

// Virtual destructor needed.
HitResol::~HitResol() { }

void HitResol::beginJob(){

  edm::Service<TFileService> fs;
  if(compSettings>0){
    edm::LogInfo("SiStripHitResolution:HitResol")<<"the compressions settings are:"<< compSettings << std::endl;
    fs->file().SetCompressionSettings(compSettings);
  }

  reso = fs->make<TTree>("reso","tree hit pairs for resolution studies");
  reso->Branch("momentum",&mymom,"momentum/F");
  reso->Branch("numHits",&numHits,"numHits/I");
  reso->Branch("trackChi2",&ProbTrackChi2,"trackChi2/F");
  reso->Branch("detID1",&iidd1,"detID1/I");
  reso->Branch("pitch1",&mypitch1,"pitch1/F");
  reso->Branch("clusterW1",&clusterWidth,"clusterW1/I");
  reso->Branch("expectedW1",&expWidth,"expectedW1/F");
  reso->Branch("atEdge1",&atEdge,"atEdge1/F");
  reso->Branch("simpleRes",&simpleRes,"simpleRes/F");
  reso->Branch("detID2",&iidd2,"detID2/I");
  reso->Branch("clusterW2",&clusterWidth_2,"clusterW2/I");
  reso->Branch("expectedW2",&expWidth_2,"expectedW2/F");
  reso->Branch("atEdge2",&atEdge_2,"atEdge2/F");
  reso->Branch("pairPath",&pairPath,"pairPath/F");
  reso->Branch("hitDX",&hitDX,"hitDX/F");
  reso->Branch("trackDX",&trackDX,"trackDX/F");
  reso->Branch("trackDXE",&trackDXE,"trackDXE/F");
  reso->Branch("trackParamX",&trackParamX,"trackParamX/F");
  reso->Branch("trackParamY",&trackParamY,"trackParamY/F");
  reso->Branch("trackParamDXDZ",&trackParamDXDZ,"trackParamDXDZ/F");
  reso->Branch("trackParamDYDZ",&trackParamDYDZ,"trackParamDYDZ/F");
  reso->Branch("trackParamXE",&trackParamXE,"trackParamXE/F");
  reso->Branch("trackParamYE",&trackParamYE,"trackParamYE/F");
  reso->Branch("trackParamDXDZE",&trackParamDXDZE,"trackParamDXDZE/F");
  reso->Branch("trackParamDYDZE",&trackParamDYDZE,"trackParamDYDZE/F");
  reso->Branch("pairsOnly",&pairsOnly,"pairsOnly/I");
  treso = fs->make<TTree>("treso","tree tracks  for resolution studies");
  treso->Branch("track_momentum",&track_momentum,"track_momentum/F");
  treso->Branch("track_eta",&track_eta,"track_eta/F");
  treso->Branch("track_trackChi2",&track_trackChi2,"track_trackChi2/F");

  events = 0;
  EventTrackCKF = 0;
}


void HitResol::analyze(const edm::Event& e, const edm::EventSetup& es){
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  //  bool DEBUG = false;

  LogDebug("SiStripHitResolution:HitResol")  << "beginning analyze from HitResol" << endl;

  using namespace edm;
  using namespace reco;
  // Step A: Get Inputs

  int run_nr = e.id().run();
  int ev_nr = e.id().event();

  //
  edm::Handle<edm::DetSetVector<SiStripRawDigi> > commonModeDigis;
  if(addCommonMode_) e.getByToken(commonModeToken_, commonModeDigis);

  //CombinatoriaTrack
  edm::Handle<reco::TrackCollection> trackCollectionCKF;
  e.getByToken(combinatorialTracks_token_,trackCollectionCKF);

  edm::Handle<std::vector<Trajectory> > TrajectoryCollectionCKF;
  e.getByToken(trajectories_token_,TrajectoryCollectionCKF);

  edm::Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  e.getByToken(trajTrackAsso_token_, trajTrackAssociationHandle);

  // Clusters
  // get the SiStripClusters from the event
  edm::Handle< edmNew::DetSetVector<SiStripCluster> > theClusters;
  e.getByToken(clusters_token_, theClusters);

  //get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  const TrackerGeometry * tkgeom=&(* tracker);

  //get Cluster Parameter Estimator
  edm::ESHandle<StripClusterParameterEstimator> parameterestimator;
  es.get<TkStripCPERecord>().get("StripCPEfromTrackAngle", parameterestimator);
  const StripClusterParameterEstimator &stripcpe(*parameterestimator);

  // get the SiStripQuality records
  edm::ESHandle<SiStripQuality> SiStripQuality_;
  es.get<SiStripQualityRcd>().get(SiStripQuality_);

  edm::ESHandle<MagneticField> magFieldHandle;
  es.get<IdealMagneticFieldRecord>().get(magFieldHandle);
  const MagneticField* magField_ = magFieldHandle.product();

  // get the list of module IDs with FED-detected errors
  edm::Handle< DetIdCollection > fedErrorIds;
  e.getByToken(digis_token_, fedErrorIds );

  ESHandle<MeasurementTracker> measurementTrackerHandle;
  es.get<CkfComponentsRecord>().get(measurementTrackerHandle);

  edm::Handle<MeasurementTrackerEvent> measurementTrackerEvent;
  e.getByToken(trackerEvent_token_,measurementTrackerEvent);

  edm::ESHandle<Chi2MeasurementEstimatorBase> est;
  es.get<TrackingComponentsRecord>().get("Chi2",est);

//   edm::ESHandle<Propagator> prop;
//   es.get<TrackingComponentsRecord>().get("PropagatorWithMaterial",prop);
//   const Propagator* thePropagator = prop.product();

  edm::Handle<std::vector<Trajectory> > trajectoryCollectionHandle;
  e.getByToken(tjToken_,trajectoryCollectionHandle);
  const TrajectoryCollection*  trajectoryCollection = trajectoryCollectionHandle.product();

  events++;

//   // *************** SiStripCluster Collection
//   const edmNew::DetSetVector<SiStripCluster>& input = *theClusters;


// List of variables for SiStripHitResolution ntuple
   mymom            = 0;
   numHits          = 0;
   ProbTrackChi2    = 0;
   iidd1            = 0;
   mypitch1         = 0;
   clusterWidth     = 0;
   expWidth         = 0;
   atEdge           = 0;
   simpleRes        = 0;
   iidd2            = 0;
   clusterWidth_2   = 0;
   expWidth_2       = 0;
   atEdge_2         = 0;
   pairPath	    = 0;
   hitDX	    = 0;
   trackDX	    = 0;
   trackDXE	    = 0;
   trackParamX      = 0;
   trackParamY      = 0;
   trackParamDXDZ   = 0;
   trackParamDYDZ   = 0;
   trackParamXE     = 0;
   trackParamYE     = 0;
   trackParamDXDZE  = 0;
   trackParamDYDZE  = 0;
   pairsOnly        = 0;

  // Tracking
  const   reco::TrackCollection *tracksCKF=trackCollectionCKF.product();

////// Plugin of Nico code:

  std::cout<<"Starting analysis, nrun nevent, tracksCKF->size(): "<<run_nr<<" "<<ev_nr<<" "<<  tracksCKF->size() <<std::endl;

  for(unsigned int iT = 0; iT < tracksCKF->size(); ++iT){
    track_momentum = tracksCKF->at(iT).pt();
    track_eta = tracksCKF->at(iT).eta();
    track_trackChi2   = ChiSquaredProbability((double)( tracksCKF->at(iT).chi2() ),(double)( tracksCKF->at(iT).ndof() ));
    treso->Fill();
  }

  // loop over trajectories from refit
  for ( const auto& traj : *trajectoryCollection ) {
    const auto& TMeas = traj.measurements();
    // Loop on each measurement and take it into consideration
    //--------------------------------------------------------
    for ( auto itm = TMeas.cbegin(); itm != TMeas.cend(); ++itm ) {
      if (!itm->updatedState().isValid()) {
        std::cout<<"NONVALIDE"<<std::endl;
        continue;
      }

      const TransientTrackingRecHit::ConstRecHitPointer mypointhit = itm->recHit();
      const TrackingRecHit*         myhit                 = (*itm->recHit()).hit();

      ProbTrackChi2 = 0;
      numHits = 0;

////    std::cout<<"TrackChi2 =  "<< ChiSquaredProbability((double)( itm->chiSquared() ),(double)( itm->ndof(false) ))  <<std::endl;
//    std::cout<<"itm->updatedState().globalMomentum().perp(): "<<  itm->updatedState().globalMomentum().perp() <<std::endl;
//    std::cout<<"numhits "<< itraj->foundHits()  <<std::endl;

      numHits = traj.foundHits() ;
      ProbTrackChi2 = ChiSquaredProbability((double)( traj.chiSquared() ),(double)( traj.ndof(false) ));

      mymom =  itm->updatedState().globalMomentum().perp();

      std::cout<<"mymom "<<mymom<<std::endl;

//      double  MomentumCut_ = 3. ;

      //Now for the first hit
      TrajectoryStateOnSurface mytsos = itm->updatedState();
      const auto hit1 = itm->recHit();
      DetId id1 = hit1->geographicalId();
//      if(id1.subdetId() < StripSubdetector::TIB || id1.subdetId() > StripSubdetector::TEC) continue;
//      if(id1.subdetId() < StripSubdetector::TIB || id1.subdetId() > StripSubdetector::TEC) std::cout<<"AUTRE"<<std::endl;

//      if (    hit1->isValid()  && mymom > MomentumCut_) std::cout<<"mymom: "<<  itm->updatedState().globalMomentum().perp() <<std::endl;

      if (    hit1->isValid()  &&
              mymom > MomentumCut_  &&
	      ( id1.subdetId() >= StripSubdetector::TIB && id1.subdetId() <= StripSubdetector::TEC) ) {

          const auto stripdet = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit( hit1->geographicalId()));
          const StripTopology& Topo  = stripdet->specificTopology();
          int Nstrips = Topo.nstrips();
          mypitch1 = stripdet->surface().bounds().width() / Topo.nstrips();

          const auto det = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(mypointhit->geographicalId()));

          TrajectoryStateOnSurface mytsos = itm->updatedState();
          LocalVector trackDirection = mytsos.localDirection();
          LocalVector drift = stripcpe.driftDirection(stripdet);

          const auto hit1d = dynamic_cast<const SiStripRecHit1D*>(myhit);

          if (hit1d) {
//             float myres = getSimHitRes(det,trackDirection,*hit1d,expWidth,&mypitch1,drift);
             getSimHitRes(det,trackDirection,*hit1d,expWidth,&mypitch1,drift);
             clusterWidth = hit1d->cluster()->amplitudes().size();
             uint16_t firstStrip = hit1d->cluster()->firstStrip();
             uint16_t lastStrip = firstStrip + (hit1d->cluster()->amplitudes()).size() -1;
             atEdge = (firstStrip == 0 || lastStrip == (Nstrips-1) );
          }


          const auto hit2d = dynamic_cast<const SiStripRecHit2D*>(myhit);

          if (hit2d) {
//             float myres = getSimHitRes(det,trackDirection,*hit2d, expWidth,&mypitch1,drift);
             getSimHitRes(det,trackDirection,*hit2d, expWidth,&mypitch1,drift);
             clusterWidth = hit2d->cluster()->amplitudes().size();
             uint16_t firstStrip = hit2d->cluster()->firstStrip();
             uint16_t lastStrip = firstStrip + (hit2d->cluster()->amplitudes()).size() -1;
             atEdge = (firstStrip == 0 || lastStrip == (Nstrips-1) );
          }

          simpleRes = getSimpleRes(&(*itm)); // simple resolution by using the track re-fit forward and backward predicted state

          // Now to see if there is a match - pair method - hit in overlapping sensors
          vector < TrajectoryMeasurement >::const_iterator itTraj2 =  TMeas.end(); // last hit along the fitted track

      for ( auto itmCompare = itm-1;
            // start to compare from the 5th hit
            itmCompare >= TMeas.cbegin() && itmCompare > itm-4;
            --itmCompare ) {
        const auto hit2 = itmCompare->recHit();
        if (!hit2->isValid()) continue;
        DetId id2 = hit2->geographicalId();

        //must be from the same detector and layer
	iidd1 = hit1->geographicalId().rawId();
	iidd2 = hit2->geographicalId().rawId();
        if ( id1.subdetId() != id2.subdetId() || checkLayer(iidd1, tTopo) !=  checkLayer(iidd2, tTopo)  ) break;
        //must both be stereo if one is
        if(tTopo->isStereo(id1) != tTopo->isStereo(id2) ) continue;
        //A check i dont completely understand but might as well keep there
        if (tTopo->glued(id1) == id1.rawId() ) cout << "BAD GLUED: Have glued layer with id = " << id1.rawId() << " and glued id = " << tTopo->glued(id1) << "  and stereo = " << tTopo->isStereo(id1) << endl;
        if (tTopo->glued(id2) == id2.rawId() ) cout << "BAD GLUED: Have glued layer with id = " << id2.rawId() << " and glued id = " << tTopo->glued(id2) << "  and stereo = " << tTopo->isStereo(id2) << endl;

        itTraj2 = itmCompare;
        break;
      }

      if ( itTraj2 == TMeas.cend() ) {
      } else {
//             std::cout<<"Found overlapping sensors "<<std::endl;

//          pairsOnly = 1;
          pairsOnly = UsePairsOnly_;

          //We found one....let's fill in the truth info!
          TrajectoryStateOnSurface tsos_2 = itTraj2->updatedState();
          LocalVector trackDirection_2 = tsos_2.localDirection();
          const auto myhit2 = itTraj2->recHit();
          const auto myhit_2 = (*itTraj2->recHit()).hit();
          const auto stripdet_2 = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(myhit2->geographicalId()));
          const StripTopology& Topo_2  = stripdet_2->specificTopology();
          int Nstrips_2 = Topo_2.nstrips();
          float mypitch_2 = stripdet_2->surface().bounds().width() / Topo_2.nstrips();

          if ( mypitch1 != mypitch_2 ) return;  // for PairsOnly


          const auto det_2 = dynamic_cast<const StripGeomDetUnit*>(tkgeom->idToDetUnit(myhit2->geographicalId()));

          LocalVector drift_2 = stripcpe.driftDirection(stripdet_2);

          const auto hit1d_2 = dynamic_cast<const SiStripRecHit1D*>(myhit_2);
          if (hit1d_2) {
             getSimHitRes(det_2,trackDirection_2,*hit1d_2,expWidth_2,&mypitch_2,drift_2);
             clusterWidth_2 = hit1d_2->cluster()->amplitudes().size();
             uint16_t firstStrip_2 = hit1d_2->cluster()->firstStrip();
             uint16_t lastStrip_2 = firstStrip_2 + (hit1d_2->cluster()->amplitudes()).size() -1;
             atEdge_2 = (firstStrip_2 == 0 || lastStrip_2 == (Nstrips_2-1) );
          }

          const auto hit2d_2 = dynamic_cast<const SiStripRecHit2D*>(myhit_2);
          if (hit2d_2) {
//             float myres_2 = getSimHitRes(det_2,trackDirection_2,*hit2d_2, expWidth_2,&mypitch_2,drift_2);
             getSimHitRes(det_2,trackDirection_2,*hit2d_2, expWidth_2,&mypitch_2,drift_2);
             clusterWidth_2 = hit2d_2->cluster()->amplitudes().size();
             uint16_t firstStrip_2 = hit2d_2->cluster()->firstStrip();
             uint16_t lastStrip_2 = firstStrip_2 + (hit2d_2->cluster()->amplitudes()).size() -1;
             atEdge_2 = (firstStrip_2 == 0 || lastStrip_2 == (Nstrips_2-1) );
          }


//        if(pairsOnly && (pitch != pitch2) ) fill = false;


	   // Make AnalyticalPropagator to use in getPairParameters
	   AnalyticalPropagator mypropagator(magField_,anyDirection);

           if(!getPairParameters(&(*magField_),mypropagator,&(*itTraj2),&(*itm),pairPath,hitDX,trackDX,trackDXE,trackParamX,trackParamY,trackParamDXDZ,trackParamDYDZ,trackParamXE,trackParamYE,trackParamDXDZE,trackParamDYDZE) ){


           }else {

//   std::cout<<"  "<<std::endl;
//   std::cout<<"  "<<std::endl;
//   std::cout<<"  "<<std::endl;
//
// //   std::cout<<" momentum "<< track_momentum <<std::endl;
// //   std::cout<<" track_trackChi2      "<<   track_trackChi2<<std::endl;
// //   std::cout<<" track_trackChi2_2   "<<    track_trackChi2_2<<std::endl;
// //   std::cout<<" track_eta      "<<         track_eta<<std::endl;
//   std::cout<<" momentum       "<<         mymom<<std::endl;
//   std::cout<<" numHits         "<<        numHits<<std::endl;
//   std::cout<<" trackChi2       "<<        ProbTrackChi2<<std::endl;
//   std::cout<<" detID1         "<<         iidd1<<std::endl;
//   std::cout<<" pitch1          "<<        mypitch1<<std::endl;
//   std::cout<<" clusterW1        "<<       clusterWidth<<std::endl;
//   std::cout<<" expectedW1       "<<       expWidth<<std::endl;
//   std::cout<<" atEdge1          "<<       atEdge<<std::endl;
//   std::cout<<" simpleRes        "<<       simpleRes<<std::endl;
//   std::cout<<" detID2           "<<       iidd2<<std::endl;
//   std::cout<<" clusterW2        "<<       clusterWidth_2<<std::endl;
//   std::cout<<" expectedW2       "<<       expWidth_2<<std::endl;
//   std::cout<<" atEdge2          "<<       atEdge_2<<std::endl;
//
//   std::cout<<" pairPath         "<<       pairPath<<std::endl;
//   std::cout<<" hitDX            "<<       hitDX<<std::endl;
//   std::cout<<" trackDX          "<<       trackDX<<std::endl;
//   std::cout<<" trackDXE         "<<       trackDXE<<std::endl;
//
//   std::cout<<" trackParamX	  "<<        trackParamX<<std::endl;
//   std::cout<<" trackParamY	  "<<        trackParamY <<std::endl;
//   std::cout<<" trackParamDXDZ     "<<     trackParamDXDZ<<std::endl;
//   std::cout<<" trackParamDYDZ     "<<     trackParamDYDZ<<std::endl;
//   std::cout<<" trackParamXE        "<<    trackParamXE<<std::endl;
//   std::cout<<" trackParamYE        "<<    trackParamYE<<std::endl;
//   std::cout<<" trackParamDXDZE     "<<    trackParamDXDZE<<std::endl;
//   std::cout<<" trackParamDYDZE     "<<    trackParamDYDZE<<std::endl;
//
   reso->Fill();
	   }
       }//itTraj2 != TMeas.end()


    }//hit1->isValid()....

      } // itm
    } // it









////// Endof Plugin of Nico code:


}






void HitResol::endJob(){
//   traj->GetDirectory()->cd();
//   traj->Write();

  LogDebug("SiStripHitResolution:HitResol") << " Events Analysed             " <<  events          << endl;
  LogDebug("SiStripHitResolution:HitResol") << " Number Of Tracked events    " <<  EventTrackCKF   << endl;

  reso->GetDirectory()->cd();
  reso->Write();
  treso->Write();
}

double HitResol::checkConsistency(const StripClusterParameterEstimator::LocalValues& parameters, double xx, double xerr) {
  double error = sqrt(parameters.second.xx() + xerr*xerr);
  double separation = abs(parameters.first.x() - xx);
  double consistency = separation/error;
  return consistency;
}

bool HitResol::isDoubleSided(unsigned int iidd, const TrackerTopology* tTopo) const {
  StripSubdetector strip=StripSubdetector(iidd);
  unsigned int subid=strip.subdetId();
  unsigned int layer = 0;
  if (subid ==  StripSubdetector::TIB) {
    layer = tTopo->tibLayer(iidd);
    if (layer == 1 || layer == 2) return true;
    else return false;
  }
  else if (subid ==  StripSubdetector::TOB) {
    layer = tTopo->tobLayer(iidd) + 4 ;
    if (layer == 5 || layer == 6) return true;
    else return false;
  }
  else if (subid ==  StripSubdetector::TID) {
    layer = tTopo->tidRing(iidd) + 10;
    if (layer == 11 || layer == 12) return true;
    else return false;
  }
  else if (subid ==  StripSubdetector::TEC) {
    layer = tTopo->tecRing(iidd) + 13 ;
    if (layer == 14 || layer == 15 || layer == 18) return true;
    else return false;
  }
  else
    return false;
}

//float HitResol::getSimHitRes(const GeomDetUnit * det, const LocalVector& trackdirection, const TrackingRecHit& recHit, float& trackWidth, float* pitch, LocalVector& drift){
void HitResol::getSimHitRes(const GeomDetUnit * det, const LocalVector& trackdirection, const TrackingRecHit& recHit, float& trackWidth, float* pitch, LocalVector& drift){
  const auto stripdet = dynamic_cast<const StripGeomDetUnit*>(det);
  const auto& topol = dynamic_cast<const StripTopology&>(stripdet->topology());

  LocalPoint position = recHit.localPosition();
  (*pitch) = topol.localPitch(position);

//  float rechitrphiresMF = -1;

  float anglealpha = 0;
  if (trackdirection.z() != 0) {
    anglealpha = atan(trackdirection.x() / trackdirection.z()) * TMath::RadToDeg();
  }


//  LocalVector drift = stripcpe.driftDirection(stripdet);
  float thickness = stripdet->surface().bounds().thickness();
  float tanalpha = tan(anglealpha * TMath::DegToRad());
  float tanalphaL = drift.x() / drift.z();
  (trackWidth) = fabs((thickness / (*pitch)) * tanalpha - (thickness / (*pitch)) * tanalphaL);


//  return rechitrphiresMF;

}

double HitResol::getSimpleRes(const TrajectoryMeasurement* traj1){
  TrajectoryStateOnSurface theCombinedPredictedState;

  if ( traj1->backwardPredictedState().isValid() )
    theCombinedPredictedState = TrajectoryStateCombiner().combine( traj1->forwardPredictedState(),
								   traj1->backwardPredictedState());
  else
    theCombinedPredictedState = traj1->forwardPredictedState();

  if (!theCombinedPredictedState.isValid()) {
    return -100;
  }

  const TransientTrackingRecHit::ConstRecHitPointer firstRecHit = traj1->recHit();
  double recHitX_1 = firstRecHit->localPosition().x();
  return (theCombinedPredictedState.localPosition().x() - recHitX_1);
}


//traj1 is the matched trajectory...traj2 is the original
bool HitResol::getPairParameters(const MagneticField* magField_, AnalyticalPropagator& propagator, const TrajectoryMeasurement* traj1, const TrajectoryMeasurement* traj2, float & pairPath,float & hitDX, float & trackDX, float & trackDXE, float & trackParamX, float &trackParamY , float & trackParamDXDZ, float &trackParamDYDZ , float & trackParamXE, float &trackParamYE, float & trackParamDXDZE, float &trackParamDYDZE){
  pairPath  = 0;
  hitDX     = 0;
  trackDX   = 0;
  trackDXE  = 0;

  trackParamX	  = 0;
  trackParamY	  = 0;
  trackParamDXDZ  = 0;
  trackParamDYDZ  = 0;
  trackParamXE    = 0;
  trackParamYE    = 0;
  trackParamDXDZE = 0;
  trackParamDYDZE = 0;

  TrajectoryStateCombiner combiner_;

  // backward predicted state at module 1
  TrajectoryStateOnSurface bwdPred1 = traj1->backwardPredictedState();
  if ( !bwdPred1.isValid() )  return false;
  //cout << "momentum from backward predicted state = " << bwdPred1.globalMomentum().mag() << endl;
  // forward predicted state at module 2
  TrajectoryStateOnSurface fwdPred2 = traj2->forwardPredictedState();
  //cout << "momentum from forward predicted state = " << fwdPred2.globalMomentum().mag() << endl;
  if ( !fwdPred2.isValid() )  return false;
  // extrapolate fwdPred2 to module 1
  TrajectoryStateOnSurface fwdPred2At1 = propagator.propagate(fwdPred2,bwdPred1.surface());
  if ( !fwdPred2At1.isValid() )  return false;
  // combine fwdPred2At1 with bwdPred1 (ref. state, best estimate without hits 1 and 2)
  TrajectoryStateOnSurface comb1 = combiner_.combine(bwdPred1,fwdPred2At1);
  if ( !comb1.isValid() )  return false;

  //
  // propagation of reference parameters to module 2
  //
  std::pair<TrajectoryStateOnSurface,double> tsosWithS =
    propagator.propagateWithPath(comb1,fwdPred2.surface());
  TrajectoryStateOnSurface comb1At2 = tsosWithS.first;
  if ( !comb1At2.isValid() )  return false;
  //distance of propagation from one surface to the next==could cut here
  pairPath = tsosWithS.second;
  if (TMath::Abs(pairPath) > 15 ) return false; //cut to remove hit pairs > 15 cm apart

  // local parameters and errors on module 1
  AlgebraicVector5 pars = comb1.localParameters().vector();
  AlgebraicSymMatrix55 errs = comb1.localError().matrix();
  //number 3 is predX
  double predX1 = pars[3];
  //track fitted parameters in local coordinates for position 0
  (trackParamX    ) = pars[3];
  (trackParamY    ) = pars[4];
  (trackParamDXDZ ) = pars[1];
  (trackParamDYDZ ) = pars[2];
  (trackParamXE   ) = TMath::Sqrt(errs(3,3));
  (trackParamYE   ) = TMath::Sqrt(errs(4,4));
  (trackParamDXDZE) = TMath::Sqrt(errs(1,1));
  (trackParamDYDZE) = TMath::Sqrt(errs(2,2));

  // local parameters and errors on module 2
  pars = comb1At2.localParameters().vector();
  errs = comb1At2.localError().matrix();
  double predX2 = pars[3];

  ////
  //// jacobians (local-to-global@1,global 1-2,global-to-local@2)
  ////
  JacobianLocalToCurvilinear jacLocToCurv(comb1.surface(),
      comb1.localParameters(),
      *magField_);
  AnalyticalCurvilinearJacobian jacCurvToCurv(comb1.globalParameters(),
      comb1At2.globalPosition(),
      comb1At2.globalMomentum(),
      tsosWithS.second);
  JacobianCurvilinearToLocal jacCurvToLoc(comb1At2.surface(),
      comb1At2.localParameters(),
      *magField_);
  // combined jacobian local-1-to-local-2
  AlgebraicMatrix55 jacobian =
      jacLocToCurv.jacobian()*jacCurvToCurv.jacobian()*jacCurvToLoc.jacobian();
  // covariance on module 1
  AlgebraicSymMatrix55 covComb1 = comb1.localError().matrix();
  // variance and correlations for predicted local_x on modules 1 and 2
  double c00 = covComb1(3,3);
  double c10(0.);
  double c11(0.);
  for ( int i=1; i<5; ++i ) {
    c10 += jacobian(3,i)*covComb1(i,3);
    for ( int j=1; j<5; ++j )  c11 += jacobian(3,i)*covComb1(i,j)*jacobian(3,j);
  }
  // choose relative sign in order to minimize error on difference
  double diff = c00 - 2*fabs(c10) + c11;
  diff = diff>0 ? sqrt(diff) : -sqrt(-diff);
  (trackDXE) = diff;
  double relativeXSign_ = c10>0 ? -1 : 1;

  (trackDX) = predX1 + relativeXSign_*predX2;

  double recHitX_1 = traj1->recHit()->localPosition().x();
  double recHitX_2 = traj2->recHit()->localPosition().x();

  (hitDX) = recHitX_1 + relativeXSign_*recHitX_2;

  return true;
}

bool HitResol::check2DPartner(unsigned int iidd, const std::vector<TrajectoryMeasurement>& traj) {
  unsigned int partner_iidd = 0;
  bool found2DPartner = false;
  // first get the id of the other detector
  if ((iidd & 0x3)==1) partner_iidd = iidd+1;
  if ((iidd & 0x3)==2) partner_iidd = iidd-1;
  // next look in the trajectory measurements for a measurement from that detector
  // loop through trajectory measurements to find the partner_iidd
  for ( const auto& tm : traj ) {
    if (tm.recHit()->geographicalId().rawId()==partner_iidd) {
      found2DPartner = true;
    }
  }
  return found2DPartner;
}

unsigned int HitResol::checkLayer( unsigned int iidd, const TrackerTopology* tTopo) {
  StripSubdetector strip=StripSubdetector(iidd);
  unsigned int subid=strip.subdetId();
  if (subid ==  StripSubdetector::TIB) {
    return tTopo->tibLayer(iidd);
  }
  if (subid ==  StripSubdetector::TOB) {
    return tTopo->tobLayer(iidd) + 4 ;
  }
  if (subid ==  StripSubdetector::TID) {
    return tTopo->tidWheel(iidd) + 10;
  }
  if (subid ==  StripSubdetector::TEC) {
    return tTopo->tecWheel(iidd) + 13 ;
  }
  return 0;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HitResol);
