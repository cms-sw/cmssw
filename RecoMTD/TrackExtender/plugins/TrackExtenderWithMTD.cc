/** \file
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"

#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetTray.h"
#include "RecoMTD/DetLayers/interface/MTDRingForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetRing.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/ForwardDetId/interface/MTDChannelIdentifier.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "RecoMTD/TransientTrackingRecHit/interface/MTDTransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"

#include "RecoTracker/TransientTrackingRecHit/interface/Traj2TrackHits.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

#include <sstream>

#include "Geometry/CommonTopologies/interface/Topology.h"

using namespace std;
using namespace edm;

constexpr char pathLengthName[] = "pathLength";
constexpr char pathLengthOrigTrkName[] = "generalTrackPathLength";
constexpr char betaOrigTrkName[] = "generalTrackBeta";
constexpr char t0OrigTrkName[] = "generalTrackt0";
constexpr char covt0t0OrigTrkName[] = "generalTrackcovt0t0";


template<class TrackCollection>
class TrackExtenderWithMTDT : public edm::stream::EDProducer<> {  
 public:
  typedef typename TrackCollection:: value_type TrackType;
  typedef edm::View<TrackType> InputCollection;
  
  TrackExtenderWithMTDT(const ParameterSet& pset); 

  void produce(edm::Event& ev, const edm::EventSetup& es) final;

  TransientTrackingRecHit::ConstRecHitContainer tryBTLLayers(const TrackType&, 
							     const MTDTrackingDetSetVector&,
							     const MTDDetLayerGeometry*, 
							     const MagneticField* field,
							     const Propagator* prop) const;

  TransientTrackingRecHit::ConstRecHitContainer tryETLLayers(const TrackType&,
							     const MTDTrackingDetSetVector&,
							     const MTDDetLayerGeometry*, 
							     const MagneticField* field,
							     const Propagator* prop) const;
  
  RefitDirection::GeometricalDirection
  checkRecHitsOrdering(TransientTrackingRecHit::ConstRecHitContainer const & recHits) const {
    
    if (!recHits.empty()){
      GlobalPoint first = gtg->idToDet(recHits.front()->geographicalId())->position();
      GlobalPoint last = gtg->idToDet(recHits.back()->geographicalId())->position();
      
      // maybe perp2?
      auto rFirst = first.mag2();
      auto rLast  = last.mag2();
      if(rFirst < rLast) return RefitDirection::insideOut;
      if(rFirst > rLast) return RefitDirection::outsideIn;
    }
    LogDebug("Reco|TrackingTools|TrackTransformer") << "Impossible to determine the rechits order" <<endl;
    return RefitDirection::undetermined;
  }

  reco::Track buildTrack(const reco::Track&, const Trajectory&, const Trajectory &, const reco::BeamSpot&, const MagneticField* field,const Propagator* prop, bool hasMTD, float& pathLength) const;
  reco::TrackExtra buildTrackExtra(const Trajectory& trajectory) const;

  string dumpLayer(const DetLayer* layer) const;

 private:
  edm::EDGetTokenT<InputCollection> tracksToken_;
  edm::EDGetTokenT<MTDTrackingDetSetVector> hitsToken_;
  edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const bool updateTraj_, updateExtra_, updatePattern_;
  const std::string mtdRecHitBuilder_,propagator_, transientTrackBuilder_;
  std::unique_ptr<MeasurementEstimator> theEstimator;
  std::unique_ptr<TrackTransformer> theTransformer;
  edm::ESHandle<TransientTrackBuilder> builder;
  edm::ESHandle<TransientTrackingRecHitBuilder> hitbuilder;
  edm::ESHandle<GlobalTrackingGeometry> gtg;
  edm::ESHandle<Propagator> prop;
};


template<class TrackCollection>  
TrackExtenderWithMTDT<TrackCollection>::TrackExtenderWithMTDT(const ParameterSet& iConfig) :
  updateTraj_(iConfig.getParameter<bool>("updateTrackTrajectory")),
  updateExtra_(iConfig.getParameter<bool>("updateTrackExtra")),
  updatePattern_(iConfig.getParameter<bool>("updateTrackHitPattern")),
  mtdRecHitBuilder_(iConfig.getParameter<std::string>("MTDRecHitBuilder")),
  propagator_(iConfig.getParameter<std::string>("Propagator")),
  transientTrackBuilder_(iConfig.getParameter<std::string>("TransientTrackBuilder")) {
  float theMaxChi2=25.;
  float theNSigma=3.;
  theEstimator = std::make_unique<Chi2MeasurementEstimator>(theMaxChi2,theNSigma);
  
  theTransformer = std::make_unique<TrackTransformer>(iConfig.getParameterSet("TrackTransformer"));

  tracksToken_ = consumes<InputCollection>(iConfig.getParameter<edm::InputTag>("tracksSrc"));
  hitsToken_ = consumes<MTDTrackingDetSetVector>(iConfig.getParameter<edm::InputTag>("hitsSrc"));
  bsToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotSrc"));

  produces<edm::ValueMap<float> >(pathLengthOrigTrkName);
  produces<edm::ValueMap<float> >(betaOrigTrkName);
  produces<edm::ValueMap<float> >(t0OrigTrkName);
  produces<edm::ValueMap<float> >(covt0t0OrigTrkName);
  produces<edm::ValueMap<float> >(pathLengthName);
  produces<edm::OwnVector<TrackingRecHit>>();
  produces<reco::TrackExtraCollection>();
  produces<TrackCollection>();
}

template<class TrackCollection>
void TrackExtenderWithMTDT<TrackCollection>::produce( edm::Event& ev,
						      const edm::EventSetup& es ) {  
  //this produces pieces of the track extra
  Traj2TrackHits t2t;

  theTransformer->setServices(es);

  TrackingRecHitRefProd hitsRefProd = ev.getRefBeforePut<TrackingRecHitCollection>();
  reco::TrackExtraRefProd extrasRefProd = ev.getRefBeforePut<reco::TrackExtraCollection>();
  
  es.get<GlobalTrackingGeometryRecord>().get(gtg);

  edm::ESHandle<MTDDetLayerGeometry> geo;
  es.get<MTDRecoGeometryRecord>().get(geo);

  edm::ESHandle<MagneticField> magfield;
  es.get<IdealMagneticFieldRecord>().get(magfield);  
    
  es.get<TransientTrackRecord>().get(transientTrackBuilder_, builder);
  es.get<TransientRecHitRecord>().get(mtdRecHitBuilder_,hitbuilder);

  edm::ESHandle<Propagator> prop;
  es.get<TrackingComponentsRecord>().get(propagator_,prop);

  edm::ESHandle<TrackerTopology> httopo;
  es.get<TrackerTopologyRcd>().get(httopo);
  const TrackerTopology& ttopo = *httopo;
    
  auto output  = std::make_unique<TrackCollection>();
  auto extras  = std::make_unique<reco::TrackExtraCollection>();
  auto outhits = std::make_unique<edm::OwnVector<TrackingRecHit>>();

  auto pathLengths = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> pathLengthsRaw;

  auto pathLengthsOrigTrk = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> pathLengthsOrigTrkRaw;

  auto betaOrigTrk = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> betaOrigTrkRaw;

  auto t0OrigTrk = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> t0OrigTrkRaw;
  
  auto covt0t0OrigTrk = std::make_unique<edm::ValueMap<float>>();
  std::vector<float> covt0t0OrigTrkRaw;
  
  edm::Handle<InputCollection> tracksH;  
  ev.getByToken(tracksToken_,tracksH);
  const auto& tracks = *tracksH;

  edm::Handle<MTDTrackingDetSetVector> hitsH;  
  ev.getByToken(hitsToken_,hitsH);
  const auto& hits = *hitsH;

  edm::Handle<reco::BeamSpot> bsH;  
  ev.getByToken(bsToken_,bsH);
  const auto& bs = *bsH;

  std::vector<unsigned> track_indices;
  unsigned itrack = 0;
  for( const auto& track : tracks ) {  
    reco::TransientTrack ttrack(track,magfield.product(),gtg);
    auto trajs = theTransformer->transform(track);
    auto thits = theTransformer->getTransientRecHits(ttrack);

    TransientTrackingRecHit::ConstRecHitContainer mtdthits;
    for( auto& ahit : tryBTLLayers(track,hits,geo.product(),magfield.product(),prop.product()) ) {
      mtdthits.push_back(ahit);
    }
    // in the future this should include an intermediate refit before propagating to the ETL
    // for now it is ok
    for( auto& ahit : tryETLLayers(track,hits,geo.product(),magfield.product(),prop.product()) ) {
      mtdthits.push_back(ahit);
    }
    
    auto ordering = checkRecHitsOrdering(thits);
    if( ordering == RefitDirection::insideOut) {
      for( auto& ahit : mtdthits ) thits.push_back(ahit);    
    } else {
      std::reverse(mtdthits.begin(),mtdthits.end());
      for( auto& ahit : thits ) mtdthits.push_back(ahit);
      thits.swap(mtdthits);
    }
    auto trajwithmtd = theTransformer->transform(ttrack,thits);
    float pathLengthMap = -1.f, betaMap = 0.f, t0Map = 0.f, covt0t0Map = -1.f;
    
    for( const auto& trj : trajwithmtd ) {
      
      const auto& thetrj = (updateTraj_ ? trj : trajs.front());
      float pathLength = 0.f;
      reco::Track result = buildTrack(track, thetrj, trj, bs, magfield.product(), 
				      prop.product(), !mtdthits.empty(),pathLength);
      if( result.ndof() >= 0 ) {
        /// setup the track extras
        reco::TrackExtra::TrajParams trajParams;
        reco::TrackExtra::Chi2sFive chi2s; 
        size_t hitsstart = outhits->size();
        if( updatePattern_ ) {
          t2t(trj,*outhits,trajParams,chi2s); // this fills the output hit collection
        } else {
          t2t(thetrj,*outhits,trajParams,chi2s);
        }
        size_t hitsend = outhits->size();
        extras->push_back(buildTrackExtra(trj)); // always push back the fully built extra, update by setting in track
        extras->back().setHits(hitsRefProd,hitsstart,hitsend-hitsstart);	
        extras->back().setTrajParams(trajParams,chi2s);        
        //create the track
        output->push_back(result);
	pathLengthsRaw.push_back(pathLength);
	pathLengthMap = pathLength;
        auto& backtrack = output->back();
	betaMap = backtrack.beta();
	t0Map = backtrack.t0();
	covt0t0Map = backtrack.covt0t0();
        reco::TrackExtraRef extraRef(extrasRefProd,extras->size()-1);
        backtrack.setExtra( (updateExtra_ ? extraRef : track.extra()) );
        for(unsigned ihit = hitsstart; ihit < hitsend; ++ihit) {
          backtrack.appendHitPattern((*outhits)[ihit],ttopo);
        }
      }
      pathLengthsOrigTrkRaw.push_back(pathLengthMap);
      betaOrigTrkRaw.push_back(betaMap);
      t0OrigTrkRaw.push_back(t0Map);
      covt0t0OrigTrkRaw.push_back(covt0t0Map);
    }
    ++itrack;
  }

  auto outTrksHandle = ev.put(std::move(output));
  ev.put(std::move(extras));
  ev.put(std::move(outhits));

  edm::ValueMap<float>::Filler fillerPathLengths(*pathLengths);
  fillerPathLengths.insert(outTrksHandle,pathLengthsRaw.cbegin(),pathLengthsRaw.cend());
  fillerPathLengths.fill();
  ev.put(std::move(pathLengths),pathLengthName);

  edm::ValueMap<float>::Filler fillerPathLengthsOrigTrk(*pathLengthsOrigTrk);
  fillerPathLengthsOrigTrk.insert(tracksH,pathLengthsOrigTrkRaw.cbegin(),pathLengthsOrigTrkRaw.cend());
  fillerPathLengthsOrigTrk.fill();
  ev.put(std::move(pathLengthsOrigTrk),pathLengthOrigTrkName);

  edm::ValueMap<float>::Filler fillerBetas(*betaOrigTrk);
  fillerBetas.insert(tracksH,betaOrigTrkRaw.cbegin(),betaOrigTrkRaw.cend());
  fillerBetas.fill();
  ev.put(std::move(betaOrigTrk),betaOrigTrkName);

  edm::ValueMap<float>::Filler fillert0s(*t0OrigTrk);
  fillert0s.insert(tracksH,t0OrigTrkRaw.cbegin(),t0OrigTrkRaw.cend());
  fillert0s.fill();
  ev.put(std::move(t0OrigTrk),t0OrigTrkName);

  edm::ValueMap<float>::Filler fillercovt0t0s(*covt0t0OrigTrk);
  fillercovt0t0s.insert(tracksH,covt0t0OrigTrkRaw.cbegin(),covt0t0OrigTrkRaw.cend());
  fillercovt0t0s.fill();
  ev.put(std::move(covt0t0OrigTrk),covt0t0OrigTrkName);
}

namespace {
  auto cmp = [](const unsigned one, const unsigned two) -> bool { return one < two; };
}


void find_hits_in_dets(const MTDTrackingDetSetVector& hits, const DetLayer* layer,
		       const TrajectoryStateOnSurface& tsos, const Propagator* prop,
		       const MeasurementEstimator& theEstimator,		       
		       const TransientTrackingRecHitBuilder& hitbuilder,
		       TransientTrackingRecHit::ConstRecHitContainer& output) {
  pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos,*prop,theEstimator);
  if( comp.first ) {    
    vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos,*prop,theEstimator);
    if (!compDets.empty()) {
      for( const auto& detWithState : compDets ) {	
	auto range = hits.equal_range(detWithState.first->geographicalId(),cmp);	  
	for( auto detitr = range.first; detitr != range.second; ++detitr ) {
	  auto best = detitr->end();
	  double best_chi2 = std::numeric_limits<double>::max();
	  for( auto itr = detitr->begin(); itr != detitr->end(); ++itr ) {
	    auto est =  theEstimator.estimate(detWithState.second,*itr);
	    if( est.first && est.second < best_chi2 ) { // just take the best chi2
	      best = itr;
	      best_chi2 = est.second;
	    }
	  }
	  if( best != detitr->end() ) {
	    output.push_back(hitbuilder.build(&*best));
	  }
	}	  	  
      }      
    }
  }
}

template<class TrackCollection>
TransientTrackingRecHit::ConstRecHitContainer
TrackExtenderWithMTDT<TrackCollection>::tryBTLLayers(const TrackType& track,
						     const MTDTrackingDetSetVector& hits,
						     const MTDDetLayerGeometry* geo,
						     const MagneticField* field,
						     const Propagator* prop) const {
  TransientTrackingRecHit::ConstRecHitContainer output;
  const vector<const DetLayer*>& layers = geo->allBTLLayers();
  auto tTrack = builder->build(track);

  for (const DetLayer* ilay : layers) {
    // get the outermost trajectory point on the track    
    TrajectoryStateOnSurface tsos = tTrack.outermostMeasurementState();
    find_hits_in_dets(hits,ilay,tsos,prop,*theEstimator,*hitbuilder,output);
  }
  return output;
}

template<class TrackCollection>
TransientTrackingRecHit::ConstRecHitContainer
TrackExtenderWithMTDT<TrackCollection>::tryETLLayers(const TrackType& track, 
						     const MTDTrackingDetSetVector& hits,
						     const MTDDetLayerGeometry* geo,
						     const MagneticField* field,
						     const Propagator* prop) const {
  TransientTrackingRecHit::ConstRecHitContainer output;
  const vector<const DetLayer*>& layers = geo->allETLLayers();

  auto tTrack = builder->build(track);
  
  for (const DetLayer* ilay : layers) {
    const BoundDisk& disk = static_cast<const MTDRingForwardDoubleLayer*>(ilay)->specificSurface();
    const double diskZ = disk.position().z();

    // get the outermost trajectory point on the track    
    TrajectoryStateOnSurface tsos = tTrack.outermostMeasurementState();
    if( tsos.globalPosition().z() * diskZ < 0 ) continue; // only propagate to the disk that's on the same side

    find_hits_in_dets(hits,ilay,tsos,prop,*theEstimator,*hitbuilder,output);    
  }
  return output;
}


//below is unfortunately ripped from other places but 
//since track producer doesn't know about MTD we have to do this
template<class TrackCollection>
reco::Track TrackExtenderWithMTDT<TrackCollection>::buildTrack(const reco::Track& orig,
							       const Trajectory& traj,
							       const Trajectory& trajWithMtd,
							       const reco::BeamSpot& bs,
							       const MagneticField* field,
							       const Propagator* thePropagator,
							       bool hasMTD,
							       float& pathLength) const {  
  // get the state closest to the beamline
  TrajectoryStateOnSurface stateForProjectionToBeamLineOnSurface = 
    traj.closestMeasurement(GlobalPoint(bs.x0(),bs.y0(),bs.z0())).updatedState();
  
  if UNLIKELY(!stateForProjectionToBeamLineOnSurface.isValid()) {
    edm::LogError("CannotPropagateToBeamLine")<<"the state on the closest measurement isnot valid. skipping track.";
    return reco::Track();
  }

  constexpr double mpi = 0.13957018;
  constexpr double c = 2.99792458e1; //[cm/ns]
  
  const FreeTrajectoryState & stateForProjectionToBeamLine=*stateForProjectionToBeamLineOnSurface.freeState();
  
  TSCBLBuilderWithPropagator tscblBuilder(*thePropagator);
  TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(stateForProjectionToBeamLine,bs);
  
  if UNLIKELY(!tscbl.isValid()) {
    return reco::Track();
  }

  GlobalPoint v = tscbl.trackStateAtPCA().position();
  math::XYZPoint  pos( v.x(), v.y(), v.z() );
  GlobalVector p = tscbl.trackStateAtPCA().momentum();
  math::XYZVector mom( p.x(), p.y(), p.z() );

  int ndof = traj.ndof();
  
  double t0 = 0.;
  double covt0t0 = -1.;
  pathLength = -1.f; // if there is no MTD flag the pathlength with -1
  double betaOut = 0.;

  //compute path length for time backpropagation, using first MTD hit for the momentum
  if (hasMTD) {
    
    bool validpropagation = true;
    double pathlength = 0.;
    double pathlength1 = 0.;
    double pathlength2 = 0.;
    for (auto it=trajWithMtd.measurements().begin(); it!=trajWithMtd.measurements().end()-1; ++it) {
       const auto &propresult = thePropagator->propagateWithPath(it->updatedState(), (it+1)->updatedState().surface());
       double layerpathlength = std::abs(propresult.second);
       if (layerpathlength==0.) {
         validpropagation = false;
       }
       pathlength1 += layerpathlength;
    }
     
    double thit = 0.;
    double thiterror = -1.;
    bool validmtd = false;
    if (trajWithMtd.direction() == alongMomentum) {
      for (auto it=trajWithMtd.measurements().begin(); it!=trajWithMtd.measurements().end(); ++it) {
        bool ismtd = it->recHit()->geographicalId().det() == DetId::Forward && ForwardSubdetector(it->recHit()->geographicalId().subdetId()) == FastTime;
        if (ismtd) {
          const auto &propresult2 = thePropagator->propagateWithPath(tscbl.trackStateAtPCA(), trajWithMtd.firstMeasurement().updatedState().surface());
	  pathlength2 = propresult2.second;
          if (pathlength2 == 0.) {
            validpropagation = false;
          }
          pathlength = pathlength1 + pathlength2; 
          const MTDTrackingRecHit *mtdhit = static_cast<const MTDTrackingRecHit*>(it->recHit()->hit());
          thit = mtdhit->time();
          thiterror = mtdhit->timeError();
          validmtd = true;
          break;          
        }
      }
    }
    else {
      for (auto it=trajWithMtd.measurements().rbegin(); it!=trajWithMtd.measurements().rend(); ++it) {
        bool ismtd = it->recHit()->geographicalId().det() == DetId::Forward && ForwardSubdetector(it->recHit()->geographicalId().subdetId()) == FastTime;
        if (ismtd) {
          const auto &propresult2 = thePropagator->propagateWithPath(tscbl.trackStateAtPCA(), trajWithMtd.lastMeasurement().updatedState().surface());
          pathlength2 = propresult2.second;
          if (pathlength2 == 0.) {
            validpropagation = false;
          }
          pathlength = pathlength1 + pathlength2;
          const MTDTrackingRecHit *mtdhit = static_cast<const MTDTrackingRecHit*>(it->recHit()->hit());
          thit = mtdhit->time();
          thiterror = mtdhit->timeError();
          validmtd = true;
          break;          
        }
      }
    }
    
    if (validmtd && validpropagation) {
      
      double magp = p.mag();
      double gammasq = 1. + magp*magp/mpi/mpi;
      double beta = std::sqrt(1.-1./gammasq);
      double dt = pathlength/beta/c;
      pathLength = pathlength; // set path length if we've got a timing hit
      t0 = thit - dt;
      covt0t0 = thiterror*thiterror;
      beta = betaOut;
      //printf("Propagation to MTD with pathlength1 = %5f, pathlength2 = %5f, path length = %5f, beta = %5f, thit = %5f, dt = %5f, t0 = %5f\n", pathlength1, pathlength2,pathlength,beta,thit,dt,t0);
    }
  }
  
  return reco::Track(traj.chiSquared(),
		     int(ndof),
		     pos, mom, tscbl.trackStateAtPCA().charge(), 
		     tscbl.trackStateAtPCA().curvilinearError(),
		     orig.algo(),reco::TrackBase::undefQuality,t0,betaOut,covt0t0,-1.);
}

template<class TrackCollection>
reco::TrackExtra TrackExtenderWithMTDT<TrackCollection>::buildTrackExtra(const Trajectory& trajectory) const {

  static const string metname = "MTD|RecoMTD|TrackExtenderWithMTD";

  const Trajectory::RecHitContainer transRecHits = trajectory.recHits();
  
  // put the collection of TrackingRecHit in the event
  
  // sets the outermost and innermost TSOSs
  // FIXME: check it!
  TrajectoryStateOnSurface outerTSOS;
  TrajectoryStateOnSurface innerTSOS;
  unsigned int innerId=0, outerId=0;
  TrajectoryMeasurement::ConstRecHitPointer outerRecHit;
  DetId outerDetId;

  if (trajectory.direction() == alongMomentum) {
    LogTrace(metname)<<"alongMomentum";
    outerTSOS = trajectory.lastMeasurement().updatedState();
    innerTSOS = trajectory.firstMeasurement().updatedState();
    outerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    outerRecHit =  trajectory.lastMeasurement().recHit();
    outerDetId =   trajectory.lastMeasurement().recHit()->geographicalId();
  } 
  else if (trajectory.direction() == oppositeToMomentum) {
    LogTrace(metname)<<"oppositeToMomentum";
    outerTSOS = trajectory.firstMeasurement().updatedState();
    innerTSOS = trajectory.lastMeasurement().updatedState();
    outerId = trajectory.firstMeasurement().recHit()->geographicalId().rawId();
    innerId = trajectory.lastMeasurement().recHit()->geographicalId().rawId();
    outerRecHit =  trajectory.firstMeasurement().recHit();
    outerDetId =   trajectory.firstMeasurement().recHit()->geographicalId();
  }
  else LogError(metname)<<"Wrong propagation direction!";
  
  const GeomDet *outerDet = gtg->idToDet(outerDetId);
  GlobalPoint outerTSOSPos = outerTSOS.globalParameters().position();
  bool inside = outerDet->surface().bounds().inside(outerDet->toLocal(outerTSOSPos));

  
  GlobalPoint hitPos = (outerRecHit->isValid()) ? outerRecHit->globalPosition() :  outerTSOS.globalParameters().position() ;
  
  if(!inside) {
    LogTrace(metname)<<"The Global Muon outerMostMeasurementState is not compatible with the recHit detector! Setting outerMost postition to recHit position if recHit isValid: " << outerRecHit->isValid();
    LogTrace(metname)<<"From " << outerTSOSPos << " to " <<  hitPos;
  }
  
  
  //build the TrackExtra
  GlobalPoint v = (inside) ? outerTSOSPos : hitPos ;
  GlobalVector p = outerTSOS.globalParameters().momentum();
  math::XYZPoint  outpos( v.x(), v.y(), v.z() );   
  math::XYZVector outmom( p.x(), p.y(), p.z() );
  
  v = innerTSOS.globalParameters().position();
  p = innerTSOS.globalParameters().momentum();
  math::XYZPoint  inpos( v.x(), v.y(), v.z() );   
  math::XYZVector inmom( p.x(), p.y(), p.z() );

  reco::TrackExtra trackExtra(outpos, outmom, true, inpos, inmom, true,
                              outerTSOS.curvilinearError(), outerId,
                              innerTSOS.curvilinearError(), innerId,
			      trajectory.direction(),trajectory.seedRef());
  
  return trackExtra;
 
}

template<class TrackCollection>
string TrackExtenderWithMTDT<TrackCollection>::dumpLayer(const DetLayer* layer) const {
  stringstream output;
  
  const BoundSurface* sur=nullptr;
  const BoundCylinder* bc=nullptr;
  const BoundDisk* bd=nullptr;

  sur = &(layer->surface());
  if ( (bc = dynamic_cast<const BoundCylinder*>(sur)) ) {
    output << "  Cylinder of radius: " << bc->radius() << endl;
  }
  else if ( (bd = dynamic_cast<const BoundDisk*>(sur)) ) {
    output << "  Disk at: " <<  bd->position().z() << endl;
  }
  return output.str();
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
typedef TrackExtenderWithMTDT<reco::TrackCollection> TrackExtenderWithMTD;

DEFINE_FWK_MODULE(TrackExtenderWithMTD);
