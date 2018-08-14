#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchNextLayers.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <typeinfo>
#include <bitset>

using namespace reco ;
using namespace std ;

PixelHitMatcher::PixelHitMatcher
 ( float phi1min, float phi1max,
   float phi2minB, float phi2maxB, float phi2minF, float phi2maxF,
   float z2minB, float z2maxB, float r2minF, float r2maxF,
   float rMinI, float rMaxI, bool searchInTIDTEC)
 : //zmin1 and zmax1 are dummy at this moment, set from beamspot later
   meas1stBLayer(phi1min,phi1max,0.,0.), meas2ndBLayer(phi2minB,phi2maxB,z2minB,z2maxB),
   meas1stFLayer(phi1min,phi1max,0.,0.), meas2ndFLayer(phi2minF,phi2maxF,r2minF,r2maxF),
   startLayers(),
   prop1stLayer(nullptr), prop2ndLayer(nullptr),theGeometricSearchTracker(nullptr),theTrackerEvent(nullptr),theTracker(nullptr),vertex_(0.),
   searchInTIDTEC_(searchInTIDTEC), useRecoVertex_(false)
 {
  meas1stFLayer.setRRangeI(rMinI,rMaxI) ;
  meas2ndFLayer.setRRangeI(rMinI,rMaxI) ;
 }

PixelHitMatcher::~PixelHitMatcher()
 {
  delete prop1stLayer ;
  delete prop2ndLayer ;
 }

void PixelHitMatcher::set1stLayer( float dummyphi1min, float dummyphi1max )
 {
  meas1stBLayer.setPhiRange(dummyphi1min,dummyphi1max) ;
  meas1stFLayer.setPhiRange(dummyphi1min,dummyphi1max) ;
 }

void PixelHitMatcher::set1stLayerZRange( float zmin1, float zmax1 )
 {
  meas1stBLayer.setZRange(zmin1,zmax1) ;
  meas1stFLayer.setRRange(zmin1,zmax1) ;
 }

void PixelHitMatcher::set2ndLayer( float dummyphi2minB, float dummyphi2maxB, float dummyphi2minF, float dummyphi2maxF )
 {
  meas2ndBLayer.setPhiRange(dummyphi2minB,dummyphi2maxB) ;
  meas2ndFLayer.setPhiRange(dummyphi2minF,dummyphi2maxF) ;
 }

void PixelHitMatcher::setUseRecoVertex( bool val )
 { useRecoVertex_ = val ; }

void PixelHitMatcher::setEvent( const MeasurementTrackerEvent & trackerData ) 
 {
    theTrackerEvent = & trackerData;
    theLayerMeasurements = LayerMeasurements(*theTracker,*theTrackerEvent);
 }
void PixelHitMatcher::setES
 ( const MagneticField * magField,
   const MeasurementTracker * theMeasurementTracker,
   const TrackerGeometry * trackerGeometry )
 {
  if (theMeasurementTracker)
   {
    theTracker = theMeasurementTracker;
    theGeometricSearchTracker=theMeasurementTracker->geometricSearchTracker() ;
    startLayers.setup(theGeometricSearchTracker) ;
   }

  theMagField = magField ;
  theTrackerGeometry = trackerGeometry ;
  float mass=.000511 ; // electron propagation
  if (prop1stLayer) delete prop1stLayer ;
  prop1stLayer = new PropagatorWithMaterial(oppositeToMomentum,mass,theMagField) ;
  if (prop2ndLayer) delete prop2ndLayer ;
  prop2ndLayer = new PropagatorWithMaterial(alongMomentum,mass,theMagField) ;
 }

vector<CLHEP::Hep3Vector> PixelHitMatcher::predicted1Hits()
 { return pred1Meas ; }

vector<CLHEP::Hep3Vector> PixelHitMatcher::predicted2Hits()
 { return pred2Meas ; }

float PixelHitMatcher::getVertex()
 { return vertex_ ; }


std::vector<SeedWithInfo>
PixelHitMatcher::compatibleSeeds
 ( TrajectorySeedCollection * seeds, const GlobalPoint & xmeas,
   const GlobalPoint & vprim, float energy, float fcharge )
 {
   typedef std::unordered_map<const GeomDet *, TrajectoryStateOnSurface> DetTsosAssoc;
   typedef std::unordered_map<std::pair<const GeomDet*,GlobalPoint>, TrajectoryStateOnSurface> PosTsosAssoc;
   const int charge = int(fcharge) ;

   // auto xmeas_phi = xmeas.barePhi();
   auto xmeas_r = xmeas.perp();
   
   const float phicut = std::cos(2.5);


  FreeTrajectoryState fts = FTSFromVertexToPointFactory::get(*theMagField, xmeas, vprim, energy, charge);
  PerpendicularBoundPlaneBuilder bpb;
  TrajectoryStateOnSurface tsos(fts, *bpb(fts.position(), fts.momentum()));
  
  std::vector<SeedWithInfo> result ;
  
  //mapTsos_fast_.clear();  
  mapTsos2_fast_.clear();  
  // mapTsos_fast_.reserve(seeds->size()) ;
  mapTsos2_fast_.reserve(seeds->size()) ;

  // std::vector<TrajectoryStateOnSurface> vTsos(theTrackerGeometry->dets().size());
  // TrajectoryStateOnSurface vTsos[theTrackerGeometry->dets().size()];

  auto ndets = theTrackerGeometry->dets().size();

  int iTsos[ndets];
  for ( auto & i : iTsos) i=-1;
  std::vector<TrajectoryStateOnSurface> vTsos; vTsos.reserve(seeds->size());

  for(const auto& seed : *seeds) {
    hit_gp_map_.clear();
    if( seed.nHits() > 9 ) {
      edm::LogWarning("GsfElectronAlgo|UnexpectedSeed") <<"We cannot deal with seeds having more than 9 hits." ;
      continue;
    }
    
    const TrajectorySeed::range& hits = seed.recHits();
    // cache the global points
   
    for( auto it = hits.first; it != hits.second; ++it ) {
      hit_gp_map_.emplace_back(it->globalPosition());      
    }

    //iterate on the hits 
    auto he =  hits.second -1;   
    for( auto it1 = hits.first; it1 < he; ++it1 ) {
      if( !it1->isValid() ) continue;
      auto  idx1 = std::distance(hits.first,it1);
      const DetId id1 = it1->geographicalId();
      const GeomDet *geomdet1 = it1->det();

      auto ix1 = geomdet1->gdetIndex();

      /*  VI: this generates regression (other cut is just in phi). in my opinion it is safe and makes sense
      auto away = geomdet1->position().basicVector().dot(xmeas.basicVector()) <0;
      if (away) continue;
      */

      const GlobalPoint& hit1Pos = hit_gp_map_[idx1];
      auto dt = hit1Pos.x()*xmeas.x()+hit1Pos.y()*xmeas.y();
      if (dt<0) continue;
      if (dt<phicut*(xmeas_r*hit1Pos.perp())) continue;

      if(iTsos[ix1]<0)   {
        iTsos[ix1] = vTsos.size();
        vTsos.push_back(prop1stLayer->propagate(tsos,geomdet1->surface()));
      }
      auto tsos1 = &vTsos[iTsos[ix1]];

      if( !tsos1->isValid() ) continue;
      std::pair<bool, double> est = ( id1.subdetId() % 2 ? 
				      meas1stBLayer.estimate(vprim, *tsos1, hit1Pos) :
				      meas1stFLayer.estimate(vprim, *tsos1, hit1Pos)  );
      if( !est.first ) continue;
      EleRelPointPair pp1(hit1Pos,tsos1->globalParameters().position(),vprim);
      const math::XYZPoint relHit1Pos(hit1Pos-vprim), relTSOSPos(tsos1->globalParameters().position() - vprim);
      const int subDet1 = id1.subdetId();
      const float dRz1 = ( id1.subdetId()%2 ? pp1.dZ() : pp1.dPerp() );
      const float dPhi1 = pp1.dPhi();
      // setup our vertex
      double zVertex;
      if (!useRecoVertex_) {
	// we don't know the z vertex position, get it from linear extrapolation
	// compute the z vertex from the cluster point and the found pixel hit
	const double pxHit1z = hit1Pos.z();
	const double pxHit1x = hit1Pos.x();
	const double pxHit1y = hit1Pos.y();
	const double r1diff = std::sqrt( (pxHit1x-vprim.x())*(pxHit1x-vprim.x()) + 
					 (pxHit1y-vprim.y())*(pxHit1y-vprim.y())   );
	const double r2diff = std::sqrt( (xmeas.x()-pxHit1x)*(xmeas.x()-pxHit1x) + 
					 (xmeas.y()-pxHit1y)*(xmeas.y()-pxHit1y)   );
	zVertex = pxHit1z - r1diff*(xmeas.z()-pxHit1z)/r2diff;
      } else { 
	// here use rather the reco vertex z position
	zVertex = vprim.z(); 
      }
      GlobalPoint vertex(vprim.x(),vprim.y(),zVertex);
      FreeTrajectoryState fts2 = FTSFromVertexToPointFactory::get(*theMagField, hit1Pos, vertex, energy, charge) ;
      // now find the matching hit
      for( auto it2 = it1+1; it2 != hits.second; ++it2 ) {
	if( !it2->isValid() ) continue;
	auto idx2 = std::distance(hits.first,it2);
	const DetId id2 = it2->geographicalId();
	const GeomDet *geomdet2 = it2->det();
	const std::pair<const GeomDet *,GlobalPoint> det_key(geomdet2,hit1Pos);	
	const TrajectoryStateOnSurface* tsos2;
	auto tsos2_itr = mapTsos2_fast_.find(det_key);
	if( tsos2_itr != mapTsos2_fast_.end() ) {
	  tsos2 = &(tsos2_itr->second);
	} else {
	  auto empl_result =
	    mapTsos2_fast_.emplace(det_key,prop2ndLayer->propagate(fts2,geomdet2->surface()));
	  tsos2 = &(empl_result.first->second);
	}
	if( !tsos2->isValid() ) continue;
	const GlobalPoint& hit2Pos = hit_gp_map_[idx2];
	std::pair<bool,double> est2  = ( id2.subdetId()%2 ? 
					 meas2ndBLayer.estimate(vertex, *tsos2,hit2Pos) :
					 meas2ndFLayer.estimate(vertex, *tsos2,hit2Pos)   );
	if (est2.first) {
	  EleRelPointPair pp2(hit2Pos,tsos2->globalParameters().position(),vertex) ;
	  const int subDet2 = id2.subdetId();
	  const float dRz2 = (subDet2%2==1)?pp2.dZ():pp2.dPerp();
	  const float dPhi2 = pp2.dPhi();
	  const unsigned char hitsMask = (1<<idx1)|(1<<idx2);
	  result.push_back(SeedWithInfo(seed,hitsMask,subDet2,dRz2,dPhi2,subDet1,dRz1,dPhi1)) ;
	}
      }// inner loop on hits
    }// outer loop on hits
  }// loop on seeds  

  mapTsos2_fast_.clear() ;
 
  return result ;
 }

//========================= OBSOLETE ? =========================

vector< pair< RecHitWithDist, PixelHitMatcher::ConstRecHitPointer > >
PixelHitMatcher::compatibleHits
 ( const GlobalPoint & xmeas,
   const GlobalPoint & vprim,
   float energy, float fcharge,
   const TrackerTopology *tTopo,
   const NavigationSchool& navigationSchool)
 {
  float SCl_phi = xmeas.phi();

  int charge = int(fcharge);
  // return all compatible RecHit pairs (vector< TSiPixelRecHit>)
  vector<pair<RecHitWithDist, ConstRecHitPointer> > result;
  LogDebug("") << "[PixelHitMatcher::compatibleHits] entering .. ";

  vector<TrajectoryMeasurement> validMeasurements;
  vector<TrajectoryMeasurement> invalidMeasurements;

  typedef vector<TrajectoryMeasurement>::const_iterator aMeas;

  pred1Meas.clear();
  pred2Meas.clear();

  typedef vector<const BarrelDetLayer*>::const_iterator BarrelLayerIterator;
  BarrelLayerIterator firstLayer = startLayers.firstBLayer();

  FreeTrajectoryState fts = FTSFromVertexToPointFactory::get(*theMagField,xmeas, vprim, energy, charge);

  PerpendicularBoundPlaneBuilder bpb;
  TrajectoryStateOnSurface tsos(fts, *bpb(fts.position(), fts.momentum()));

  if (tsos.isValid()) {
    vector<TrajectoryMeasurement> pixelMeasurements =
      theLayerMeasurements.measurements(**firstLayer,tsos,
					 *prop1stLayer, meas1stBLayer);

    LogDebug("") <<"[PixelHitMatcher::compatibleHits] nbr of hits compatible with extrapolation to first layer: " << pixelMeasurements.size();
    for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++){
     if (m->recHit()->isValid()) {
       float localDphi = normalized_phi(SCl_phi-m->forwardPredictedState().globalPosition().barePhi()) ;
       if(std::abs(localDphi)>2.5)continue;
	CLHEP::Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
			      m->forwardPredictedState().globalPosition().y(),
			      m->forwardPredictedState().globalPosition().z());
	LogDebug("") << "[PixelHitMatcher::compatibleHits] compatible hit position " << m->recHit()->globalPosition();
	LogDebug("") << "[PixelHitMatcher::compatibleHits] predicted position " << m->forwardPredictedState().globalPosition();
	pred1Meas.push_back( prediction);

	validMeasurements.push_back(*m);

	LogDebug("") <<"[PixelHitMatcher::compatibleHits] Found a rechit in layer ";
	const BarrelDetLayer *bdetl = dynamic_cast<const BarrelDetLayer *>(*firstLayer);
	if (bdetl) {
	  LogDebug("") <<" with radius "<<bdetl->specificSurface().radius();
	}
	else  LogDebug("") <<"Could not downcast!!";
     }
    }


    // check if there are compatible 1st hits in the second layer
    firstLayer++;

    vector<TrajectoryMeasurement> pixel2Measurements =
      theLayerMeasurements.measurements(**firstLayer,tsos,
					 *prop1stLayer, meas1stBLayer);

    for (aMeas m=pixel2Measurements.begin(); m!=pixel2Measurements.end(); m++){
      if (m->recHit()->isValid()) {
	float localDphi = normalized_phi(SCl_phi-m->forwardPredictedState().globalPosition().barePhi()) ;
	if(std::abs(localDphi)>2.5)continue;
        CLHEP::Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
			      m->forwardPredictedState().globalPosition().y(),
			      m->forwardPredictedState().globalPosition().z());
	pred1Meas.push_back( prediction);
        LogDebug("")  << "[PixelHitMatcher::compatibleHits] compatible hit position " << m->recHit()->globalPosition() << endl;
        LogDebug("") << "[PixelHitMatcher::compatibleHits] predicted position " << m->forwardPredictedState().globalPosition() << endl;

	validMeasurements.push_back(*m);
	LogDebug("") <<"[PixelHitMatcher::compatibleHits] Found a rechit in layer ";
	const BarrelDetLayer *bdetl = dynamic_cast<const BarrelDetLayer *>(*firstLayer);
	if (bdetl) {
	  LogDebug("") <<" with radius "<<bdetl->specificSurface().radius();
	}
	else  LogDebug("") <<"Could not downcast!!";
      }

    }
  }


  // check if there are compatible 1st hits the forward disks
  typedef vector<const ForwardDetLayer*>::const_iterator ForwardLayerIterator;
  ForwardLayerIterator flayer;

  TrajectoryStateOnSurface tsosfwd(fts, *bpb(fts.position(), fts.momentum()));
  if (tsosfwd.isValid()) {

    for (int i=0; i<2; i++) {
      i == 0 ? flayer = startLayers.pos1stFLayer() : flayer = startLayers.neg1stFLayer();

      if (i==0 && xmeas.z() < -100. ) continue;
      if (i==1 && xmeas.z() > 100. ) continue;

      vector<TrajectoryMeasurement> pixelMeasurements =
	theLayerMeasurements.measurements(**flayer, tsosfwd,
					   *prop1stLayer, meas1stFLayer);

      for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++){
	if (m->recHit()->isValid()) {
	  float localDphi = normalized_phi(SCl_phi-m->forwardPredictedState().globalPosition().barePhi());
	  if(std::abs(localDphi)>2.5)continue;
	  CLHEP::Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
				m->forwardPredictedState().globalPosition().y(),
				m->forwardPredictedState().globalPosition().z());
	  pred1Meas.push_back( prediction);

	  validMeasurements.push_back(*m);
	}
      }

      //check if there are compatible 1st hits the outer forward disks
      if (searchInTIDTEC_) {
	flayer++;

	vector<TrajectoryMeasurement> pixel2Measurements =
	  theLayerMeasurements.measurements(**flayer, tsosfwd,
					     *prop1stLayer, meas1stFLayer);

	for (aMeas m=pixel2Measurements.begin(); m!=pixel2Measurements.end(); m++){
	  if (m->recHit()->isValid()) {
	    float localDphi = normalized_phi(SCl_phi-m->forwardPredictedState().globalPosition().barePhi()) ;
	    if(std::abs(localDphi)>2.5)continue;
	    CLHEP::Hep3Vector prediction(m->forwardPredictedState().globalPosition().x(),
				  m->forwardPredictedState().globalPosition().y(),
				  m->forwardPredictedState().globalPosition().z());
	    pred1Meas.push_back( prediction);

	    validMeasurements.push_back(*m);
	  }
	  //	else{std::cout<<" hit non valid "<<std::endl; }
	}  //end 1st hit in outer f disk
      }
    }
  }

  // now we have the vector of all valid measurements of the first point
  for (unsigned i=0; i<validMeasurements.size(); i++){

    const DetLayer * newLayer = theGeometricSearchTracker->detLayer(validMeasurements[i].recHit()->det()->geographicalId());

    double zVertex ;
    if (!useRecoVertex_)
     {
      // we don't know the z vertex position, get it from linear extrapolation
      // compute the z vertex from the cluster point and the found pixel hit
      double pxHit1z = validMeasurements[i].recHit()->det()->surface().toGlobal(
        validMeasurements[i].recHit()->localPosition()).z();
      double pxHit1x = validMeasurements[i].recHit()->det()->surface().toGlobal(
        validMeasurements[i].recHit()->localPosition()).x();
      double pxHit1y = validMeasurements[i].recHit()->det()->surface().toGlobal(
        validMeasurements[i].recHit()->localPosition()).y();
      double r1diff = (pxHit1x-vprim.x())*(pxHit1x-vprim.x()) + (pxHit1y-vprim.y())*(pxHit1y-vprim.y());
      r1diff=sqrt(r1diff);
      double r2diff = (xmeas.x()-pxHit1x)*(xmeas.x()-pxHit1x) + (xmeas.y()-pxHit1y)*(xmeas.y()-pxHit1y);
      r2diff=sqrt(r2diff);
      zVertex = pxHit1z - r1diff*(xmeas.z()-pxHit1z)/r2diff;
     }
    else
     {
      // here we use the reco vertex z position
      zVertex = vprim.z();
     }

    if (i==0)
     { vertex_ = zVertex; }

    GlobalPoint vertexPred(vprim.x(),vprim.y(),zVertex) ;
    GlobalPoint hitPos( validMeasurements[i].recHit()->det()->surface().toGlobal( validMeasurements[i].recHit()->localPosition() ) ) ;

    FreeTrajectoryState secondFTS = FTSFromVertexToPointFactory::get(*theMagField, hitPos, vertexPred, energy, charge);

    PixelMatchNextLayers secondHit(&theLayerMeasurements, newLayer, secondFTS,
				   prop2ndLayer, &meas2ndBLayer,&meas2ndFLayer,
				   tTopo,navigationSchool,searchInTIDTEC_);
    vector<CLHEP::Hep3Vector> predictions = secondHit.predictionInNextLayers();

    for (unsigned it = 0; it < predictions.size(); it++) pred2Meas.push_back(predictions[it]);

    // we may get more than one valid second measurements here even for single electrons:
    // two hits from the same layer/disk (detector overlap) or from the loop over the
    // next layers in EPMatchLoopNextLayers. Take only the 1st hit.

    if(!secondHit.measurementsInNextLayers().empty()){
      for(unsigned int shit=0; shit<secondHit.measurementsInNextLayers().size(); shit++)
      	{
	  float dphi = normalized_phi(pred1Meas[i].phi()-validMeasurements[i].recHit()->globalPosition().barePhi()) ;
	  if (std::abs(dphi)<2.5)
	    {
	      ConstRecHitPointer pxrh = validMeasurements[i].recHit();
	      RecHitWithDist rh(pxrh,dphi);

	      //  pxrh = secondHit.measurementsInNextLayers()[0].recHit();
	      pxrh = secondHit.measurementsInNextLayers()[shit].recHit();

	      pair<RecHitWithDist,ConstRecHitPointer> compatiblePair = pair<RecHitWithDist,ConstRecHitPointer>(rh,pxrh) ;
	      result.push_back(compatiblePair);
	      break;
	    }
	}
    }
  }
  return result;
}

