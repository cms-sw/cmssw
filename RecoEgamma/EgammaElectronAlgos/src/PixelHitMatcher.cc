
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelHitMatcher.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchNextLayers.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronUtilities.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTracker.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <typeinfo>

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
   prop1stLayer(0), prop2ndLayer(0),theGeometricSearchTracker(0),theLayerMeasurements(0),vertex_(0.),
   searchInTIDTEC_(searchInTIDTEC), useRecoVertex_(false)
 {
  meas1stFLayer.setRRangeI(rMinI,rMaxI) ;
  meas2ndFLayer.setRRangeI(rMinI,rMaxI) ;
 }

PixelHitMatcher::~PixelHitMatcher()
 {
  delete prop1stLayer ;
  delete prop2ndLayer ;
  delete theLayerMeasurements ;
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

void PixelHitMatcher::setES
 ( const MagneticField * magField,
   const MeasurementTracker * theMeasurementTracker,
   const TrackerGeometry * trackerGeometry )
 {
  if (theMeasurementTracker)
   {
    theGeometricSearchTracker=theMeasurementTracker->geometricSearchTracker() ;
    startLayers.setup(theGeometricSearchTracker) ;
    if (theLayerMeasurements ) delete theLayerMeasurements ;
    theLayerMeasurements = new LayerMeasurements(theMeasurementTracker) ;
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

//CLHEP::Hep3Vector point_to_vector( const GlobalPoint & p )
// { return CLHEP::Hep3Vector(p.x(),p.y(),p.z()) ; }

std::vector<SeedWithInfo>
PixelHitMatcher::compatibleSeeds
 ( TrajectorySeedCollection * seeds, const GlobalPoint & xmeas,
   const GlobalPoint & vprim, float energy, float fcharge )
 {
  int charge = int(fcharge) ;

  FreeTrajectoryState fts = myFTS(theMagField,xmeas, vprim, energy, charge);
  PerpendicularBoundPlaneBuilder bpb;
  TrajectoryStateOnSurface tsos(fts, *bpb(fts.position(), fts.momentum()));

  std::vector<SeedWithInfo> result ;
  mapTsos_.clear() ;
  mapTsos2_.clear() ;
  mapTsos_.reserve(seeds->size()) ;
  mapTsos2_.reserve(seeds->size()) ;

  for (unsigned int i=0;i<seeds->size();++i)
   {
    if ((*seeds)[i].nHits()>9)
     {
      edm::LogWarning("GsfElectronAlgo|UnexpectedSeed") <<"We cannot deal with seeds having more than 9 hits." ;
      continue ;
     }
    TrajectorySeed::range rhits=(*seeds)[i].recHits() ;

    // build all possible pairs
    unsigned char rank1, rank2, hitsMask ;
    TrajectorySeed::const_iterator it1, it2 ;
    for ( rank1=0, it1=rhits.first ; it1!=rhits.second ; rank1++, it1++ )
     {
      for ( rank2=rank1+1, it2=it1+1 ; it2!=rhits.second ; rank2++, it2++ )
       {
        //TrajectorySeed::range r(it1,it2) ;

        // first Hit
        TrajectorySeed::const_iterator it=it1 ;
        if (!(*it).isValid()) continue;
        DetId id=(*it).geographicalId();
        const GeomDet *geomdet=theTrackerGeometry->idToDet((*it).geographicalId());
        LocalPoint lp=(*it).localPosition() ;
        GlobalPoint hitPos=geomdet->surface().toGlobal(lp) ;

        TrajectoryStateOnSurface tsos1;
        bool found = false;
        std::vector<std::pair<const GeomDet *, TrajectoryStateOnSurface> >::iterator itTsos ;
        for (itTsos=mapTsos_.begin();itTsos!=mapTsos_.end();++itTsos)
         {
          if ((*itTsos).first==geomdet)
           { found=true ; break ; }
         }
        if (!found)
         {
          tsos1 = prop1stLayer->propagate(tsos,geomdet->surface()) ;
          mapTsos_.push_back(std::pair<const GeomDet *, TrajectoryStateOnSurface>(geomdet,tsos1));
         }
        else
         { tsos1=(*itTsos).second ; }

        if (tsos1.isValid())
         {
          std::pair<bool,double> est;
          if (id.subdetId()%2==1) est=meas1stBLayer.estimate(vprim, tsos1,hitPos);
          else est=meas1stFLayer.estimate(vprim, tsos1,hitPos);
          if (!est.first) continue ;

          if (std::abs(normalized_phi(hitPos.phi()-xmeas.phi()))>2.5) continue ;
          EleRelPointPair pp1(hitPos,tsos1.globalParameters().position(),vprim) ;
          int subDet1 = id.subdetId() ;
          float dRz1 = (subDet1%2==1)?pp1.dZ():pp1.dPerp() ;
          float dPhi1 = pp1.dPhi() ;

          // now second Hit
          //CC@@
          //it++;
          it=it2 ;
          if (!(*it).isValid()) continue ;

          DetId id2=(*it).geographicalId();
          const GeomDet *geomdet2=theTrackerGeometry->idToDet((*it).geographicalId());
          TrajectoryStateOnSurface tsos2;

          double zVertex;
          if (!useRecoVertex_) // we don't know the z vertex position, get it from linear extrapolation
           {
            // compute the z vertex from the cluster point and the found pixel hit
            double pxHit1z = hitPos.z();
            double pxHit1x = hitPos.x();
            double pxHit1y = hitPos.y();
            double r1diff = (pxHit1x-vprim.x())*(pxHit1x-vprim.x()) + (pxHit1y-vprim.y())*(pxHit1y-vprim.y()) ;
            r1diff=sqrt(r1diff) ;
            double r2diff = (xmeas.x()-pxHit1x)*(xmeas.x()-pxHit1x) + (xmeas.y()-pxHit1y)*(xmeas.y()-pxHit1y) ;
            r2diff=sqrt(r2diff);
            zVertex = pxHit1z - r1diff*(xmeas.z()-pxHit1z)/r2diff;
           }
          else // here use rather the reco vertex z position
           { zVertex = vprim.z() ; }

          GlobalPoint vertex(vprim.x(),vprim.y(),zVertex) ;
          FreeTrajectoryState fts2 = myFTS(theMagField,hitPos,vertex,energy, charge) ;

          found = false;
          std::vector<std::pair< std::pair<const GeomDet *,GlobalPoint>, TrajectoryStateOnSurface> >::iterator itTsos2 ;
          for (itTsos2=mapTsos2_.begin();itTsos2!=mapTsos2_.end();++itTsos2)
           {
            if (((*itTsos2).first).first==geomdet2 &&
                (((*itTsos2).first).second).x()==hitPos.x() &&
                (((*itTsos2).first).second).y()==hitPos.y() &&
                (((*itTsos2).first).second).z()==hitPos.z()  )
             {
              found=true;
              break;
             }
           }
          if (!found)
           {
            tsos2 = prop2ndLayer->propagate(fts2,geomdet2->surface()) ;
            std::pair<const GeomDet *,GlobalPoint> pair(geomdet2,hitPos);
            mapTsos2_.push_back(std::pair<std::pair<const GeomDet *,GlobalPoint>, TrajectoryStateOnSurface> (pair,tsos2));
           }
          else
           { tsos2=(*itTsos2).second ; }

          if (tsos2.isValid())
           {
            LocalPoint lp2=(*it).localPosition() ;
            GlobalPoint hitPos2=geomdet2->surface().toGlobal(lp2) ;
            std::pair<bool,double> est2 ;
            if (id2.subdetId()%2==1) est2=meas2ndBLayer.estimate(vertex, tsos2,hitPos2) ;
            else est2=meas2ndFLayer.estimate(vertex, tsos2,hitPos2) ;
            if (est2.first)
             {
              EleRelPointPair pp2(hitPos2,tsos2.globalParameters().position(),vertex) ;
              int subDet2 = id2.subdetId() ;
              float dRz2 = (subDet2%2==1)?pp2.dZ():pp2.dPerp() ;
              float dPhi2 = pp2.dPhi() ;
              hitsMask = (1<<rank1)|(1<<rank2) ;
              result.push_back(SeedWithInfo((*seeds)[i],hitsMask,subDet2,dRz2,dPhi2,subDet1,dRz1,dPhi1)) ;
             }
           }
         } // end tsos1 is valid
       } // end loop on second seed hit
     } // end loop on first seed hit
   } // end loop on seeds

  mapTsos_.clear() ;
  mapTsos2_.clear() ;

  return result ;
 }

//========================= OBSOLETE ? =========================

vector< pair< RecHitWithDist, PixelHitMatcher::ConstRecHitPointer > >
PixelHitMatcher::compatibleHits
 ( const GlobalPoint & xmeas,
   const GlobalPoint & vprim,
   float energy, float fcharge,
   const TrackerTopology *tTopo)
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

  typedef vector<BarrelDetLayer*>::const_iterator BarrelLayerIterator;
  BarrelLayerIterator firstLayer = startLayers.firstBLayer();

  FreeTrajectoryState fts =myFTS(theMagField,xmeas, vprim,
				 energy, charge);

  PerpendicularBoundPlaneBuilder bpb;
  TrajectoryStateOnSurface tsos(fts, *bpb(fts.position(), fts.momentum()));

  if (tsos.isValid()) {
    vector<TrajectoryMeasurement> pixelMeasurements =
      theLayerMeasurements->measurements(**firstLayer,tsos,
					 *prop1stLayer, meas1stBLayer);

    LogDebug("") <<"[PixelHitMatcher::compatibleHits] nbr of hits compatible with extrapolation to first layer: " << pixelMeasurements.size();
    for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++){
     if (m->recHit()->isValid()) {
       float localDphi = normalized_phi(SCl_phi-m->forwardPredictedState().globalPosition().phi()) ;
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
      theLayerMeasurements->measurements(**firstLayer,tsos,
					 *prop1stLayer, meas1stBLayer);

    for (aMeas m=pixel2Measurements.begin(); m!=pixel2Measurements.end(); m++){
      if (m->recHit()->isValid()) {
	float localDphi = normalized_phi(SCl_phi-m->forwardPredictedState().globalPosition().phi()) ;
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
  typedef vector<ForwardDetLayer*>::const_iterator ForwardLayerIterator;
  ForwardLayerIterator flayer;

  TrajectoryStateOnSurface tsosfwd(fts, *bpb(fts.position(), fts.momentum()));
  if (tsosfwd.isValid()) {

    for (int i=0; i<2; i++) {
      i == 0 ? flayer = startLayers.pos1stFLayer() : flayer = startLayers.neg1stFLayer();

      if (i==0 && xmeas.z() < -100. ) continue;
      if (i==1 && xmeas.z() > 100. ) continue;

      vector<TrajectoryMeasurement> pixelMeasurements =
	theLayerMeasurements->measurements(**flayer, tsosfwd,
					   *prop1stLayer, meas1stFLayer);

      for (aMeas m=pixelMeasurements.begin(); m!=pixelMeasurements.end(); m++){
	if (m->recHit()->isValid()) {
	  float localDphi = normalized_phi(SCl_phi-m->forwardPredictedState().globalPosition().phi());
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
	  theLayerMeasurements->measurements(**flayer, tsosfwd,
					     *prop1stLayer, meas1stFLayer);

	for (aMeas m=pixel2Measurements.begin(); m!=pixel2Measurements.end(); m++){
	  if (m->recHit()->isValid()) {
	    float localDphi = normalized_phi(SCl_phi-m->forwardPredictedState().globalPosition().phi()) ;
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

    FreeTrajectoryState secondFTS=myFTS(theMagField,hitPos,vertexPred,energy, charge);

    PixelMatchNextLayers secondHit(theLayerMeasurements, newLayer, secondFTS,
				   prop2ndLayer, &meas2ndBLayer,&meas2ndFLayer,
				   tTopo,searchInTIDTEC_);
    vector<CLHEP::Hep3Vector> predictions = secondHit.predictionInNextLayers();

    for (unsigned it = 0; it < predictions.size(); it++) pred2Meas.push_back(predictions[it]);

    // we may get more than one valid second measurements here even for single electrons:
    // two hits from the same layer/disk (detector overlap) or from the loop over the
    // next layers in EPMatchLoopNextLayers. Take only the 1st hit.

    if(!secondHit.measurementsInNextLayers().empty()){
      for(unsigned int shit=0; shit<secondHit.measurementsInNextLayers().size(); shit++)
      	{
	  float dphi = normalized_phi(pred1Meas[i].phi()-validMeasurements[i].recHit()->globalPosition().phi()) ;
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


