#ifndef PIXELHITMATCHER_H
#define PIXELHITMATCHER_H

// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      PixelHitMatcher
//
/**\class PixelHitMatcher EgammaElectronAlgos/PixelHitMatcher

 Description: Class to match an ECAL cluster to the pixel hits.
  Two compatible hits in the pixel layers are required.

 Implementation:
     future redesign
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
//
//

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/BarrelMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ForwardMeasurementEstimator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchStartLayers.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/FTSFromVertexToPointFactory.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

#include "RecoTracker/TransientTrackingRecHit/interface/TSiPixelRecHit.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "TrackingTools/MeasurementDet/interface/LayerMeasurements.h"

#include "CLHEP/Vector/ThreeVector.h"
#include <vector>
#include <unordered_map>
#include <limits>

/** Class to match an ECAL cluster to the pixel hits.
 *  Two compatible hits in the pixel layers are required.
 */

class MeasurementTracker;
class MeasurementTrackerEvent;
class MagneticField;
class GeometricSearchTracker;
class TrackerGeometry;
class TrackerTopology;

namespace std{
  template<>
    struct hash<std::pair<const GeomDet*,GlobalPoint> > {
      std::size_t operator()(const std::pair<const GeomDet*,GlobalPoint>& g) const {
	auto h1 = std::hash<unsigned long long>()((unsigned long long)g.first);
        unsigned long long k; memcpy(&k, &g.second,sizeof(k));
        auto h2 = std::hash<unsigned long long>()(k);
        return h1 ^ (h2 << 1);
      }
    };
  template<>
    struct equal_to<std::pair<const GeomDet*,GlobalPoint> > : public std::binary_function<std::pair<const GeomDet*,GlobalPoint>,std::pair<const GeomDet*,GlobalPoint>,bool> {
      bool operator()(const std::pair<const GeomDet*,GlobalPoint>& a, 
		      const std::pair<const GeomDet*,GlobalPoint>& b)  const {
	return (a.first == b.first) & (a.second == b.second);
      }
    };
}

class RecHitWithDist
 {
  public :

    typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
    typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
    typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
    

    RecHitWithDist( ConstRecHitPointer rh, float & dphi )
     : rh_(rh), dphi_(dphi)
     {}

    ConstRecHitPointer recHit() const { return rh_ ; }
    float dPhi() const { return dphi_ ; }

    void invert() { dphi_*=-1. ; }

  private :

    ConstRecHitPointer rh_ ;
    float dphi_ ;

 } ;


class RecHitWithInfo
 {
  public :

    typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer ;
    typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer ;
    typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer ;

    RecHitWithInfo( ConstRecHitPointer rh, int subDet =0,
       float dRz = std::numeric_limits<float>::infinity(),
       float dPhi = std::numeric_limits<float>::infinity() )
     : rh_(rh), subDet_(subDet), dRz_(dRz), dPhi_(dPhi)
     {}

    ConstRecHitPointer recHit() const { return rh_; }
    int subDet() const { return subDet_ ; }
    float dRz() const { return dRz_ ; }
    float dPhi() const { return dPhi_ ; }

    void invert() { dPhi_*=-1. ; }

  private:

    ConstRecHitPointer rh_;
    int subDet_ ;
    float dRz_ ;
    float dPhi_ ;

 } ;

class SeedWithInfo
 {
  public :

    SeedWithInfo( TrajectorySeed seed, unsigned char hitsMask, int subDet2, float dRz2, float dPhi2 , int subDet1, float dRz1, float dPhi1)
     : seed_(seed), hitsMask_(hitsMask),
       subDet2_(subDet2), dRz2_(dRz2), dPhi2_(dPhi2),
       subDet1_(subDet1), dRz1_(dRz1), dPhi1_(dPhi1)
     {}

    const TrajectorySeed & seed() const { return seed_ ; }
    unsigned char hitsMask() const { return hitsMask_ ; }

    int subDet2() const { return subDet2_ ; }
    float dRz2() const { return dRz2_ ; }
    float dPhi2() const { return dPhi2_ ; }

    int subDet1() const { return subDet1_ ; }
    float dRz1() const { return dRz1_ ; }
    float dPhi1() const { return dPhi1_ ; }

  private :

    TrajectorySeed seed_ ;
    unsigned char hitsMask_ ;
    int subDet2_ ;
    float dRz2_ ;
    float dPhi2_ ;
    int subDet1_ ;
    float dRz1_ ;
    float dPhi1_ ;
 } ;

class PixelHitMatcher
 {
  public :

    typedef TransientTrackingRecHit::ConstRecHitPointer   ConstRecHitPointer;
    typedef TransientTrackingRecHit::RecHitPointer        RecHitPointer;
    typedef TransientTrackingRecHit::RecHitContainer      RecHitContainer;
    // Next typedef uses double in ROOT 6 rather than Double32_t due to a bug in ROOT 5,
    // which otherwise would make ROOT5 files unreadable in ROOT6.  This does not increase
    // the size on disk, because due to the bug, double was actually stored on disk in ROOT 5.
    typedef ROOT::Math::PositionVector3D<ROOT::Math::CylindricalEta3D<double> > REPPoint;

    PixelHitMatcher
     ( float phi1min, float phi1max,
       //float phi2min, float phi2max,
       float phi2minB, float phi2maxB, float phi2minF, float phi2maxF,
		   float z2minB, float z2maxB, float r2minF, float r2maxF,
		   float rMinI, float rMaxI, bool searchInTIDTEC ) ;

    virtual ~PixelHitMatcher() ;
    void setES( const MagneticField *, const MeasurementTracker * theMeasurementTracker, const TrackerGeometry * trackerGeometry ) ;

    void setEvent( const MeasurementTrackerEvent & event ) ;

    std::vector<SeedWithInfo>
    compatibleSeeds
      ( TrajectorySeedCollection * seeds, const GlobalPoint & xmeas,
        const GlobalPoint & vprim, float energy, float charge ) ;


    std::vector<std::pair<RecHitWithDist,ConstRecHitPointer> >
    compatibleHits(const GlobalPoint& xmeas, const GlobalPoint& vprim, 
		   float energy, float charge,
		   const TrackerTopology *tTopo,
                   const NavigationSchool& navigationSchool) ;


    std::vector<CLHEP::Hep3Vector> predicted1Hits() ;
    std::vector<CLHEP::Hep3Vector> predicted2Hits();

    void set1stLayer( float dummyphi1min, float dummyphi1max ) ;
    void set1stLayerZRange( float zmin1, float zmax1 ) ;
    //void set2ndLayer( float dummyphi2min, float dummyphi2max ) ;
    void set2ndLayer( float dummyphi2minB, float dummyphi2maxB, float dummyphi2minF, float dummyphi2maxF ) ;

    float getVertex() ;
    void setUseRecoVertex( bool val ) ;

  private :

    RecHitContainer hitsInTrack ;

    std::vector<CLHEP::Hep3Vector> pred1Meas ;
    std::vector<CLHEP::Hep3Vector> pred2Meas ;
    BarrelMeasurementEstimator meas1stBLayer ;
    BarrelMeasurementEstimator meas2ndBLayer ;
    ForwardMeasurementEstimator meas1stFLayer ;
    ForwardMeasurementEstimator meas2ndFLayer ;
    PixelMatchStartLayers startLayers ;
    PropagatorWithMaterial * prop1stLayer ;
    PropagatorWithMaterial * prop2ndLayer ;
    const GeometricSearchTracker * theGeometricSearchTracker ;
    const MeasurementTrackerEvent *theTrackerEvent;
    const MeasurementTracker      *theTracker;
    LayerMeasurements theLayerMeasurements ;
    const MagneticField* theMagField ;
    const TrackerGeometry * theTrackerGeometry ;

    float vertex_;

    bool searchInTIDTEC_ ;
    bool useRecoVertex_ ;
    std::unordered_map<const GeomDet*, TrajectoryStateOnSurface> mapTsos_fast_;
    std::unordered_map<std::pair<const GeomDet*,GlobalPoint>, TrajectoryStateOnSurface> mapTsos2_fast_;    
    std::vector<GlobalPoint> hit_gp_map_;    
} ;

#endif








