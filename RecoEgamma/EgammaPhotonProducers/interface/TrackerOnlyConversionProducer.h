#ifndef RecoEgamma_EgammaPhotonProducers_TrackerOnlyConversionProducer_h
#define RecoEgamma_EgammaPhotonProducers_TrackerOnlyConversionProducer_h
/** \class TrackerOnlyConversionProducer
 **
 **
 **  $Id:
 **  $Date: 2009/02/06 15:45:55 $
 **  $Revision: 1.1 $
 **  \author H. Liu, UC of Riverside US
 **
 ***/
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//ECAL clusters
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include "DataFormats/GeometrySurface/interface/SimpleCylinderBounds.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"


//Tracker tracks
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

class TrackerOnlyConversionProducer : public edm::EDProducer {
    public:
      explicit TrackerOnlyConversionProducer(const edm::ParameterSet&);
      ~TrackerOnlyConversionProducer();

      void getCircleCenter(const reco::TrackRef& tk, const double r, double& x0, double& y0, bool muon = false);
      void getCircleCenter(const edm::RefToBase<reco::Track>& tk, const double r, double& x0, double& y0, bool muon = false);
   private:

      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&);
      virtual void endRun(const edm::Run&, const edm::EventSetup&);

      // ----------member data ---------------------------
      typedef math::XYZPointD Point;
      typedef std::vector<Point> PointCollection;

      std::vector<edm::InputTag>  src_; 

      edm::InputTag bcBarrelCollection_;
      edm::InputTag bcEndcapCollection_;
      std::string ConvertedPhotonCollection_;

      bool allowD0_, allowTrackBC_, allowDeltaCot_, allowMinApproach_, allowOppCharge_;

      double halfWayEta_, halfWayPhi_;//halfway open angle to search in basic clusters

      double energyBC_;//1.5GeV for track BC selection
      double energyTotalBC_;//5GeV for track pair BC selection
      double d0Cut_;//0 for d0*charge cut
      double dEtaTkBC_, dPhiTkBC_;//0.06 0.6 for track and BC matching

      double maxChi2Left_, maxChi2Right_;//5. 5. for track chi2 quality
      double maxHitsLeft_, maxHitsRight_;//5 2 for track hits quality 

      double deltaCotTheta_, deltaPhi_, minApproach_;//0.02 0.2 for track pair open angle and > -0.1 cm

      bool allowSingleLeg_;//if single track conversion ?
      bool rightBC_;//if right leg requires matching BC?

};


inline const GeomDet * recHitDet( const TrackingRecHit & hit, const TrackingGeometry * geom ) {
    return geom->idToDet( hit.geographicalId() );
}

inline const BoundPlane & recHitSurface( const TrackingRecHit & hit, const TrackingGeometry * geom ) {
    return recHitDet( hit, geom )->surface();
}

inline LocalVector toLocal( const reco::Track::Vector & v, const Surface & s ) {
    return s.toLocal( GlobalVector( v.x(), v.y(), v.z() ) );
}

inline double map_phi2(double phi) {
    // map phi to [-pi,pi]
    double result = phi;
    if ( result < 1.0*Geom::pi() ) result = result + Geom::twoPi();
    if ( result >= Geom::pi())  result = result - Geom::twoPi();
    return result;
}



#endif
