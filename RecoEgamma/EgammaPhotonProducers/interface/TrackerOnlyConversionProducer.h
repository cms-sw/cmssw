#ifndef RecoEgamma_EgammaPhotonProducers_TrackerOnlyConversionProducer_h
#define RecoEgamma_EgammaPhotonProducers_TrackerOnlyConversionProducer_h
/** \class TrackerOnlyConversionProducer
 **
 **
 **  $Id:
 **  $Date: 2009/03/25 13:56:04 $
 **  $Revision: 1.2 $
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

//photon data format
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"


#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

using namespace edm;
using namespace reco;
using namespace std;

class TrackerOnlyConversionProducer : public edm::EDProducer {
    public:
      explicit TrackerOnlyConversionProducer(const edm::ParameterSet&);
      ~TrackerOnlyConversionProducer();

      void buildCollection( edm::Event& iEvent, const edm::EventSetup& iSetup,
	      const reco::TrackRefVector& allTracks,
	      const std::multimap<double, reco::CaloClusterPtr>& basicClusterPtrs,
	      reco::ConversionCollection & outputConvPhotonCollection);

      void buildCollection( edm::Event& iEvent, const edm::EventSetup& iSetup,
	      const reco::TrackRefVector& allTracks,
	      const reco::CaloClusterPtr& basicClusterPtrs,
	      reco::ConversionCollection & outputConvPhotonCollection);

      //track quality cut, returns pass or no
      inline bool trackQualityFilter(const edm::Ref<reco::TrackCollection>&  ref, bool isLeft);
      inline bool trackD0Cut(const edm::Ref<reco::TrackCollection>&  ref);

      //track impact point at ECAL wall, returns validity to access position ew
      bool getTrackImpactPosition(const TrackRef& tk_ref, 
	      const TrackerGeometry* trackerGeom, const MagneticField* magField, 
	      math::XYZPoint& ew);

      //distance at min approaching point, returns distance
      double getMinApproach(const TrackRef& ll, const TrackRef& rr, 
	      const MagneticField* magField);

      //cut-based selection, TODO remove global cut variables
      bool checkTrackPair(const std::pair<reco::TrackRef, reco::CaloClusterPtr>& ll,
	      const std::pair<reco::TrackRef, reco::CaloClusterPtr>& rr,
	      const MagneticField* magField,
	      double& appDist);

      //check the closest BC, returns true for found a BC
      bool getMatchedBC(const std::multimap<double, reco::CaloClusterPtr>& bcMap, 
	      const math::XYZPoint& trackImpactPosition,
	      reco::CaloClusterPtr& closestBC);

      bool getMatchedBC(const reco::CaloClusterPtrVector& bcMap, 
	      const math::XYZPoint& trackImpactPosition,
	      reco::CaloClusterPtr& closestBC);

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&);
      virtual void endRun(const edm::Run&, const edm::EventSetup&);

      inline void getCircleCenter(const reco::TrackRef& tk, 
	      const double r, double& x0, double& y0, 
	      bool muon = false);
      inline void getCircleCenter(const edm::RefToBase<reco::Track>& tk, 
	      const double r, double& x0, double& y0, 
	      bool muon = false);

      // ----------member data ---------------------------
      std::string algoName_;

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
      double minHitsLeft_, minHitsRight_;//5 2 for track hits quality 

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
