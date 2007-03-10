#ifndef HTrackAssociator_HTrackAssociator_h
#define HTrackAssociator_HTrackAssociator_h 1

// -*- C++ -*-
//
// Package:    HTrackAssociator
// Class:      HTrackAssociator
// 
/*

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Dmytro Kovalskyi
// Modified for ECAL+HCAL by:  Michal Szleper
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "Calibration/Tools/interface/CaloDetIdAssociator.h"
#include "Calibration/Tools/interface/EcalDetIdAssociator.h"
#include "Calibration/Tools/interface/HcalDetIdAssociator.h"
#include "Calibration/Tools/interface/TrackDetMatchInfo.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"


class HTrackAssociator {
 public:
   HTrackAssociator();
   ~HTrackAssociator();
   
   class HAssociatorParameters {
    public:
      HAssociatorParameters() {
	 // default parameters
	 // define match cones, dR=sqrt(dEta^2+dPhi^2)
	 dREcal = 0.03;
	 dRHcal = 0.07;
	 dRCalo = 0.07;
	 
	 idREcal = 1;
	 idRHcal = 4;
	 idRCalo = 4;
	 
	 // match all sub-detectors by default
	 useEcal = true;
	 useHcal = true;
	 useCalo = true;
      }
      double dREcal;
      double dRHcal;
      double dRCalo;
      int idREcal;
      int idRHcal;
      int idRCalo;
      
      bool useEcal;
      bool useHcal;
      bool useCalo;
   };
   
   
   /// propagate a track across the whole detector and
   /// find associated objects. Association is done in
   /// two modes 
   ///  1) an object is associated to a track only if 
   ///     crossed by track
   ///  2) an object is associated to a track if it is
   ///     withing an eta-phi cone of some radius with 
   ///     respect to a track.
   ///     (the cone origin is at (0,0,0))
   HTrackDetMatchInfo            associate( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const HAssociatorParameters& );

   /// associate ECAL only and return RecHits
   /// negative dR means only crossed elements
   std::vector<EcalRecHit>  associateEcal( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const double dR = -1 );
   
   /// associate ECAL only and return energy
   /// negative dR means only crossed elements
   double                   getEcalEnergy( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const double dR = -1 );
   
   /// associate ECAL only and return RecHits
   /// negative dR means only crossed elements
   std::vector<CaloTower>   associateHcal( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const double dR = -1 );

   /// associate ECAL only and return energy
   /// negative dR means only crossed elements
   double                   getHcalEnergy( const edm::Event&,
					   const edm::EventSetup&,
					   const FreeTrajectoryState&,
					   const double dR = -1 );
   /// use a user configured propagator
   void setPropagator( Propagator* );
   
   /// use the default propagator
   void useDefaultPropagator();
   
   /// specify names of EDProducts to use for different input data types
   void addDataLabels( const std::string className,
		       const std::string moduleLabel,
		       const std::string productInstanceLabel = "");
   
   /// get FreeTrajectoryState from different track representations
   FreeTrajectoryState getFreeTrajectoryState( const edm::EventSetup&, 
					       const reco::Track& );
   FreeTrajectoryState getFreeTrajectoryState( const edm::EventSetup&, 
					       const SimTrack&, 
					       const SimVertex& );
   
 private:
   void       fillEcal( const edm::Event&,
			const edm::EventSetup&,
			HTrackDetMatchInfo&, 
			const FreeTrajectoryState&,
			const int,
			const double);

   void       fillHcal( const edm::Event&,
                        const edm::EventSetup&,
                        HTrackDetMatchInfo&,
                        const FreeTrajectoryState&,
                        const int,
                        const double);

   void fillHcalTowers( const edm::Event&,
			const edm::EventSetup&,
			HTrackDetMatchInfo&, 
			const FreeTrajectoryState&,
			const int,
			const double);
   
   void fillCaloTowers( const edm::Event&,
			const edm::EventSetup&,
			HTrackDetMatchInfo&, 
			const FreeTrajectoryState&,
			const int,
			const double);
   
   void           init( const edm::EventSetup&);
   
   math::XYZPoint getPoint( const GlobalPoint& point)
     {
	return math::XYZPoint(point.x(),point.y(),point.z());
     }
   
   math::XYZPoint getPoint( const LocalPoint& point)
     {
	return math::XYZPoint(point.x(),point.y(),point.z());
     }
   
   math::XYZVector getVector( const GlobalVector& vec)
     {
	return math::XYZVector(vec.x(),vec.y(),vec.z());
     }
   
   math::XYZVector getVector( const LocalVector& vec)
     {
	return math::XYZVector(vec.x(),vec.y(),vec.z());
     }
   
   Propagator* ivProp_;
   Propagator* defProp_;
   bool useDefaultPropagator_;
   int debug_;
   std::vector<std::vector<std::set<uint32_t> > >* caloTowerMap_;
   
   HEcalDetIdAssociator ecalDetIdAssociator_;
   HHcalDetIdAssociator hcalDetIdAssociator_;
   HCaloDetIdAssociator caloDetIdAssociator_;
   
   edm::ESHandle<CaloGeometry> theCaloGeometry_;
   
   /// Labels of the detector EDProducts (empty by default)
   /// ECAL
   std::vector<std::string> EBRecHitCollectionLabels;
   std::vector<std::string> EERecHitCollectionLabels;
   /// HCAL
   std::vector<std::string> HBHERecHitCollectionLabels;
   /// CaloTowers
   std::vector<std::string> CaloTowerCollectionLabels;
};
#endif
