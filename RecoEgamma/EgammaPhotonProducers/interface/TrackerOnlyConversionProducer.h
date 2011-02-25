#ifndef RecoEgamma_EgammaPhotonProducers_TrackerOnlyConversionProducer_h
#define RecoEgamma_EgammaPhotonProducers_TrackerOnlyConversionProducer_h
/** \class TrackerOnlyConversionProducer
 **
 **
 **  $Id:
 **  $Date: 2011/01/26 19:59:07 $
 **  $Revision: 1.22.2.1 $
 **  \authors H. Liu, UC of Riverside US, N. Marinelli Univ of Notre Dame
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

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicVertex.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicTree.h"
#include "RecoVertex/KinematicFitPrimitives/interface/TransientTrackKinematicParticle.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"


//Tracker tracks
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"


//photon data format
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaTrackReco/interface/ConversionTrackFwd.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

class TransientTrackBuilder;
class ConversionVertexFinder;

class TrackerOnlyConversionProducer : public edm::EDProducer {
    public:
      explicit TrackerOnlyConversionProducer(const edm::ParameterSet&);
      ~TrackerOnlyConversionProducer();

      void buildCollection( edm::Event& iEvent, const edm::EventSetup& iSetup,
			    //const reco::TrackRefVector& allTracks,
			    const reco::ConversionTrackCollection& allTracks,
	      const std::multimap<double, reco::CaloClusterPtr>& basicClusterPtrs,
	      const reco::Vertex& the_pvtx,
	      reco::ConversionCollection & outputConvPhotonCollection);

      void buildCollection( edm::Event& iEvent, const edm::EventSetup& iSetup,
	      const reco::ConversionTrackCollection& allTracks,
	      const reco::CaloClusterPtr& basicClusterPtrs,
	      const reco::Vertex& the_pvtx,
	      reco::ConversionCollection & outputConvPhotonCollection);

      //track quality cut, returns pass or no
      inline bool trackQualityFilter(const  edm::RefToBase<reco::Track>&  ref, bool isLeft);
      inline bool trackD0Cut(const edm::RefToBase<reco::Track>& ref);
      inline bool trackD0Cut(const edm::RefToBase<reco::Track>& ref, const reco::Vertex& the_pvtx);

      //track impact point at ECAL wall, returns validity to access position ew
      bool getTrackImpactPosition(const reco::Track* tk_ref, 
	      const TrackerGeometry* trackerGeom, const MagneticField* magField, 
	      math::XYZPoint& ew);

      //distance at min approaching point, returns distance
      //      double getMinApproach(const edm::RefToBase<reco::Track>& ll, const edm::RefToBase<reco::Track>& rr, 
      //	      const MagneticField* magField);

      bool preselectTrackPair(const reco::TransientTrack &ttk_l, const reco::TransientTrack &ttk_r,
              double& appDist);
              
      //cut-based selection, TODO remove global cut variables
      bool checkTrackPair(const std::pair<edm::RefToBase<reco::Track>, reco::CaloClusterPtr>& ll,
	      const std::pair<edm::RefToBase<reco::Track>, reco::CaloClusterPtr>& rr);

      //kinematic vertex fitting, return true for valid vertex
      bool checkVertex(const reco::TransientTrack &ttk_l, const reco::TransientTrack &ttk_r,
	      const MagneticField* magField,
	      reco::Vertex& the_vertex);
      bool checkPhi(const edm::RefToBase<reco::Track>& tk_l, const edm::RefToBase<reco::Track>& tk_r,
	      const TrackerGeometry* trackerGeom, const MagneticField* magField,
	      const reco::Vertex& the_vertex);

      //check the closest BC, returns true for found a BC
      bool getMatchedBC(const std::multimap<double, reco::CaloClusterPtr>& bcMap, 
	      const math::XYZPoint& trackImpactPosition,
	      reco::CaloClusterPtr& closestBC);

      bool getMatchedBC(const reco::CaloClusterPtrVector& bcMap, 
	      const math::XYZPoint& trackImpactPosition,
	      reco::CaloClusterPtr& closestBC);



   private:

      virtual void produce(edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
      std::string algoName_;

      typedef math::XYZPointD Point;
      typedef std::vector<Point> PointCollection;

      edm::InputTag src_; 

      edm::InputTag bcBarrelCollection_;
      edm::InputTag bcEndcapCollection_;
      std::string ConvertedPhotonCollection_;

      bool allowD0_, allowDeltaPhi_, allowTrackBC_, allowDeltaCot_, allowMinApproach_, allowOppCharge_, allowVertex_;

      bool usePvtx_;//if use primary vertices
      std::string vertexProducer_;
      ConversionVertexFinder*         theVertexFinder_;

      const TransientTrackBuilder *thettbuilder_;


      double halfWayEta_, halfWayPhi_;//halfway open angle to search in basic clusters
      unsigned int  maxNumOfTrackInPU_;

      double energyBC_;//1.5GeV for track BC selection
      double energyTotalBC_;//5GeV for track pair BC selection
      double d0Cut_;//0 for d0*charge cut
      double dzCut_;//innerposition of z diff cut
      double dEtaTkBC_, dPhiTkBC_;//0.06 0.6 for track and BC matching

      double maxChi2Left_, maxChi2Right_;//5. 5. for track chi2 quality
      double minHitsLeft_, minHitsRight_;//5 2 for track hits quality 

      double deltaCotTheta_, deltaPhi_, minApproachLow_, minApproachHigh_;//0.02 0.2 for track pair open angle and > -0.1 cm


      double r_cut;//cross_r cut
      double vtxChi2_;//vertex chi2 probablity cut

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

#endif
