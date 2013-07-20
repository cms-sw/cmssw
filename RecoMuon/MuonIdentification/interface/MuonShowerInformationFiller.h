/**
 *  \package: MuonIdentification
 *  \class: MuonShowerInformationFiller
 *
 *  Description: class for muon shower identification
 *
 *  $Date: 2012/12/26 10:17:17 $
 *  $Revision: 1.4 $
 *
 *  \author: A. Svyatkovskiy, Purdue University
 *
 **/

#ifndef MuonIdentification_MuonShowerInformationFiller_h
#define MuonIdentification_MuonShowerInformationFiller_h

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include "DataFormats/MuonReco/interface/MuonShower.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}
namespace reco {class TransientTrack; class MuonShower;}

class MuonServiceProxy;
class Trajectory;
class Cylinder;
class Disk;
class BarrelDetLayer;
class ForwardDetLayer;
class TransientTrackingRecHitBuilder;
class GeometricSearchTracker;
class GlobalTrackingGeometry;
class MuonDetLayerGeometry;


class MuonShowerInformationFiller {

  public:

    typedef TransientTrackingRecHit::ConstRecHitContainer ConstRecHitContainer;
    typedef MuonTransientTrackingRecHit::MuonRecHitContainer MuonRecHitContainer;
    typedef MuonTransientTrackingRecHit::ConstMuonRecHitPointer ConstMuonRecHitPointer;
	
  public:

    ///constructors
    MuonShowerInformationFiller() {};
    MuonShowerInformationFiller(const edm::ParameterSet&);

    ///destructor
    ~MuonShowerInformationFiller();
   
    /// fill muon shower variables  
    reco::MuonShower fillShowerInformation( const reco::Muon& muon, const edm::Event&, const edm::EventSetup&);

    /// pass the Event to the algorithm at each event
    virtual void setEvent(const edm::Event&);

    /// set the services needed
    void setServices(const edm::EventSetup&);

    //set the data members
    void fillHitsByStation(const reco::Muon&);

  protected:

    const MuonServiceProxy* getService() const { return theService; }

  private:

    std::vector<float> theStationShowerDeltaR;
    std::vector<float> theStationShowerTSize;
    std::vector<int>   theAllStationHits;
    std::vector<int>   theCorrelatedStationHits;

    MuonServiceProxy* theService;

    GlobalPoint crossingPoint(const GlobalPoint&, const GlobalPoint&, const BarrelDetLayer* ) const;
    GlobalPoint crossingPoint(const GlobalPoint&, const GlobalPoint&, const Cylinder& ) const;
    GlobalPoint crossingPoint(const GlobalPoint&, const GlobalPoint&, const ForwardDetLayer* ) const;
    GlobalPoint crossingPoint(const GlobalPoint&, const GlobalPoint&, const Disk& ) const;
    std::vector<const GeomDet*> dtPositionToDets(const GlobalPoint&) const;
    std::vector<const GeomDet*> cscPositionToDets(const GlobalPoint&) const;
    MuonRecHitContainer findPerpCluster(MuonRecHitContainer& muonRecHits) const;
    MuonRecHitContainer findPhiCluster(MuonRecHitContainer&, const GlobalPoint&) const;
    TransientTrackingRecHit::ConstRecHitContainer findThetaCluster(TransientTrackingRecHit::ConstRecHitContainer&, const GlobalPoint&) const;
    TransientTrackingRecHit::ConstRecHitContainer hitsFromSegments(const GeomDet*,edm::Handle<DTRecSegment4DCollection>, edm::Handle<CSCSegmentCollection>) const;
    std::vector<const GeomDet*> getCompatibleDets(const reco::Track&) const;

   struct LessMag {
       LessMag(const GlobalPoint& point) : thePoint(point) {}
       bool operator()(const GlobalPoint& lhs,
                       const GlobalPoint& rhs) const{ 
            return (lhs - thePoint).mag() < (rhs -thePoint).mag();
        }
        bool operator()(const MuonTransientTrackingRecHit::MuonRecHitPointer& lhs,
                       const MuonTransientTrackingRecHit::MuonRecHitPointer& rhs) const{
           return (lhs->globalPosition() - thePoint).mag() < (rhs->globalPosition() -thePoint).mag();
        }
      GlobalPoint thePoint;
   };

   struct LessDPhi {
        LessDPhi(const GlobalPoint& point) : thePoint(point) {}
        bool operator()(const MuonTransientTrackingRecHit::MuonRecHitPointer& lhs,
                       const MuonTransientTrackingRecHit::MuonRecHitPointer& rhs) const{
           return deltaPhi(lhs->globalPosition().phi(), thePoint.phi()) < deltaPhi(rhs->globalPosition().phi(), thePoint.phi());
        }
      GlobalPoint thePoint;
    };

    struct AbsLessDPhi {
        AbsLessDPhi(const GlobalPoint& point) : thePoint(point) {}
        bool operator()(const MuonTransientTrackingRecHit::MuonRecHitPointer& lhs,
                       const MuonTransientTrackingRecHit::MuonRecHitPointer& rhs) const{
           return ( fabs(deltaPhi(lhs->globalPosition().phi(), thePoint.phi())) < fabs(deltaPhi(rhs->globalPosition().phi(), thePoint.phi())) );
        }
      GlobalPoint thePoint;
    };

    struct AbsLessDTheta {
        AbsLessDTheta(const GlobalPoint& point) : thePoint(point) {}
        bool operator()(const TransientTrackingRecHit::ConstRecHitPointer& lhs,
                       const TransientTrackingRecHit::ConstRecHitPointer& rhs) const{
           return ( fabs(lhs->globalPosition().phi() - thePoint.phi()) < fabs(rhs->globalPosition().phi() - thePoint.phi()) );
        }
      GlobalPoint thePoint;
    };

    struct LessPhi {
        LessPhi() : thePoint(0,0,0) {}
        bool operator()(const MuonTransientTrackingRecHit::MuonRecHitPointer& lhs,
                       const MuonTransientTrackingRecHit::MuonRecHitPointer& rhs) const{
           return (lhs->globalPosition().phi() < rhs->globalPosition().phi());
        }
        GlobalPoint thePoint;
    };

    struct LessPerp {
        LessPerp() : thePoint(0,0,0) {}
        bool operator()(const MuonTransientTrackingRecHit::MuonRecHitPointer& lhs,
                       const MuonTransientTrackingRecHit::MuonRecHitPointer& rhs) const{
           return (lhs->globalPosition().perp() < rhs->globalPosition().perp());
        }
      GlobalPoint thePoint;
    };

    struct LessAbsMag {
        LessAbsMag() : thePoint(0,0,0) {}
        bool operator()(const MuonTransientTrackingRecHit::MuonRecHitPointer& lhs,
                       const MuonTransientTrackingRecHit::MuonRecHitPointer& rhs) const{
           return (lhs->globalPosition().mag() < rhs->globalPosition().mag());
        }
      GlobalPoint thePoint;
    };

    std::string category_;

    unsigned long long theCacheId_TRH;
    unsigned long long theCacheId_MT;

    std::string theTrackerRecHitBuilderName;
    edm::ESHandle<TransientTrackingRecHitBuilder> theTrackerRecHitBuilder;

    std::string theMuonRecHitBuilderName;
    edm::ESHandle<TransientTrackingRecHitBuilder> theMuonRecHitBuilder;

    edm::InputTag theDTRecHitLabel;
    edm::InputTag theCSCRecHitLabel;
    edm::InputTag theCSCSegmentsLabel;
    edm::InputTag theDT4DRecSegmentLabel;
    edm::Handle<DTRecHitCollection> theDTRecHits;
    edm::Handle<CSCRecHit2DCollection> theCSCRecHits;
    edm::Handle<CSCSegmentCollection> theCSCSegments;
    edm::Handle<DTRecSegment4DCollection> theDT4DRecSegments;

    // geometry
    edm::ESHandle<GeometricSearchTracker> theTracker;
    edm::ESHandle<GlobalTrackingGeometry> theTrackingGeometry;
    edm::ESHandle<MagneticField> theField;
    edm::ESHandle<CSCGeometry> theCSCGeometry;
    edm::ESHandle<DTGeometry> theDTGeometry;

};
#endif
