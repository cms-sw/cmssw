#ifndef _StripSubClusterShapeTrajectoryFilter_h_
#define _StripSubClusterShapeTrajectoryFilter_h_

#include <vector>
#include <unordered_map>
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkSeedingLayers/interface/SeedComparitor.h"

class ClusterShapeHitFilter;
class TrackerTopology;
class TrackerGeometry;
class TrajectoryMeasurement;
class TrajectoryStateOnSurface;
class MeasurementTrackerEvent;
class SiStripNoises;
class TTree;
namespace edm { class Event; class EventSetup; class ConsumesCollector; }

//#define StripSubClusterShapeFilterBase_COUNTERS


class StripSubClusterShapeFilterBase {
    public:
        StripSubClusterShapeFilterBase(const edm::ParameterSet &iConfig, edm::ConsumesCollector& iC);
        virtual ~StripSubClusterShapeFilterBase();

    protected:

        void setEventBase(const edm::Event &, const edm::EventSetup &) ;

        bool testLastHit(const TrackingRecHit *hit, const TrajectoryStateOnSurface &tsos, bool mustProject=false) const ;
        bool testLastHit(const TrackingRecHit *hit, const GlobalPoint &gpos, const GlobalVector &gdir, bool mustProject=false) const ;

        // who am i
        std::string label_;

        // pass-through of clusters with too many consecutive saturated strips
        uint32_t maxNSat_;

        // trimming parameters
        uint8_t trimMaxADC_;
        float   trimMaxFracTotal_, trimMaxFracNeigh_;

        // maximum difference after peak finding
        float   maxTrimmedSizeDiffPos_, maxTrimmedSizeDiffNeg_;

        // peak finding parameters
        float subclusterWindow_;
        float seedCutMIPs_, seedCutSN_;
        float subclusterCutMIPs_, subclusterCutSN_;

        // layers in which to apply the filter 
        std::array<std::array<uint8_t,10>, 7> layerMask_;

#ifdef StripSubClusterShapeFilterBase_COUNTERS
        mutable uint64_t called_, saturated_, test_, passTrim_, failTooLarge_, passSC_, failTooNarrow_;
#endif

        edm::ESHandle<TrackerGeometry> theTracker;
        edm::ESHandle<ClusterShapeHitFilter> theFilter;
        edm::ESHandle<SiStripNoises>  theNoise;
        edm::ESHandle<TrackerTopology> theTopology;
};

class StripSubClusterShapeTrajectoryFilter: public StripSubClusterShapeFilterBase, public TrajectoryFilter {
    public:
        StripSubClusterShapeTrajectoryFilter(const edm::ParameterSet &iConfig, edm::ConsumesCollector& iC):
            StripSubClusterShapeFilterBase(iConfig,iC) {}

        virtual ~StripSubClusterShapeTrajectoryFilter() {}

        virtual bool qualityFilter(const TempTrajectory&) const override;
        virtual bool qualityFilter(const Trajectory&) const override;

        virtual bool toBeContinued(TempTrajectory&) const override;
        virtual bool toBeContinued(Trajectory&) const override;

        virtual std::string name() const override { return "StripSubClusterShapeTrajectoryFilter"; }

        virtual void setEvent(const edm::Event & e, const edm::EventSetup & es) override {
            setEventBase(e,es);
        }

    protected:
        using StripSubClusterShapeFilterBase::testLastHit;
        bool testLastHit(const TrajectoryMeasurement &last) const ;
};

class StripSubClusterShapeSeedFilter: public StripSubClusterShapeFilterBase, public SeedComparitor {
    public:
        StripSubClusterShapeSeedFilter(const edm::ParameterSet &iConfig, edm::ConsumesCollector& iC) ;

        virtual ~StripSubClusterShapeSeedFilter() {}

        virtual void init(const edm::Event& ev, const edm::EventSetup& es) override {
            setEventBase(ev,es);
        }
        // implemented
        virtual bool compatible(const TrajectoryStateOnSurface &tsos,  SeedingHitSet::ConstRecHitPointer hit) const override ;
        // not implemented 
        virtual bool compatible(const SeedingHitSet &hits) const override { return true; }
        virtual bool compatible(const SeedingHitSet &hits, const GlobalTrajectoryParameters &helixStateAtVertex, const FastHelix &helix) const override ;

    protected:
        bool filterAtHelixStage_;
};



#endif
