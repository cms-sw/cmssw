#ifndef RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H
#define RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H

/** \class MuonTrackingRegionBuilder
 *
 *  Build a TrackingRegion around a standalone muon
 *
 * Options:
 *  Beamspot : Origin is defined by primary vertex
 *  Vertex   : Origin is defined by primary vertex (first valid vertex in the VertexCollection)
 *             if no vertex is found the beamspot is used instead
 *  DynamicZError           
 *  DynamicEtaError
 *  DynamicphiError
 *
 *  \author N. Neumeister   Purdue University
 *  \author A. Everett      Purdue University
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class MuonServiceProxy;
class MeasurementTrackerEvent;

namespace edm {class ParameterSet; class Event;}

class MuonTrackingRegionBuilder : public TrackingRegionProducer {
  
  public:
 
    /// Constructor
    explicit MuonTrackingRegionBuilder(const edm::ParameterSet& par, edm::ConsumesCollector& iC) { build(par, iC); }
    explicit MuonTrackingRegionBuilder(const edm::ParameterSet& par, edm::ConsumesCollector&& iC) { build(par, iC); }

    /// Destructor
    virtual ~MuonTrackingRegionBuilder() {}

    /// Create Region of Interest
    virtual std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event&, const edm::EventSetup&) const override;
  
    /// Define tracking region
    std::unique_ptr<RectangularEtaPhiTrackingRegion> region(const reco::TrackRef&) const;
    std::unique_ptr<RectangularEtaPhiTrackingRegion> region(const reco::Track& t) const { return region(t,*theEvent); }
    std::unique_ptr<RectangularEtaPhiTrackingRegion> region(const reco::Track&, const edm::Event&) const;

    /// Pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

    /// Add Fill Descriptions
    static void fillDescriptions(edm::ParameterSetDescription& descriptions);

  private:
    
    void build(const edm::ParameterSet&, edm::ConsumesCollector&);

    const edm::Event* theEvent;

    bool useVertex;
    bool useFixedZ;
    bool useFixedPt;
    bool useFixedPhi;
    bool useFixedEta;
    bool thePrecise;

    int theMaxRegions;

    double theNsigmaEta;
    double theNsigmaPhi;
    double theNsigmaDz;
  
    double theEtaRegionPar1; 
    double theEtaRegionPar2;
    double thePhiRegionPar1;
    double thePhiRegionPar2;

    double thePtMin;
    double thePhiMin;
    double theEtaMin;
    double theDeltaR;
    double theHalfZ;
    double theDeltaPhi;
    double theDeltaEta;

    RectangularEtaPhiTrackingRegion::UseMeasurementTracker theOnDemand;
    edm::EDGetTokenT<MeasurementTrackerEvent> theMeasurementTrackerToken;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotToken;
    edm::EDGetTokenT<reco::VertexCollection> vertexCollectionToken;
    edm::EDGetTokenT<reco::TrackCollection> inputCollectionToken;
};
#endif
