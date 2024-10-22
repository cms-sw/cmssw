#ifndef RecoMuon_GlobalTrackingTools_MuonTrackingRegionByPtBuilder_h
#define RecoMuon_GlobalTrackingTools_MuonTrackingRegionByPtBuilder_h

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
class MagneticField;
class IdealMagneticFieldRecord;
class MultipleScatteringParametrisationMaker;
class TrackerMultipleScatteringRecord;

class MuonTrackingRegionByPtBuilder : public TrackingRegionProducer {
public:
  /// Constructor
  explicit MuonTrackingRegionByPtBuilder(const edm::ParameterSet& par, edm::ConsumesCollector& iC) { build(par, iC); }
  explicit MuonTrackingRegionByPtBuilder(const edm::ParameterSet& par, edm::ConsumesCollector&& iC) { build(par, iC); }

  /// Destructor
  ~MuonTrackingRegionByPtBuilder() override = default;

  /// Create Region of Interest
  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event&, const edm::EventSetup&) const override;

  /// Define tracking region
  std::unique_ptr<RectangularEtaPhiTrackingRegion> region(const reco::TrackRef&) const;
  std::unique_ptr<RectangularEtaPhiTrackingRegion> region(const reco::Track& t) const {
    return region(t, *theEvent, *theEventSetup);
  }
  std::unique_ptr<RectangularEtaPhiTrackingRegion> region(const reco::Track&,
                                                          const edm::Event&,
                                                          const edm::EventSetup&) const;

  /// Pass the Event to the algo at each event
  void setEvent(const edm::Event&, const edm::EventSetup&);

  /// Add Fill Descriptions
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void build(const edm::ParameterSet&, edm::ConsumesCollector&);

  const edm::Event* theEvent;
  const edm::EventSetup* theEventSetup;

  bool useVertex;
  bool useFixedZ;
  bool useFixedPt;
  bool thePrecise;

  int theMaxRegions;

  double theNsigmaDz;

  double thePtMin;
  double theDeltaR;
  double theHalfZ;

  std::vector<double> ptRanges_;
  std::vector<double> deltaEtas_;
  std::vector<double> deltaPhis_;

  RectangularEtaPhiTrackingRegion::UseMeasurementTracker theOnDemand;
  edm::EDGetTokenT<MeasurementTrackerEvent> theMeasurementTrackerToken;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotToken;
  edm::EDGetTokenT<reco::VertexCollection> vertexCollectionToken;
  edm::EDGetTokenT<reco::TrackCollection> inputCollectionToken;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> bfieldToken;
  edm::ESGetToken<MultipleScatteringParametrisationMaker, TrackerMultipleScatteringRecord> msmakerToken;
};
#endif
