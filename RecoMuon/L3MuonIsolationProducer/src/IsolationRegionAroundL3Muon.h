#ifndef RecoMuon_L3MuonIsolationProducer_IsolationRegionAroundL3Muon_H 
#define RecoMuon_L3MuonIsolationProducer_IsolationRegionAroundL3Muon_H 

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"


class IsolationRegionAroundL3Muon : public TrackingRegionProducer {

public:

  IsolationRegionAroundL3Muon(const edm::ParameterSet& cfg,
	edm::ConsumesCollector && iC) { 

    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

    theVertexSrc   = regionPSet.getParameter<edm::InputTag>("vertexSrc");
    if (theVertexSrc.label().length()>1) theVertexToken   = iC.consumes<reco::VertexCollection>(theVertexSrc);
    theInputTrkToken = iC.consumes<reco::TrackCollection>(regionPSet.getParameter<edm::InputTag>("TrkSrc"));

    thePtMin              = regionPSet.getParameter<double>("ptMin");
    theOriginRadius       = regionPSet.getParameter<double>("originRadius");
    theOriginHalfLength   = regionPSet.getParameter<double>("originHalfLength");
    theVertexZconstrained = regionPSet.getParameter<bool>("vertexZConstrained");
    theOriginZPos  = regionPSet.getParameter<double>("vertexZDefault");

    theDeltaEta = regionPSet.getParameter<double>("deltaEtaRegion");
    theDeltaPhi =  regionPSet.getParameter<double>("deltaPhiRegion");
    theMeasurementTrackerToken = iC.consumes<MeasurementTrackerEvent>(regionPSet.getParameter<std::string>("measurementTrackerName"));
  }   

  virtual ~IsolationRegionAroundL3Muon(){}

  virtual std::vector<TrackingRegion* > regions(const edm::Event& ev, 
      const edm::EventSetup& es) const {

    std::vector<TrackingRegion* > result;

    // optional constraint for vertex
    // get highest Pt pixel vertex (if existing)
    double deltaZVertex =  theOriginHalfLength;
    double originz = theOriginZPos;
    if (theVertexSrc.label().length()>1) {
      edm::Handle<reco::VertexCollection> vertices;
      ev.getByToken(theVertexToken,vertices);
      const reco::VertexCollection vertCollection = *(vertices.product());
      reco::VertexCollection::const_iterator ci = vertCollection.begin();
      if (vertCollection.size()>0) {
            originz = ci->z();
      } else {
            originz = theOriginZPos;
            deltaZVertex = 15.;
      }
    }

    edm::Handle<reco::TrackCollection> trks;
    ev.getByToken(theInputTrkToken, trks);

    edm::Handle<MeasurementTrackerEvent> hmte;
    ev.getByToken(theMeasurementTrackerToken, hmte);
    const MeasurementTrackerEvent *measurementTrackerEvent = hmte.product();

    for(reco::TrackCollection::const_iterator iTrk = trks->begin();iTrk != trks->end();iTrk++) {
      double vz = (theVertexZconstrained) ? iTrk->dz() : originz;
      GlobalVector dirVector((iTrk)->px(),(iTrk)->py(),(iTrk)->pz());
      result.push_back( 
          new RectangularEtaPhiTrackingRegion( dirVector, GlobalPoint(0,0,float(vz)), 
					       thePtMin, theOriginRadius, deltaZVertex, theDeltaEta, theDeltaPhi,
					       RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kForSiStrips,
                                               true,measurementTrackerEvent) );
    }

    return result;
  }

private:

  edm::InputTag theVertexSrc;
  edm::EDGetTokenT<reco::VertexCollection> theVertexToken;
  edm::EDGetTokenT<reco::TrackCollection> theInputTrkToken;

  double thePtMin; 
  double theOriginRadius; 
  double theOriginHalfLength; 
  bool   theVertexZconstrained;
  double theOriginZPos;

  double theDeltaEta; 
  double theDeltaPhi;
  edm::EDGetTokenT<MeasurementTrackerEvent> theMeasurementTrackerToken;
};

#endif 

