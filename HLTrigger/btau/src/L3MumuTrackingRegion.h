#ifndef HLTrigger_btau_L3MumuTrackingRegion_H 
#define HLTrigger_btau_L3MumuTrackingRegion_H 

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

class L3MumuTrackingRegion : public TrackingRegionProducer {

public:

  L3MumuTrackingRegion(const edm::ParameterSet& cfg, edm::ConsumesCollector && iC) { 

    edm::ParameterSet regionPSet = cfg.getParameter<edm::ParameterSet>("RegionPSet");

    theVertexTag    = regionPSet.getParameter<edm::InputTag>("vertexSrc");
    theVertex       = (theVertexTag.label().length()>1);
    theInputTrkTag  = regionPSet.getParameter<edm::InputTag>("TrkSrc");
    useVtxTks = regionPSet.getParameter<bool>("UseVtxTks");

    if (theVertex) theVertexToken  = iC.consumes<reco::VertexCollection>(theVertexTag);
    if (!(theVertex && useVtxTks)) theInputTrkToken= iC.consumes<reco::TrackCollection>(theInputTrkTag);

    thePtMin              = regionPSet.getParameter<double>("ptMin");
    theOriginRadius       = regionPSet.getParameter<double>("originRadius");
    theOriginHalfLength   = regionPSet.getParameter<double>("originHalfLength");
    theOriginZPos  = regionPSet.getParameter<double>("vertexZDefault");

    theDeltaEta = regionPSet.getParameter<double>("deltaEtaRegion");
    theDeltaPhi =  regionPSet.getParameter<double>("deltaPhiRegion");
    if (regionPSet.exists("searchOpt")){
      m_searchOpt    = regionPSet.getParameter<bool>("searchOpt");
    }
    else{
      m_searchOpt = false;
    }
    m_howToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kForSiStrips;
    if (regionPSet.exists("measurementTrackerName")){
      // FIXME: when next time altering the configuration of this
      // class, please change the types of the following parameters:
      // - howToUseMeasurementTracker to at least int32 or to a string
      //   corresponding to the UseMeasurementTracker enumeration
      // - measurementTrackerName to InputTag
      if (regionPSet.exists("howToUseMeasurementTracker")){
	m_howToUseMeasurementTracker = RectangularEtaPhiTrackingRegion::doubleToUseMeasurementTracker(regionPSet.getParameter<double>("howToUseMeasurementTracker"));
      }
      if(m_howToUseMeasurementTracker != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
        theMeasurementTrackerToken = iC.consumes<MeasurementTrackerEvent>(regionPSet.getParameter<std::string>("measurementTrackerName"));
      }
    }
  }   

  virtual ~L3MumuTrackingRegion(){}

  virtual std::vector<TrackingRegion* > regions(const edm::Event& ev, 
      const edm::EventSetup& es) const {

    std::vector<TrackingRegion* > result;

    const MeasurementTrackerEvent *measurementTracker = nullptr;
    if(!theMeasurementTrackerToken.isUninitialized()) {
      edm::Handle<MeasurementTrackerEvent> hmte;
      ev.getByToken(theMeasurementTrackerToken, hmte);
      measurementTracker = hmte.product();
    }

    // optional constraint for vertex
    // get highest Pt pixel vertex (if existing)
    double deltaZVertex =  theOriginHalfLength;
    double originz = theOriginZPos;
    if (theVertex) {
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
      if (useVtxTks) {
	for(ci=vertCollection.begin();ci!=vertCollection.end();ci++)
          for (reco::Vertex::trackRef_iterator trackIt =  ci->tracks_begin();trackIt !=  ci->tracks_end();trackIt++){
	    reco::TrackRef iTrk =  (*trackIt).castTo<reco::TrackRef>() ;
            GlobalVector dirVector((iTrk)->px(),(iTrk)->py(),(iTrk)->pz());
            result.push_back(
                             new RectangularEtaPhiTrackingRegion( dirVector, GlobalPoint(0,0,float(ci->z())),
                                                                  thePtMin, theOriginRadius, deltaZVertex, theDeltaEta, theDeltaPhi,
								  m_howToUseMeasurementTracker,
								  true,
								  measurementTracker,
								  m_searchOpt) );
          }
        return result;
      }
    }

    edm::Handle<reco::TrackCollection> trks;
    if (!theInputTrkToken.isUninitialized()) ev.getByToken(theInputTrkToken, trks);
    for(reco::TrackCollection::const_iterator iTrk = trks->begin();iTrk != trks->end();iTrk++) {
      GlobalVector dirVector((iTrk)->px(),(iTrk)->py(),(iTrk)->pz());
      result.push_back( 
	  new RectangularEtaPhiTrackingRegion( dirVector, GlobalPoint(0,0,float(originz)), 
					       thePtMin, theOriginRadius, deltaZVertex, theDeltaEta, theDeltaPhi,
					       m_howToUseMeasurementTracker,
					       true,
					       measurementTracker,
					       m_searchOpt) );
    }
    return result;
  }

private:

  edm::InputTag                            theVertexTag;
  bool                                     theVertex;
  edm::EDGetTokenT<reco::VertexCollection> theVertexToken;
  edm::InputTag                            theInputTrkTag;
  edm::EDGetTokenT<reco::TrackCollection>  theInputTrkToken;

  bool useVtxTks;

  double thePtMin; 
  double theOriginRadius; 
  double theOriginHalfLength; 
  double theOriginZPos;

  double theDeltaEta; 
  double theDeltaPhi;
  edm::EDGetTokenT<MeasurementTrackerEvent> theMeasurementTrackerToken;
  RectangularEtaPhiTrackingRegion::UseMeasurementTracker m_howToUseMeasurementTracker;
  bool m_searchOpt;
};

#endif 

