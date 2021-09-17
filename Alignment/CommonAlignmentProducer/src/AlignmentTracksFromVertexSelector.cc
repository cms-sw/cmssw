#include "Alignment/CommonAlignmentProducer/interface/AlignmentTracksFromVertexSelector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

// vertices
#include "RecoVertex/VertexTools/interface/VertexDistanceXY.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"

// constructor ----------------------------------------------------------------
AlignmentTrackFromVertexSelector::AlignmentTrackFromVertexSelector(const edm::ParameterSet& cfg,
                                                                   edm::ConsumesCollector& iC)
    : ttbESToken_(
          iC.esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))),
      vertexToken_(iC.consumes<reco::VertexCollection>(cfg.getParameter<edm::InputTag>("vertices"))),
      diLeptonToken_(iC.consumes<reco::TrackCollection>(cfg.getParameter<edm::InputTag>("leptonTracks"))),
      useClosestVertex_(cfg.getParameter<bool>("useClosestVertexToDilepton")),
      vertexIndex_(cfg.getParameter<unsigned int>("vertexIndex")) {}

// destructor -----------------------------------------------------------------
AlignmentTrackFromVertexSelector::~AlignmentTrackFromVertexSelector() {}

// compute the closest vertex to di-lepton ------------------------------------
const reco::Vertex* AlignmentTrackFromVertexSelector::findClosestVertex(const reco::TrackCollection& leptonTracks,
                                                                        const reco::VertexCollection* vertices,
                                                                        const edm::EventSetup& setup) const {
  reco::Vertex* defaultVtx = nullptr;

  // fill the transient track collection with the lepton tracks
  const TransientTrackBuilder* theB = &setup.getData(ttbESToken_);
  std::vector<reco::TransientTrack> tks;
  for (const auto& track : leptonTracks) {
    reco::TransientTrack trajectory = theB->build(track);
    tks.push_back(trajectory);
  }

  // compute the secondary vertex
  TransientVertex aTransVtx;
  KalmanVertexFitter kalman(true);
  aTransVtx = kalman.vertex(tks);

  if (!aTransVtx.isValid())
    return defaultVtx;

  // find the closest vertex to the secondary vertex in 3D
  VertexDistance3D vertTool3D;
  float minD = 9999.;
  int closestVtxIndex = 0;
  int counter = 0;
  for (const auto& vtx : *vertices) {
    double dist3D = vertTool3D.distance(aTransVtx, vtx).value();
    if (dist3D < minD) {
      minD = dist3D;
      closestVtxIndex = counter;
    }
    counter++;
  }
  if ((*vertices).at(closestVtxIndex).isValid()) {
    return &(vertices->at(closestVtxIndex));
  } else {
    return defaultVtx;
  }
}

// do selection ---------------------------------------------------------------
AlignmentTrackFromVertexSelector::Tracks AlignmentTrackFromVertexSelector::select(
    const edm::Handle<reco::TrackCollection>& tc, const edm::Event& evt, const edm::EventSetup& setup) const {
  Tracks result;

  std::vector<unsigned int> thePVkeys;

  // get collection of reconstructed vertices from event
  edm::Handle<reco::VertexCollection> vertexHandle = evt.getHandle(vertexToken_);

  // get collection of the di-lepton traxks
  const auto& leptonTracks = evt.get(diLeptonToken_);

  // fill the vector of keys
  if (vertexHandle.isValid()) {
    const reco::VertexCollection* vertices = vertexHandle.product();
    const reco::Vertex* theVtx = nullptr;

    if (useClosestVertex_) {
      theVtx = findClosestVertex(leptonTracks, vertices, setup);
    } else {
      if ((*vertices).at(vertexIndex_).isValid()) {
        theVtx = &(vertices->at(vertexIndex_));
      }
    }

    if (theVtx) {
      for (auto tv = theVtx->tracks_begin(); tv != theVtx->tracks_end(); tv++) {
        if (tv->isNonnull()) {
          const reco::TrackRef trackRef = tv->castTo<reco::TrackRef>();
          thePVkeys.push_back(trackRef.key());
        }
      }
    }
  }

  std::sort(thePVkeys.begin(), thePVkeys.end());

  LogDebug("AlignmentTrackFromVertexSelector")
      << "after collecting the PV keys: thePVkeys.size()" << thePVkeys.size() << std::endl;
  for (const auto& key : thePVkeys) {
    LogDebug("AlignmentTrackFromVertexSelector") << key << ", ";
  }
  LogDebug("AlignmentTrackFromVertexSelector") << std::endl;

  if (tc.isValid()) {
    int indx(0);
    // put the track in the collection is it was used for the vertex
    for (reco::TrackCollection::const_iterator tk = tc->begin(); tk != tc->end(); ++tk, ++indx) {
      reco::TrackRef trackRef = reco::TrackRef(tc, indx);
      if (std::find(thePVkeys.begin(), thePVkeys.end(), trackRef.key()) != thePVkeys.end()) {
        LogDebug("AlignmentTrackFromVertexSelector") << "track index: " << indx << "filling result vector" << std::endl;
        result.push_back(&(*tk));
      }  // if a valid key is found
    }    // end loop over tracks
  }      // if the handle is valid
  return result;
}
