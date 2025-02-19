//
// $Id: LeptonVertexSignificance.cc,v 1.3 2010/10/15 22:44:33 wmtan Exp $
//

#include "PhysicsTools/PatUtils/interface/LeptonVertexSignificance.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h" 
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

using namespace pat;

// constructor
LeptonVertexSignificance::LeptonVertexSignificance(const edm::EventSetup & iSetup) {
  // instantiate the transient-track builder
  edm::ESHandle<TransientTrackBuilder> builder;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  theTrackBuilder_ = new TransientTrackBuilder(*builder.product());
}

// destructor
LeptonVertexSignificance::~LeptonVertexSignificance() {
  delete theTrackBuilder_;
}

// calculate the TrackIsoPt for the lepton object
float LeptonVertexSignificance::calculate(const Electron & theElectron, const edm::Event & iEvent) {
  return this->calculate(*theElectron.gsfTrack(), iEvent);
}

float LeptonVertexSignificance::calculate(const Muon & theMuon, const edm::Event & iEvent) {
  return this->calculate(*theMuon.track(), iEvent);
}

// calculate the TrackIsoPt for the lepton's track
float LeptonVertexSignificance::calculate(const reco::Track & theTrack, const edm::Event & iEvent) {
  // FIXME: think more about how to handle events without vertices
  // lepton LR calculation should have nothing to do with event selection
  edm::Handle<reco::VertexCollection> vertexHandle;
  iEvent.getByLabel("offlinePrimaryVerticesFromCTFTracks", vertexHandle);
  if (vertexHandle.product()->size() == 0) return 0;
  reco::Vertex theVertex = vertexHandle.product()->front();
  // calculate the track-vertex association significance
  reco::TransientTrack theTrTrack = theTrackBuilder_->build(&theTrack);
  GlobalPoint theVertexPoint(theVertex.position().x(), theVertex.position().y(), theVertex.position().z());
  FreeTrajectoryState theLeptonNearVertex = theTrTrack.trajectoryStateClosestToPoint(theVertexPoint).theState();
  return fabs(theVertex.position().z() - theLeptonNearVertex.position().z())
    / sqrt(std::pow(theVertex.zError(), 2) + theLeptonNearVertex.cartesianError().position().czz());
}

