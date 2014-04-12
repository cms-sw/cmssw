// -*- C++ -*-
//
// Package:    VertexFromTrackProducer
// Class:      VertexFromTrackProducer
// 
/**\class VertexFromTrackProducer VertexFromTrackProducer.cc RecoVertex/PrimaryVertexProducer/src/VertexFromTrackProducer.cc

 Description: produces a primary vertex extrapolating the track of a candidate on the beam axis

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andreas Hinzmann
//         Created:  Tue Dec 6 17:16:45 CET 2011
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

//
// class declaration
//

class VertexFromTrackProducer : public edm::EDProducer {
public:
  explicit VertexFromTrackProducer(const edm::ParameterSet&);
  ~VertexFromTrackProducer();
  
  virtual void produce(edm::Event&, const edm::EventSetup&) override;

  // access to config
  edm::ParameterSet config() const { return theConfig; }
  edm::InputTag trackLabel;
  edm::EDGetTokenT<edm::View<reco::Track> > trackToken;
  edm::EDGetTokenT<reco::RecoCandidate> candidateToken;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> triggerFilterElectronsSrc;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> triggerFilterMuonsSrc;
  edm::EDGetTokenT<edm::View<reco::Vertex> > vertexLabel;
  edm::EDGetTokenT<reco::BeamSpot> beamSpotLabel;
  
private:
  // ----------member data ---------------------------
  bool fIsRecoCandidate;
  bool fUseBeamSpot;
  bool fUseVertex;
  bool fUseTriggerFilterElectrons, fUseTriggerFilterMuons;
  edm::ParameterSet theConfig;
  bool fVerbose;
};
