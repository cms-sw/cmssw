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
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

//
// class declaration
//

class VertexFromTrackProducer : public edm::global::EDProducer<> {
public:
  explicit VertexFromTrackProducer(const edm::ParameterSet&);
  ~VertexFromTrackProducer();
  
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // tokens
  const edm::EDGetTokenT<edm::View<reco::Track> > trackToken;
  const edm::EDGetTokenT<edm::View<reco::RecoCandidate> > candidateToken;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> triggerFilterElectronsSrc;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> triggerFilterMuonsSrc;
  const edm::EDGetTokenT<edm::View<reco::Vertex> > vertexLabel;
  const edm::EDGetTokenT<reco::BeamSpot> beamSpotLabel;
  
  // ----------member data ---------------------------
  const bool fIsRecoCandidate;
  const bool fUseBeamSpot;
  const bool fUseVertex;
  const bool fUseTriggerFilterElectrons, fUseTriggerFilterMuons;
  const bool fVerbose;
};
