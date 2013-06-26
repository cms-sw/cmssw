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
  edm::InputTag triggerFilterElectronsSrc;
  edm::InputTag triggerFilterMuonsSrc;
  edm::InputTag vertexLabel;
  edm::InputTag beamSpotLabel;
  
private:
  // ----------member data ---------------------------
  bool fIsRecoCandidate;
  bool fUseBeamSpot;
  bool fUseVertex;
  bool fUseTriggerFilterElectrons, fUseTriggerFilterMuons;
  edm::ParameterSet theConfig;
  bool fVerbose;
};
