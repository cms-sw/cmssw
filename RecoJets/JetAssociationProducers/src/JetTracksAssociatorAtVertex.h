// \class JetTracksAssociatorAtVertex JetTracksAssociatorAtVertex.cc
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// Accommodated for Jet Package by: Fedor Ratnikov Jul. 30, 2007
//
//
#ifndef JetTracksAssociatorAtVertex_h
#define JetTracksAssociatorAtVertex_h

#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertex.h"
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRVertexAssigned.h"

class JetTracksAssociatorAtVertex : public edm::stream::EDProducer<> {
public:
  JetTracksAssociatorAtVertex(const edm::ParameterSet&);
  ~JetTracksAssociatorAtVertex() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<edm::View<reco::Jet>> mJets;
  const edm::EDGetTokenT<reco::TrackCollection> mTracks;

  const JetTracksAssociationDRVertex mAssociator;
  const JetTracksAssociationDRVertexAssigned mAssociatorAssigned;
  const bool useAssigned;  /// if true, use the track/jet association with vertex assignment to tracks
  edm::EDGetTokenT<reco::VertexCollection> pvSrc;  /// if useAssigned, will read this PV collection.
};

#endif
