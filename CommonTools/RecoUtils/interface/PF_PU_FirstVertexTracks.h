#ifndef PF_PU_FirstVertexTracks_h
#define PF_PU_FirstVertexTracks_h

// -*- C++ -*-
//
// Package:    PF_PU_AssoMap
// Class:      PF_PU_FirstVertexTracks
//
/**\class PF_PU_AssoMap PF_PU_FirstVertexTracks.cc CommonTools/RecoUtils/plugins/PF_PU_FirstVertexTracks.cc

  Description: Produces collection of tracks associated to the first vertex based on the pf_pu Association Map
*/
//

// Original Author:  Matthias Geisler
//         Created:  Wed Apr 18 14:48:37 CEST 2012
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::VertexCollection, reco::TrackCollection, int> >
    TrackToVertexAssMap;
typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::TrackCollection, reco::VertexCollection, int> >
    VertexToTrackAssMap;

//
// class declaration
//

class PF_PU_FirstVertexTracks : public edm::global::EDProducer<> {
public:
  explicit PF_PU_FirstVertexTracks(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  bool trackMatch(const reco::Track&, const reco::Track&) const;

  // ----------member data ---------------------------

  edm::InputTag input_AssociationType_;

  edm::EDGetTokenT<TrackToVertexAssMap> token_TrackToVertexAssMap_;
  edm::EDGetTokenT<VertexToTrackAssMap> token_VertexToTrackAssMap_;
  edm::EDGetTokenT<reco::TrackCollection> token_generalTracksCollection_;
  edm::EDGetTokenT<reco::VertexCollection> token_VertexCollection_;

  int input_MinQuality_;
};

#endif
