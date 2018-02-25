#ifndef HLTScoutingTrackProducer_h
#define HLTScoutingTrackProducer_h

// -*- C++ -*-
//
// Package:    HLTrigger/Muon
// Class:      HLTScoutingTrackProducer
//
/**\class HLTScoutingTrackProducer HLTScoutingTrackProducer.h HLTScoutingTrackProducer.h

Description: Producer for Scouting Tracks

*/
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DataFormats/Scouting/interface/ScoutingMuon.h"
#include "DataFormats/Scouting/interface/ScoutingTrack.h"
#include "DataFormats/Scouting/interface/ScoutingVertex.h"

#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

class HLTScoutingTrackProducer : public edm::global::EDProducer<> {

    public:
        explicit HLTScoutingTrackProducer(const edm::ParameterSet&);
        ~HLTScoutingTrackProducer() override;

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
        void produce(edm::StreamID sid, edm::Event & iEvent, edm::EventSetup const & setup)
            const final;

        const edm::EDGetTokenT<reco::TrackCollection> OtherTrackCollection_;


};

#endif
