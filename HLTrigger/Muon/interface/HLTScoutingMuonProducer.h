#ifndef HLTScoutingMuonProducer_h
#define HLTScoutingMuonProducer_h

// -*- C++ -*-
//
// Package:    HLTrigger/Muon
// Class:      HLTScoutingMuonProducer
//
/**\class HLTScoutingMuonProducer HLTScoutingMuonProducer.h HLTrigger/Muon/interface/HLTScoutingMuonProducer.h

Description: Producer for ScoutingMuon

*/
//
// Original Author:  David G. Sheffield (Rutgers)
//         Created:  Fri, 31 Jul 2015
//
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

#include "DataFormats/Scouting/interface/ScoutingMuon.h"

class HLTScoutingMuonProducer : public edm::global::EDProducer<> {
    typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::RecoChargedCandidate>, float,
                                                unsigned int> > RecoChargedCandMap;
    public:
        explicit HLTScoutingMuonProducer(const edm::ParameterSet&);
        ~HLTScoutingMuonProducer();

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
        virtual void produce(edm::StreamID sid, edm::Event & iEvent, edm::EventSetup const & setup)
            const override final;

        const edm::EDGetTokenT<reco::RecoChargedCandidateCollection> ChargedCandidateCollection_;
        const edm::EDGetTokenT<reco::TrackCollection> TrackCollection_;
        const edm::EDGetTokenT<RecoChargedCandMap> EcalPFClusterIsoMap_;
        const edm::EDGetTokenT<RecoChargedCandMap> HcalPFClusterIsoMap_;
        const edm::EDGetTokenT<edm::ValueMap<double>> TrackIsoMap_;

        const double muonPtCut;
        const double muonEtaCut;
};

#endif
