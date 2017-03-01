#ifndef HLTScoutingEgammaProducer_h
#define HLTScoutingEgammaProducer_h

// -*- C++ -*-
//
// Package:    HLTrigger/Egamma
// Class:      HLTScoutingEgammaProducer
//
/**\class HLTScoutingEgammaProducer HLTScoutingEgammaProducer.h HLTrigger/Egamma/interface/HLTScoutingEgammaProducer.h

Description: Producer for ScoutingElectron and ScoutingPhoton

*/
//
// Original Author:  David G. Sheffield (Rutgers)
//         Created:  Mon, 20 Jul 2015
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
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "DataFormats/Scouting/interface/ScoutingElectron.h"
#include "DataFormats/Scouting/interface/ScoutingPhoton.h"

class HLTScoutingEgammaProducer : public edm::global::EDProducer<> {
    typedef edm::AssociationMap<edm::OneToValue<std::vector<reco::RecoEcalCandidate>, float,
                                                unsigned int> > RecoEcalCandMap;
    public:
        explicit HLTScoutingEgammaProducer(const edm::ParameterSet&);
        ~HLTScoutingEgammaProducer();

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
        virtual void produce(edm::StreamID sid, edm::Event & iEvent, edm::EventSetup const & setup)
            const override final;

        const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> EgammaCandidateCollection_;
        const edm::EDGetTokenT<reco::GsfTrackCollection> EgammaGsfTrackCollection_;
        const edm::EDGetTokenT<RecoEcalCandMap> SigmaIEtaIEtaMap_;
        const edm::EDGetTokenT<RecoEcalCandMap> HoverEMap_;
        const edm::EDGetTokenT<RecoEcalCandMap> DetaMap_;
        const edm::EDGetTokenT<RecoEcalCandMap> DphiMap_;
        const edm::EDGetTokenT<RecoEcalCandMap> MissingHitsMap_;
        const edm::EDGetTokenT<RecoEcalCandMap> OneOEMinusOneOPMap_;
        const edm::EDGetTokenT<RecoEcalCandMap> EcalPFClusterIsoMap_;
        const edm::EDGetTokenT<RecoEcalCandMap> EleGsfTrackIsoMap_;
        const edm::EDGetTokenT<RecoEcalCandMap> HcalPFClusterIsoMap_;

        const double egammaPtCut;
        const double egammaEtaCut;
        const double egammaHoverECut;
};

#endif
