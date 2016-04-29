// -*- C++ -*-
//
// Package:    HLTrigger/Egamma
// Class:      HLTScoutingEgammaProducer
//
/**\class HLTScoutingEgammaProducer HLTScoutingEgammaProducer.cc HLTrigger/Egamma/src/HLTScoutingEgammaProducer.cc

Description: Producer for ScoutingElectron and ScoutingPhoton

*/
//
// Original Author:  David G. Sheffield (Rutgers)
//         Created:  Mon, 20 Jul 2015
//
//

#include "HLTrigger/Egamma/interface/HLTScoutingEgammaProducer.h"

//
// constructors and destructor
//
HLTScoutingEgammaProducer::HLTScoutingEgammaProducer(const edm::ParameterSet& iConfig):
    EgammaCandidateCollection_(consumes<reco::RecoEcalCandidateCollection>
                               (iConfig.getParameter<edm::InputTag>("EgammaCandidates"))),
    EgammaGsfTrackCollection_(consumes<reco::GsfTrackCollection>
                              (iConfig.getParameter<edm::InputTag>("EgammaGsfTracks"))),
    SigmaIEtaIEtaMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>(
                                                    "SigmaIEtaIEtaMap"))),
    HoverEMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("HoverEMap"))),
    DetaMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("DetaMap"))),
    DphiMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("DphiMap"))),
    MissingHitsMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>(
                                                  "MissingHitsMap"))),
    OneOEMinusOneOPMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>(
                                                      "OneOEMinusOneOPMap"))),
    EcalPFClusterIsoMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>(
                                                       "EcalPFClusterIsoMap"))),
    EleGsfTrackIsoMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>(
                                                     "EleGsfTrackIsoMap"))),
    HcalPFClusterIsoMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>(
                                                       "HcalPFClusterIsoMap"))),
    egammaPtCut(iConfig.getParameter<double>("egammaPtCut")),
    egammaEtaCut(iConfig.getParameter<double>("egammaEtaCut")),
    egammaHoverECut(iConfig.getParameter<double>("egammaHoverECut"))
{
    //register products
    produces<ScoutingElectronCollection>();
    produces<ScoutingPhotonCollection>();
}

HLTScoutingEgammaProducer::~HLTScoutingEgammaProducer()
{ }

// ------------ method called to produce the data  ------------
void HLTScoutingEgammaProducer::produce(edm::StreamID sid, edm::Event & iEvent, edm::EventSetup const & setup) const
{
    using namespace edm;

    std::unique_ptr<ScoutingElectronCollection> outElectrons(new ScoutingElectronCollection());
    std::unique_ptr<ScoutingPhotonCollection> outPhotons(new ScoutingPhotonCollection());

    // Get RecoEcalCandidate
    Handle<reco::RecoEcalCandidateCollection> EgammaCandidateCollection;
    if(!iEvent.getByToken(EgammaCandidateCollection_,
                          EgammaCandidateCollection)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get GsfTrack
    Handle<reco::GsfTrackCollection> EgammaGsfTrackCollection;
    if(!iEvent.getByToken(EgammaGsfTrackCollection_,
                          EgammaGsfTrackCollection)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get SigmaIEtaIEtaMap
    Handle<RecoEcalCandMap> SigmaIEtaIEtaMap;
    if(!iEvent.getByToken(SigmaIEtaIEtaMap_, SigmaIEtaIEtaMap)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get HoverEMap
    Handle<RecoEcalCandMap> HoverEMap;
    if(!iEvent.getByToken(HoverEMap_, HoverEMap)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get DetaMap
    Handle<RecoEcalCandMap> DetaMap;
    if(!iEvent.getByToken(DetaMap_, DetaMap)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get DphiMap
    Handle<RecoEcalCandMap> DphiMap;
    if(!iEvent.getByToken(DphiMap_, DphiMap)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get MissingHitsMap
    Handle<RecoEcalCandMap> MissingHitsMap;
    if(!iEvent.getByToken(MissingHitsMap_, MissingHitsMap)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get 1/E - 1/p Map
    Handle<RecoEcalCandMap> OneOEMinusOneOPMap;
    if(!iEvent.getByToken(OneOEMinusOneOPMap_, OneOEMinusOneOPMap)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get EcalPFClusterIsoMap
    Handle<RecoEcalCandMap> EcalPFClusterIsoMap;
    if(!iEvent.getByToken(EcalPFClusterIsoMap_, EcalPFClusterIsoMap)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get EleGsfTrackIsoMap
    Handle<RecoEcalCandMap> EleGsfTrackIsoMap;
    if(!iEvent.getByToken(EleGsfTrackIsoMap_, EleGsfTrackIsoMap)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Get HcalPFClusterIsoMap
    Handle<RecoEcalCandMap> HcalPFClusterIsoMap;
    if(!iEvent.getByToken(HcalPFClusterIsoMap_, HcalPFClusterIsoMap)){
        iEvent.put(std::move(outElectrons));
        iEvent.put(std::move(outPhotons));
        return;
    }

    // Produce electrons and photons
    int index = 0;
    for (auto &candidate : *EgammaCandidateCollection) {
        reco::RecoEcalCandidateRef candidateRef = getRef(EgammaCandidateCollection, index);
        ++index;
        if (candidateRef.isNull() && !candidateRef.isAvailable())
            continue;

        if ((*HoverEMap)[candidateRef] > egammaHoverECut)
            continue;
        if (candidate.pt() < egammaPtCut)
            continue;
        if (fabs(candidate.eta()) > egammaEtaCut)
            continue;

        reco::SuperClusterRef scRef = candidate.superCluster();
        if (scRef.isNull() && !scRef.isAvailable())
            continue;
        float d0 = 0.0;
        float dz = 0.0;
        int charge = -999;
        for (auto &track: *EgammaGsfTrackCollection) {
            RefToBase<TrajectorySeed> seed = track.extra()->seedRef();
            reco::ElectronSeedRef elseed = seed.castTo<reco::ElectronSeedRef>();
            RefToBase<reco::CaloCluster> caloCluster = elseed->caloCluster();
            reco::SuperClusterRef scRefFromTrk = caloCluster.castTo<reco::SuperClusterRef>() ;
            if (scRefFromTrk == scRef) {
                d0 = track.d0();
                dz = track.dz();
                charge = track.charge();
            }
        }
        if (charge == -999) { // No associated track. Candidate is a scouting photon
            outPhotons->emplace_back(candidate.pt(), candidate.eta(), candidate.phi(),
                                     candidate.mass(), (*SigmaIEtaIEtaMap)[candidateRef],
                                     (*HoverEMap)[candidateRef],
                                     (*EcalPFClusterIsoMap)[candidateRef],
                                     (*HcalPFClusterIsoMap)[candidateRef]);
        } else { // Candidate is a scouting electron
            outElectrons->emplace_back(candidate.pt(), candidate.eta(), candidate.phi(),
                                       candidate.mass(), d0, dz, (*DetaMap)[candidateRef],
                                       (*DphiMap)[candidateRef], (*SigmaIEtaIEtaMap)[candidateRef],
                                       (*HoverEMap)[candidateRef],
                                       (*OneOEMinusOneOPMap)[candidateRef],
                                       (*MissingHitsMap)[candidateRef], charge,
                                       (*EcalPFClusterIsoMap)[candidateRef],
                                       (*HcalPFClusterIsoMap)[candidateRef],
                                       (*EleGsfTrackIsoMap)[candidateRef]);
        }
    }

    // Put output
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HLTScoutingEgammaProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("EgammaCandidates", edm::InputTag("hltEgammaCandidates"));
    desc.add<edm::InputTag>("EgammaGsfTracks", edm::InputTag("hltEgammaGsfTracks"));
    desc.add<edm::InputTag>("SigmaIEtaIEtaMap", edm::InputTag(
                                "hltEgammaClusterShape:sigmaIEtaIEta5x5"));
    desc.add<edm::InputTag>("HoverEMap", edm::InputTag("hltEgammaHoverE"));
    desc.add<edm::InputTag>("DetaMap", edm::InputTag("hltEgammaGsfTrackVars:Deta"));
    desc.add<edm::InputTag>("DphiMap", edm::InputTag("hltEgammaGsfTrackVars:Dphi"));
    desc.add<edm::InputTag>("MissingHitsMap", edm::InputTag("hltEgammaGsfTrackVars:MissingHits"));
    desc.add<edm::InputTag>("OneOEMinusOneOPMap", edm::InputTag(
                                "hltEgammaGsfTrackVars:OneOESuperMinusOneOP"));
    desc.add<edm::InputTag>("EcalPFClusterIsoMap", edm::InputTag("hltEgammaEcalPFClusterIso"));
    desc.add<edm::InputTag>("EleGsfTrackIsoMap", edm::InputTag("hltEgammaEleGsfTrackIso"));
    desc.add<edm::InputTag>("HcalPFClusterIsoMap", edm::InputTag("hltEgammaHcalPFClusterIso"));
    desc.add<double>("egammaPtCut", 4.0);
    desc.add<double>("egammaEtaCut", 2.5);
    desc.add<double>("egammaHoverECut", 1.0);
    descriptions.add("hltScoutingEgammaProducer", desc);
}
