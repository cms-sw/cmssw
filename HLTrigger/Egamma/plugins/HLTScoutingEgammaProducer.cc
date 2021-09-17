// -*- C++ -*-
//
// Package:    HLTrigger/Egamma
// Class:      HLTScoutingEgammaProducer
//
/**\class HLTScoutingEgammaProducer HLTScoutingEgammaProducer.cc HLTrigger/Egamma/src/HLTScoutingEgammaProducer.cc

Description: Producer for Run3ScoutingElectron and Run3ScoutingPhoton

*/
//
// Original Author:  David G. Sheffield (Rutgers)
//         Created:  Mon, 20 Jul 2015
//
//

#include "HLTScoutingEgammaProducer.h"

// function to find rechhit associated to detid and return energy
float recHitE(const DetId id, const EcalRecHitCollection& recHits) {
  if (id == DetId(0)) {
    return 0;
  } else {
    EcalRecHitCollection::const_iterator it = recHits.find(id);
    if (it != recHits.end())
      return (*it).energy();
  }
  return 0;
}

float recHitT(const DetId id, const EcalRecHitCollection& recHits) {
  if (id == DetId(0)) {
    return 0;
  } else {
    EcalRecHitCollection::const_iterator it = recHits.find(id);
    if (it != recHits.end())
      return (*it).time();
  }
  return 0;
}

//
// constructors and destructor
//
HLTScoutingEgammaProducer::HLTScoutingEgammaProducer(const edm::ParameterSet& iConfig)
    : EgammaCandidateCollection_(
          consumes<reco::RecoEcalCandidateCollection>(iConfig.getParameter<edm::InputTag>("EgammaCandidates"))),
      EgammaGsfTrackCollection_(
          consumes<reco::GsfTrackCollection>(iConfig.getParameter<edm::InputTag>("EgammaGsfTracks"))),
      SigmaIEtaIEtaMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("SigmaIEtaIEtaMap"))),
      R9Map_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("r9Map"))),
      HoverEMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("HoverEMap"))),
      DetaMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("DetaMap"))),
      DphiMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("DphiMap"))),
      MissingHitsMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("MissingHitsMap"))),
      OneOEMinusOneOPMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("OneOEMinusOneOPMap"))),
      EcalPFClusterIsoMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("EcalPFClusterIsoMap"))),
      EleGsfTrackIsoMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("EleGsfTrackIsoMap"))),
      HcalPFClusterIsoMap_(consumes<RecoEcalCandMap>(iConfig.getParameter<edm::InputTag>("HcalPFClusterIsoMap"))),
      egammaPtCut(iConfig.getParameter<double>("egammaPtCut")),
      egammaEtaCut(iConfig.getParameter<double>("egammaEtaCut")),
      egammaHoverECut(iConfig.getParameter<double>("egammaHoverECut")),
      mantissaPrecision(iConfig.getParameter<int>("mantissaPrecision")),
      saveRecHitTiming(iConfig.getParameter<bool>("saveRecHitTiming")),
      rechitMatrixSize(iConfig.getParameter<int>("rechitMatrixSize")),  //(2n+1)^2
      ecalRechitEB_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRechitEB"))),
      ecalRechitEE_(consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecalRechitEE"))) {
  //register products
  produces<Run3ScoutingElectronCollection>();
  produces<Run3ScoutingPhotonCollection>();
  topologyToken_ = esConsumes();
}

HLTScoutingEgammaProducer::~HLTScoutingEgammaProducer() = default;

// ------------ method called to produce the data  ------------
void HLTScoutingEgammaProducer::produce(edm::StreamID sid, edm::Event& iEvent, edm::EventSetup const& setup) const {
  using namespace edm;

  auto outElectrons = std::make_unique<Run3ScoutingElectronCollection>();
  auto outPhotons = std::make_unique<Run3ScoutingPhotonCollection>();

  // Get RecoEcalCandidate
  Handle<reco::RecoEcalCandidateCollection> EgammaCandidateCollection;
  if (!iEvent.getByToken(EgammaCandidateCollection_, EgammaCandidateCollection)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get GsfTrack
  Handle<reco::GsfTrackCollection> EgammaGsfTrackCollection;
  if (!iEvent.getByToken(EgammaGsfTrackCollection_, EgammaGsfTrackCollection)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get SigmaIEtaIEtaMap
  Handle<RecoEcalCandMap> SigmaIEtaIEtaMap;
  if (!iEvent.getByToken(SigmaIEtaIEtaMap_, SigmaIEtaIEtaMap)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  Handle<RecoEcalCandMap> R9Map;
  if (!iEvent.getByToken(R9Map_, R9Map)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get HoverEMap
  Handle<RecoEcalCandMap> HoverEMap;
  if (!iEvent.getByToken(HoverEMap_, HoverEMap)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get DetaMap
  Handle<RecoEcalCandMap> DetaMap;
  if (!iEvent.getByToken(DetaMap_, DetaMap)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get DphiMap
  Handle<RecoEcalCandMap> DphiMap;
  if (!iEvent.getByToken(DphiMap_, DphiMap)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get MissingHitsMap
  Handle<RecoEcalCandMap> MissingHitsMap;
  if (!iEvent.getByToken(MissingHitsMap_, MissingHitsMap)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get 1/E - 1/p Map
  Handle<RecoEcalCandMap> OneOEMinusOneOPMap;
  if (!iEvent.getByToken(OneOEMinusOneOPMap_, OneOEMinusOneOPMap)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get EcalPFClusterIsoMap
  Handle<RecoEcalCandMap> EcalPFClusterIsoMap;
  if (!iEvent.getByToken(EcalPFClusterIsoMap_, EcalPFClusterIsoMap)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get EleGsfTrackIsoMap
  Handle<RecoEcalCandMap> EleGsfTrackIsoMap;
  if (!iEvent.getByToken(EleGsfTrackIsoMap_, EleGsfTrackIsoMap)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  // Get HcalPFClusterIsoMap
  Handle<RecoEcalCandMap> HcalPFClusterIsoMap;
  if (!iEvent.getByToken(HcalPFClusterIsoMap_, HcalPFClusterIsoMap)) {
    iEvent.put(std::move(outElectrons));
    iEvent.put(std::move(outPhotons));
    return;
  }

  edm::Handle<EcalRecHitCollection> rechitsEB;
  edm::Handle<EcalRecHitCollection> rechitsEE;
  iEvent.getByToken(ecalRechitEB_, rechitsEB);
  iEvent.getByToken(ecalRechitEE_, rechitsEE);

  const CaloTopology* topology = &setup.getData(topologyToken_);

  // Produce electrons and photons
  int index = 0;
  for (auto& candidate : *EgammaCandidateCollection) {
    reco::RecoEcalCandidateRef candidateRef = getRef(EgammaCandidateCollection, index);
    ++index;
    if (candidateRef.isNull() && !candidateRef.isAvailable())
      continue;

    if (candidate.pt() < egammaPtCut)
      continue;
    if (fabs(candidate.eta()) > egammaEtaCut)
      continue;

    reco::SuperClusterRef scRef = candidate.superCluster();
    if (scRef.isNull() && !scRef.isAvailable())
      continue;

    reco::CaloClusterPtr SCseed = candidate.superCluster()->seed();
    const EcalRecHitCollection* rechits = (std::abs(scRef->eta()) < 1.479) ? rechitsEB.product() : rechitsEE.product();
    Cluster2ndMoments moments = EcalClusterTools::cluster2ndMoments(*SCseed, *rechits);
    float sMin = moments.sMin;
    float sMaj = moments.sMaj;

    unsigned int seedId = (*SCseed).seed();

    std::vector<DetId> mDetIds = EcalClusterTools::matrixDetId((topology), (*SCseed).seed(), rechitMatrixSize);

    int detSize = mDetIds.size();
    std::vector<float> mEnergies(detSize, 0.);
    std::vector<float> mTimes(detSize, 0.);

    for (int i = 0; i < detSize; i++) {
      mEnergies[i] =
          MiniFloatConverter::reduceMantissaToNbitsRounding(recHitE(mDetIds.at(i), *rechits), mantissaPrecision);
      if (saveRecHitTiming)
        mTimes[i] =
            MiniFloatConverter::reduceMantissaToNbitsRounding(recHitT(mDetIds.at(i), *rechits), mantissaPrecision);
    }

    float HoE = 999.;
    if (candidate.superCluster()->energy() != 0.)
      HoE = (*HoverEMap)[candidateRef] / candidate.superCluster()->energy();

    float d0 = 0.0;
    float dz = 0.0;
    int charge = -999;
    for (auto& track : *EgammaGsfTrackCollection) {
      RefToBase<TrajectorySeed> seed = track.extra()->seedRef();
      reco::ElectronSeedRef elseed = seed.castTo<reco::ElectronSeedRef>();
      RefToBase<reco::CaloCluster> caloCluster = elseed->caloCluster();
      reco::SuperClusterRef scRefFromTrk = caloCluster.castTo<reco::SuperClusterRef>();
      if (scRefFromTrk == scRef) {
        d0 = track.d0();
        dz = track.dz();
        charge = track.charge();
      }
    }
    if (charge == -999) {  // No associated track. Candidate is a scouting photon
      outPhotons->emplace_back(candidate.pt(),
                               candidate.eta(),
                               candidate.phi(),
                               candidate.mass(),
                               (*SigmaIEtaIEtaMap)[candidateRef],
                               HoE,
                               (*EcalPFClusterIsoMap)[candidateRef],
                               (*HcalPFClusterIsoMap)[candidateRef],
                               0.,
                               (*R9Map)[candidateRef],
                               sMin,
                               sMaj,
                               seedId,
                               mEnergies,
                               mTimes);  //read for(ieta){for(iphi){}}
    } else {                             // Candidate is a scouting electron
      outElectrons->emplace_back(candidate.pt(),
                                 candidate.eta(),
                                 candidate.phi(),
                                 candidate.mass(),
                                 d0,
                                 dz,
                                 (*DetaMap)[candidateRef],
                                 (*DphiMap)[candidateRef],
                                 (*SigmaIEtaIEtaMap)[candidateRef],
                                 HoE,
                                 (*OneOEMinusOneOPMap)[candidateRef],
                                 (*MissingHitsMap)[candidateRef],
                                 charge,
                                 (*EcalPFClusterIsoMap)[candidateRef],
                                 (*HcalPFClusterIsoMap)[candidateRef],
                                 (*EleGsfTrackIsoMap)[candidateRef],
                                 (*R9Map)[candidateRef],
                                 sMin,
                                 sMaj,
                                 seedId,
                                 mEnergies,
                                 mTimes);  //read for(ieta){for(iphi){}}
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
  desc.add<edm::InputTag>("SigmaIEtaIEtaMap", edm::InputTag("hltEgammaClusterShape:sigmaIEtaIEta5x5"));
  desc.add<edm::InputTag>("r9Map", edm::InputTag("hltEgammaR9ID:r95x5"));
  desc.add<edm::InputTag>("HoverEMap", edm::InputTag("hltEgammaHoverE"));
  desc.add<edm::InputTag>("DetaMap", edm::InputTag("hltEgammaGsfTrackVars:DetaSeed"));
  desc.add<edm::InputTag>("DphiMap", edm::InputTag("hltEgammaGsfTrackVars:Dphi"));
  desc.add<edm::InputTag>("MissingHitsMap", edm::InputTag("hltEgammaGsfTrackVars:MissingHits"));
  desc.add<edm::InputTag>("OneOEMinusOneOPMap", edm::InputTag("hltEgammaGsfTrackVars:OneOESuperMinusOneOP"));
  desc.add<edm::InputTag>("EcalPFClusterIsoMap", edm::InputTag("hltEgammaEcalPFClusterIso"));
  desc.add<edm::InputTag>("EleGsfTrackIsoMap", edm::InputTag("hltEgammaEleGsfTrackIso"));
  desc.add<edm::InputTag>("HcalPFClusterIsoMap", edm::InputTag("hltEgammaHcalPFClusterIso"));
  desc.add<double>("egammaPtCut", 4.0);
  desc.add<double>("egammaEtaCut", 2.5);
  desc.add<double>("egammaHoverECut", 1.0);
  desc.add<bool>("saveRecHitTiming", false);
  desc.add<int>("mantissaPrecision", 10)->setComment("default float16, change to 23 for float32");
  desc.add<int>("rechitMatrixSize", 10);
  desc.add<edm::InputTag>("ecalRechitEB", edm::InputTag("hltEcalRecHit:EcalRecHitsEB"));
  desc.add<edm::InputTag>("ecalRechitEE", edm::InputTag("hltEcalRecHit:EcalRecHitsEE"));
  descriptions.add("hltScoutingEgammaProducer", desc);
}

// declare this class as a framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTScoutingEgammaProducer);
