import FWCore.ParameterSet.Config as cms

singleMuonCut = 500
doubleMuonCut = 300
tripleMuonCut = 100
singleElectronCut = 500
doubleElectronCut = 300
tripleElectronCut = 100
singlePhotonCut = 800
doublePhotonCut = 400
triplePhotonCut = 200
singleJetCut = 1500
doubleJetCut = 500
multiJetCut = 100
multiJetNJets = 8
htCut = 4000
dimuonMassCut = 500
dielectronMassCut = 500
diEMuMassCut = 500

#one muon
singleMuonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string( "pt() > "+str(singleMuonCut) )
)
singleMuonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("singleMuonSelector"),
    minNumber = cms.uint32(1)
)
hotlineSkimSingleMuon = cms.Path(singleMuonSelector * singleMuonFilter)

#two muons
doubleMuonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string( "pt() > "+str(doubleMuonCut) )
)
doubleMuonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("doubleMuonSelector"),
    minNumber = cms.uint32(2)
)
hotlineSkimDoubleMuon = cms.Path(doubleMuonSelector * doubleMuonFilter)

#three muons
tripleMuonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("muons"),
    cut = cms.string( "pt() > "+str(tripleMuonCut) )
)
tripleMuonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("tripleMuonSelector"),
    minNumber = cms.uint32(3)
)
hotlineSkimTripleMuon = cms.Path(tripleMuonSelector * tripleMuonFilter)

#one electron
singleElectronSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedGsfElectrons"),
    cut = cms.string( "pt() > "+str(singleElectronCut) )
)
singleElectronFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("singleElectronSelector"),
    minNumber = cms.uint32(1)
)
hotlineSkimSingleElectron = cms.Path(singleElectronSelector * singleElectronFilter)

#two electrons
doubleElectronSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedGsfElectrons"),
    cut = cms.string( "pt() > "+str(doubleElectronCut) )
)
doubleElectronFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("doubleElectronSelector"),
    minNumber = cms.uint32(2)
)
hotlineSkimDoubleElectron = cms.Path(doubleElectronSelector * doubleElectronFilter)

#three electrons
tripleElectronSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedGsfElectrons"),
    cut = cms.string( "pt() > "+str(tripleElectronCut) )
)
tripleElectronFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("tripleElectronSelector"),
    minNumber = cms.uint32(3)
)
hotlineSkimTripleElectron = cms.Path(tripleElectronSelector * tripleElectronFilter)

#one photon
singlePhotonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedPhotons"),
    cut = cms.string( "pt() > "+str(singlePhotonCut) )
)
singlePhotonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("singlePhotonSelector"),
    minNumber = cms.uint32(1)
)
hotlineSkimSinglePhoton = cms.Path(singlePhotonSelector * singlePhotonFilter)

#two photons
doublePhotonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedPhotons"),
    cut = cms.string( "pt() > "+str(doublePhotonCut) )
)
doublePhotonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("doublePhotonSelector"),
    minNumber = cms.uint32(2)
)
hotlineSkimDoublePhoton = cms.Path(doublePhotonSelector * doublePhotonFilter)

#three photons
triplePhotonSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("gedPhotons"),
    cut = cms.string( "pt() > "+str(triplePhotonCut) )
)
triplePhotonFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("triplePhotonSelector"),
    minNumber = cms.uint32(3)
)
hotlineSkimTriplePhoton = cms.Path(triplePhotonSelector * triplePhotonFilter)

#one jet
singleJetSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("ak4PFJets"),
    cut = cms.string( "pt() > "+str(singleJetCut) )
)
singleJetFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("singleJetSelector"),
    minNumber = cms.uint32(1)
)
hotlineSkimSingleJet = cms.Path(singleJetSelector * singleJetFilter)

#two jets
doubleJetSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("ak4PFJets"),
    cut = cms.string( "pt() > "+str(doubleJetCut) )
)
doubleJetFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("doubleJetSelector"),
    minNumber = cms.uint32(2)
)
hotlineSkimDoubleJet = cms.Path(doubleJetSelector * doubleJetFilter)

#many jets
multiJetSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("ak4PFJets"),
    cut = cms.string( "pt() > "+str(multiJetCut) )
)
multiJetFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("multiJetSelector"),
    minNumber = cms.uint32(multiJetNJets)
)
hotlineSkimMultiJet = cms.Path(multiJetSelector * multiJetFilter)

#HT
htMht = cms.EDProducer( "HLTHtMhtProducer",
    usePt = cms.bool( False ),
    minPtJetHt = cms.double( 40.0 ),
    maxEtaJetMht = cms.double( 5.0 ),
    minNJetMht = cms.int32( 0 ),
    jetsLabel = cms.InputTag( "ak4CaloJets" ),
    maxEtaJetHt = cms.double( 3.0 ),
    minPtJetMht = cms.double( 30.0 ),
    minNJetHt = cms.int32( 0 ),
    pfCandidatesLabel = cms.InputTag( "" ),
    excludePFMuons = cms.bool( False )
)
htSelector = cms.EDFilter(
    "CandViewSelector",
    src = cms.InputTag("htMht"),
    cut = cms.string( "sumEt() > "+str(htCut) )
)
htFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("htSelector"),
    minNumber = cms.uint32(1)
)
hotlineSkimHT = cms.Path(htMht * htSelector * htFilter)

#high-mass dileptons
dimuons = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("muons muons"),
    checkCharge = cms.bool(False),
    cut = cms.string("mass > "+str(dimuonMassCut)),
)
dielectrons = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("gedGsfElectrons gedGsfElectrons"),
    checkCharge = cms.bool(False),
    cut = cms.string("mass > "+str(dielectronMassCut)),
)
diEMu = cms.EDProducer(
    "CandViewShallowCloneCombiner",
    decay = cms.string("muons gedGsfElectrons"),
    checkCharge = cms.bool(False),
    cut = cms.string("mass > "+str(diEMuMassCut)),
)
dimuonMassFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("dimuons"),
    minNumber = cms.uint32(1)
)
dielectronMassFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("dielectrons"),
    minNumber = cms.uint32(1)
)
diEMuMassFilter = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("diEMu"),
    minNumber = cms.uint32(1)
)

hotlineSkimMassiveDimuon = cms.Path(dimuons * dimuonMassFilter)
hotlineSkimMassiveDielectron = cms.Path(dielectrons * dielectronMassFilter)
hotlineSkimMassiveEMu = cms.Path(diEMu * diEMuMassFilter)
